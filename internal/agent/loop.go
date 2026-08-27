// Package agent implements the core loop: call the provider, run any tool
// calls it returns, feed the results back, and repeat until it stops asking.
package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// Guards bounds how long a single Run can keep asking the provider for more
// tool calls. A zero field disables that guard. Repetition detection has no
// field: CLAUDE.md fixes it at 3 identical consecutive tool calls.
type Guards struct {
	MaxTurns    int
	TokenBudget int
	WallClock   time.Duration
}

const repetitionThreshold = 3

// Agent drives one conversation: a provider to talk to, a tool registry to
// satisfy its tool calls, and the history both read from and append to.
type Agent struct {
	provider    llm.Provider
	tools       *tools.Registry
	history     *history.History
	system      string
	maxTokens   int
	guards      Guards
	toolTimeout time.Duration
}

// New returns an Agent ready to run turns against provider, using registry
// to execute any tool call it issues, appending to h. toolTimeout bounds a
// single tool call (0 = unlimited).
func New(provider llm.Provider, registry *tools.Registry, h *history.History, system string, maxTokens int, guards Guards, toolTimeout time.Duration) *Agent {
	return &Agent{provider: provider, tools: registry, history: h, system: system, maxTokens: maxTokens, guards: guards, toolTimeout: toolTimeout}
}

// Run appends userInput to history, then drives the call-provider /
// run-tools cycle until a reply has no ToolUse blocks (its text is
// returned) or a guard trips (a final explanatory Text message is appended
// and returned instead). Every round's tool calls are answered before
// either exit, so history always satisfies CLAUDE.md invariant 3.
func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: userInput}}})

	start := time.Now()
	var turns, totalTokens, repeatStreak int
	var lastSignature string

	for {
		resp, err := a.provider.Chat(ctx, llm.Request{
			System:    a.system,
			Messages:  a.history.Snapshot(),
			Tools:     a.tools.Definitions(),
			MaxTokens: a.maxTokens,
		})
		if err != nil {
			return "", fmt.Errorf("agent: chat: %w", err)
		}
		turns++
		totalTokens += resp.Usage.InputTokens + resp.Usage.OutputTokens
		a.history.Append(resp.Message)

		calls := toolUseBlocks(resp.Message)
		if len(calls) == 0 {
			return textOf(resp.Message), nil
		}

		if sig := callSignature(calls); sig == lastSignature {
			repeatStreak++
		} else {
			lastSignature, repeatStreak = sig, 1
		}

		a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: a.runTools(ctx, calls)})

		if reason, tripped := a.guardTripped(turns, totalTokens, time.Since(start), repeatStreak); tripped {
			return a.stopWithExplanation(reason), nil
		}
	}
}

func (a *Agent) guardTripped(turns, totalTokens int, elapsed time.Duration, repeatStreak int) (string, bool) {
	switch {
	case a.guards.MaxTurns > 0 && turns >= a.guards.MaxTurns:
		return fmt.Sprintf("Stopping: reached the maximum of %d turn(s).", a.guards.MaxTurns), true
	case a.guards.TokenBudget > 0 && totalTokens >= a.guards.TokenBudget:
		return fmt.Sprintf("Stopping: reached the token budget of %d.", a.guards.TokenBudget), true
	case a.guards.WallClock > 0 && elapsed >= a.guards.WallClock:
		return fmt.Sprintf("Stopping: exceeded the wall-clock timeout of %s.", a.guards.WallClock), true
	case repeatStreak >= repetitionThreshold:
		return fmt.Sprintf("Stopping: the same tool call repeated %d times in a row.", repeatStreak), true
	default:
		return "", false
	}
}

func (a *Agent) stopWithExplanation(reason string) string {
	a.history.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: reason}}})
	return reason
}

// runTools executes every call concurrently and returns their ToolResult
// blocks in calls' original order, independent of completion order.
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolUse) []llm.Block {
	results := make([]llm.Block, len(calls))
	var g errgroup.Group
	for i, call := range calls {
		g.Go(func() error {
			results[i] = a.runTool(ctx, call)
			return nil
		})
	}
	_ = g.Wait()
	return results
}

type toolOutcome struct {
	out string
	err error
}

// runTool always produces a ToolResult, even if the tool panics, times out,
// or ctx is cancelled while it's running. The tool runs in its own
// goroutine so a badly-behaved implementation that ignores ctx can't block
// this call past toolTimeout or cancellation; if the deadline wins the race,
// the goroutine is simply abandoned rather than waited on.
func (a *Agent) runTool(ctx context.Context, call llm.ToolUse) llm.Block {
	toolCtx := ctx
	if a.toolTimeout > 0 {
		var cancel context.CancelFunc
		toolCtx, cancel = context.WithTimeout(ctx, a.toolTimeout)
		defer cancel()
	}

	done := make(chan toolOutcome, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- toolOutcome{err: fmt.Errorf("tool %q panicked: %v", call.Name, r)}
			}
		}()
		out, err := a.tools.Run(toolCtx, call.Name, call.Input)
		done <- toolOutcome{out: out, err: err}
	}()

	select {
	case o := <-done:
		if o.err != nil {
			return llm.ToolResult{ToolUseID: call.ID, Content: o.err.Error(), IsError: true}
		}
		return llm.ToolResult{ToolUseID: call.ID, Content: o.out, IsError: false}
	case <-toolCtx.Done():
		return llm.ToolResult{ToolUseID: call.ID, Content: fmt.Sprintf("tool %q: %v", call.Name, toolCtx.Err()), IsError: true}
	}
}

func callSignature(calls []llm.ToolUse) string {
	parts := make([]string, len(calls))
	for i, c := range calls {
		parts[i] = c.Name + ":" + compactJSON(c.Input)
	}
	return strings.Join(parts, "|")
}

func compactJSON(raw json.RawMessage) string {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return string(raw)
	}
	return buf.String()
}

func toolUseBlocks(m llm.Message) []llm.ToolUse {
	var calls []llm.ToolUse
	for _, b := range m.Blocks {
		if tu, ok := b.(llm.ToolUse); ok {
			calls = append(calls, tu)
		}
	}
	return calls
}

func textOf(m llm.Message) string {
	var b strings.Builder
	for _, blk := range m.Blocks {
		t, ok := blk.(llm.Text)
		if !ok {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(t.Text)
	}
	return b.String()
}
