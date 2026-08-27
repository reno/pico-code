// Package agent implements the core loop: call the provider, and if its
// reply carries no tool calls, hand back the text; otherwise run every tool
// call from that reply, feed the results back, and repeat. It owns policy
// (CLAUDE.md layer ownership) — providers only translate wire formats and
// tools only execute; the loop decides when to stop and how results get
// reassembled.
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

// Guards bundles the loop's stopping conditions beyond "the model stopped
// asking for tools" — CLAUDE.md layer ownership: the agent loop owns policy
// (max turns, token budget, timeouts, loop detection), not the provider or
// tools. A zero value in any field means that guard is disabled. Repetition
// detection has no field: CLAUDE.md fixes it at 3 identical consecutive
// tool calls, not a per-session tuning knob.
type Guards struct {
	MaxTurns    int           // provider.Chat calls allowed before stopping; 0 = unlimited.
	TokenBudget int           // cumulative Usage tokens allowed before stopping; 0 = unlimited.
	WallClock   time.Duration // elapsed time since Run started before stopping; 0 = unlimited.
}

// repetitionThreshold is CLAUDE.md's fixed number of identical consecutive
// tool calls (same tool, same arguments, round over round) the loop
// tolerates before breaking — small local models are documented to repeat a
// call forever rather than give up.
const repetitionThreshold = 3

// Agent drives one conversation: a provider to talk to, a tool registry to
// satisfy its tool calls, and the history both read from and append to.
type Agent struct {
	provider  llm.Provider
	tools     *tools.Registry
	history   *history.History
	system    string
	maxTokens int
	guards    Guards
}

// New returns an Agent ready to run turns against provider, using registry
// to execute any tool call it issues, appending to h. system and maxTokens
// are sent on every Request the loop builds; guards bounds how long a
// single Run can keep asking the provider for more tool calls.
func New(provider llm.Provider, registry *tools.Registry, h *history.History, system string, maxTokens int, guards Guards) *Agent {
	return &Agent{provider: provider, tools: registry, history: h, system: system, maxTokens: maxTokens, guards: guards}
}

// Run appends userInput to history as a user turn, then drives the
// call-provider / run-tools cycle to completion: each round appends the
// provider's reply to history, and if that reply has no ToolUse blocks,
// Run returns its text. Otherwise every ToolUse in that reply runs
// concurrently, the results are appended — in the original call order,
// regardless of which tool finished first — as the next user message.
//
// After each such round, if a guard has tripped (see Guards), the loop
// stops without making another provider call: it appends one final
// assistant Text message explaining why and returns that text. The
// triggering round's tool calls are always answered first, so history
// stays valid (CLAUDE.md invariant 3) no matter which guard trips.
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

// guardTripped reports whether any configured guard has been exceeded by
// the round that just completed, and if so, the explanation to hand back to
// the user (and the model, via history).
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

// stopWithExplanation appends reason as a final plain-text assistant
// message — not a ToolUse, so it needs no ToolResult and closes out history
// in a valid state — and returns it as Run's result.
func (a *Agent) stopWithExplanation(reason string) string {
	a.history.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: reason}}})
	return reason
}

// runTools executes every call concurrently and returns their ToolResult
// blocks in calls' original order — CLAUDE.md invariant 3 requires results
// answer their ToolUse in call order, independent of completion order.
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolUse) []llm.Block {
	results := make([]llm.Block, len(calls))
	var g errgroup.Group
	for i, call := range calls {
		g.Go(func() error {
			results[i] = a.runTool(ctx, call)
			return nil
		})
	}
	// Every goroutine above always returns nil: a tool failure becomes an
	// IsError ToolResult (CLAUDE.md invariant 4, "tool failures are data,
	// not control flow"), never a Go error that could short-circuit the
	// group and leave a result slot empty.
	_ = g.Wait()
	return results
}

func (a *Agent) runTool(ctx context.Context, call llm.ToolUse) llm.Block {
	out, err := a.tools.Run(ctx, call.Name, call.Input)
	if err != nil {
		return llm.ToolResult{ToolUseID: call.ID, Content: err.Error(), IsError: true}
	}
	return llm.ToolResult{ToolUseID: call.ID, Content: out, IsError: false}
}

// callSignature builds a comparison key for a round's tool calls, in call
// order, so two rounds count as "identical" (for repetition detection) only
// when they call the same tools with the same arguments in the same order.
// Input is JSON-compacted first so whitespace differences between two
// otherwise-identical arguments don't defeat the comparison.
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
