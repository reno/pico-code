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
	"github.com/reno/pico-code/internal/ui"
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
	approver    Approver
}

// New returns an Agent ready to run turns against provider, using registry
// to execute any tool call it issues, appending to h. toolTimeout bounds a
// single tool call (0 = unlimited). approver is consulted before running any
// tools.ApprovalRequired call; pass AutoApprove for --yes.
func New(provider llm.Provider, registry *tools.Registry, h *history.History, system string, maxTokens int, guards Guards, toolTimeout time.Duration, approver Approver) *Agent {
	return &Agent{provider: provider, tools: registry, history: h, system: system, maxTokens: maxTokens, guards: guards, toolTimeout: toolTimeout, approver: approver}
}

// Run appends userInput to history, then drives the call-provider /
// run-tools cycle until a reply has no ToolUse blocks (its text is
// returned) or a guard trips (a final explanatory Text message is appended
// and returned instead). Every round's tool calls are answered before
// either exit, so history always satisfies CLAUDE.md invariant 3.
func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: userInput}}})

	rs := &roundState{start: time.Now()}
	for {
		resp, err := a.provider.Chat(ctx, a.request())
		if err != nil {
			return "", fmt.Errorf("agent: chat: %w", err)
		}
		if text, done := a.runRound(ctx, resp, rs, nil); done {
			return text, nil
		}
	}
}

// RunStream behaves like Run but drives each round's provider call through
// Stream instead of Chat, handing the raw event channel to renderer so it
// can display text as it arrives. If renderer also implements
// ui.ToolStatusReporter, it additionally learns about each tool call's
// start and outcome — a step Render alone can't see, since running a tool
// happens after a round's events finish, not during them.
func (a *Agent) RunStream(ctx context.Context, userInput string, renderer ui.Renderer) (string, error) {
	a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: userInput}}})

	reporter, _ := renderer.(ui.ToolStatusReporter)
	rs := &roundState{start: time.Now()}
	for {
		ch, err := a.provider.Stream(ctx, a.request())
		if err != nil {
			return "", fmt.Errorf("agent: stream: %w", err)
		}
		resp, err := renderer.Render(ctx, ch)
		if err != nil {
			return "", fmt.Errorf("agent: render: %w", err)
		}
		if text, done := a.runRound(ctx, resp, rs, reporter); done {
			return text, nil
		}
	}
}

func (a *Agent) request() llm.Request {
	return llm.Request{
		System:    a.system,
		Messages:  a.history.Snapshot(),
		Tools:     a.tools.Definitions(),
		MaxTokens: a.maxTokens,
	}
}

// roundState carries the bookkeeping Run and RunStream both accumulate
// across rounds of the same turn.
type roundState struct {
	start              time.Time
	turns, totalTokens int
	repeatStreak       int
	lastSignature      string
}

// runRound appends resp to history, runs any tool calls it contains, and
// reports whether the turn is over — either because resp had no tool calls
// or because a guard tripped — along with the text to return in that case.
func (a *Agent) runRound(ctx context.Context, resp *llm.Response, rs *roundState, reporter ui.ToolStatusReporter) (string, bool) {
	rs.turns++
	rs.totalTokens += resp.Usage.InputTokens + resp.Usage.OutputTokens
	a.history.Append(resp.Message)

	calls := toolUseBlocks(resp.Message)
	if len(calls) == 0 {
		return textOf(resp.Message), true
	}

	if sig := callSignature(calls); sig == rs.lastSignature {
		rs.repeatStreak++
	} else {
		rs.lastSignature, rs.repeatStreak = sig, 1
	}

	a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: a.runTools(ctx, calls, reporter)})

	if reason, tripped := a.guardTripped(rs.turns, rs.totalTokens, time.Since(rs.start), rs.repeatStreak); tripped {
		return a.stopWithExplanation(reason), true
	}
	return "", false
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
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolUse, reporter ui.ToolStatusReporter) []llm.Block {
	results := make([]llm.Block, len(calls))
	var g errgroup.Group
	for i, call := range calls {
		g.Go(func() error {
			results[i] = a.runTool(ctx, call, reporter)
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
func (a *Agent) runTool(ctx context.Context, call llm.ToolUse, reporter ui.ToolStatusReporter) llm.Block {
	if result, denied := a.checkApproval(ctx, call); denied {
		return result
	}

	if reporter != nil {
		reporter.ToolStarted(call.ID, call.Name, call.Input)
	}
	result := a.doRunTool(ctx, call)
	if reporter != nil {
		tr := result.(llm.ToolResult)
		reporter.ToolFinished(call.ID, call.Name, tr.Content, tr.IsError)
	}
	return result
}

func (a *Agent) doRunTool(ctx context.Context, call llm.ToolUse) llm.Block {
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

// checkApproval reports whether call was denied, in which case the
// returned ToolResult should be used as-is instead of running the tool.
func (a *Agent) checkApproval(ctx context.Context, call llm.ToolUse) (llm.Block, bool) {
	t, err := a.tools.Get(call.Name)
	if err != nil {
		return nil, false
	}
	ar, ok := t.(tools.ApprovalRequired)
	if !ok || !ar.NeedsApproval() {
		return nil, false
	}

	preview := ""
	if p, ok := t.(tools.Previewable); ok {
		if text, err := p.Preview(ctx, call.Input); err == nil {
			preview = text
		} else {
			preview = fmt.Sprintf("(preview unavailable: %v)", err)
		}
	}

	approved, err := a.approver.Approve(ctx, call.Name, call.Input, preview)
	if err != nil {
		return llm.ToolResult{ToolUseID: call.ID, Content: fmt.Sprintf("approval error: %v", err), IsError: true}, true
	}
	if !approved {
		return llm.ToolResult{ToolUseID: call.ID, Content: fmt.Sprintf("tool %q call denied by user", call.Name), IsError: true}, true
	}
	return nil, false
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
