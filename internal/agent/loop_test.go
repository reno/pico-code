package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
	"github.com/reno/pico-code/internal/ui"
)

// scriptedProvider returns responses in order, one per Chat call — the
// "scripted fake provider" 4.1's AC calls for.
type scriptedProvider struct {
	responses []*llm.Response
	calls     int
}

func (s *scriptedProvider) Name() string { return "scripted" }

func (s *scriptedProvider) Chat(ctx context.Context, _ llm.Request) (*llm.Response, error) {
	// A real adapter checks ctx before/while talking to the network
	// (CLAUDE.md invariant 6); mirroring that here is what lets a
	// cancellation test observe Run exiting cleanly via this path rather
	// than via scriptedProvider running out of scripted responses.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.calls >= len(s.responses) {
		return nil, fmt.Errorf("scripted: no response scripted for call %d", s.calls+1)
	}
	resp := s.responses[s.calls]
	s.calls++
	return resp, nil
}

// Stream reuses the same scripted responses as Chat, turning each one into
// the event sequence llm.CollectStream would reassemble back into it, so
// RunStream tests can script conversations the same way Run tests do.
func (s *scriptedProvider) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Event, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.calls >= len(s.responses) {
		return nil, fmt.Errorf("scripted: no response scripted for call %d", s.calls+1)
	}
	resp := s.responses[s.calls]
	s.calls++
	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		for _, b := range resp.Message.Blocks {
			switch v := b.(type) {
			case llm.Text:
				if !send(llm.TextDelta(v)) {
					return
				}
			case llm.ToolUse:
				if !send(llm.ToolUseStart{ID: v.ID, Name: v.Name}) {
					return
				}
				if !send(llm.ToolUseDone{ID: v.ID, Input: v.Input}) {
					return
				}
			}
		}
		send(llm.MessageDone{StopReason: resp.StopReason, Usage: resp.Usage})
	}), nil
}

// delayedTool sleeps for delay before recording its own name to order (under
// mu) and returning result. Delays let the test prove the loop reassembles
// results in call order even when the slower call was issued first.
type delayedTool struct {
	name   string
	delay  time.Duration
	result string
	mu     *sync.Mutex
	order  *[]string
}

func (d *delayedTool) Name() string            { return d.name }
func (d *delayedTool) Description() string     { return "a delayed test tool" }
func (d *delayedTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }

func (d *delayedTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	time.Sleep(d.delay)
	d.mu.Lock()
	*d.order = append(*d.order, d.name)
	d.mu.Unlock()
	return d.result, nil
}

var _ tools.Tool = (*delayedTool)(nil)

// TestRunDrivesScriptedConversationWithParallelTools is 4.1's AC: a scripted
// fake provider drives a 3-round conversation with 2 parallel tools; the
// final history passes Validate(); result order is deterministic.
func TestRunDrivesScriptedConversationWithParallelTools(t *testing.T) {
	var mu sync.Mutex
	var completionOrder []string

	slow := &delayedTool{name: "slow_tool", delay: 30 * time.Millisecond, result: "slow result", mu: &mu, order: &completionOrder}
	fast := &delayedTool{name: "fast_tool", delay: 0, result: "fast result", mu: &mu, order: &completionOrder}

	reg := tools.NewRegistry()
	if err := reg.Register(slow); err != nil {
		t.Fatalf("Register(slow) error = %v", err)
	}
	if err := reg.Register(fast); err != nil {
		t.Fatalf("Register(fast) error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		{
			// Round 1: two parallel calls. slow_tool is called first but
			// finishes last, so a naive sequential-append implementation
			// would still happen to get this right — the completion-order
			// assertion below is what actually catches a bug here.
			Message: llm.Message{
				Role: llm.RoleAssistant,
				Blocks: []llm.Block{
					llm.ToolUse{ID: "call_1", Name: "slow_tool", Input: json.RawMessage(`{}`)},
					llm.ToolUse{ID: "call_2", Name: "fast_tool", Input: json.RawMessage(`{}`)},
				},
			},
			StopReason: "tool_use",
		},
		{
			// Round 2: one more tool call.
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: "call_3", Name: "fast_tool", Input: json.RawMessage(`{}`)}},
			},
			StopReason: "tool_use",
		},
		{
			// Round 3: final text, no tool calls.
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "done"}}},
			StopReason: "end_turn",
		},
	}}

	h := history.New()
	a := New(provider, reg, h, "you are a test agent", 1024, Guards{}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "please help")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "done" {
		t.Errorf("Run() = %q, want %q", got, "done")
	}
	if provider.calls != 3 {
		t.Errorf("provider was called %d times, want 3", provider.calls)
	}

	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	mu.Lock()
	order := append([]string(nil), completionOrder...)
	mu.Unlock()
	want := []string{"fast_tool", "slow_tool", "fast_tool"}
	if diff := cmp.Diff(want, order); diff != "" {
		t.Errorf("tool completion order (-want +got):\n%s\n(fast_tool completing before the slow_tool it was issued after proves the round-1 pair ran concurrently)", diff)
	}

	msgs := h.Snapshot()
	if len(msgs) != 6 {
		t.Fatalf("history has %d messages, want 6 (user, assistant, user, assistant, user, assistant)", len(msgs))
	}

	round1Results := msgs[2]
	if round1Results.Role != llm.RoleUser {
		t.Fatalf("messages[2].Role = %q, want %q", round1Results.Role, llm.RoleUser)
	}
	if len(round1Results.Blocks) != 2 {
		t.Fatalf("messages[2] has %d blocks, want 2", len(round1Results.Blocks))
	}
	gotIDs := []string{
		round1Results.Blocks[0].(llm.ToolResult).ToolUseID,
		round1Results.Blocks[1].(llm.ToolResult).ToolUseID,
	}
	wantIDs := []string{"call_1", "call_2"}
	if diff := cmp.Diff(wantIDs, gotIDs); diff != "" {
		t.Errorf("round-1 ToolResult order (-want +got):\n%s\n(must match call order, not completion order)", diff)
	}
	if got := round1Results.Blocks[0].(llm.ToolResult).Content; got != "slow result" {
		t.Errorf("ToolResult for call_1 = %q, want %q", got, "slow result")
	}
	if got := round1Results.Blocks[1].(llm.ToolResult).Content; got != "fast result" {
		t.Errorf("ToolResult for call_2 = %q, want %q", got, "fast result")
	}
}

func TestRunReturnsTextImmediatelyWhenNoToolCall(t *testing.T) {
	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "hello there"}}},
			StopReason: "end_turn",
		},
	}}

	h := history.New()
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "hello there" {
		t.Errorf("Run() = %q, want %q", got, "hello there")
	}
	if provider.calls != 1 {
		t.Errorf("provider was called %d times, want 1", provider.calls)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestRunTurnsUnknownToolIntoErrorResult(t *testing.T) {
	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "does_not_exist", Input: json.RawMessage(`{}`)}},
			},
			StopReason: "tool_use",
		},
		{
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "gave up"}}},
			StopReason: "end_turn",
		},
	}}

	h := history.New()
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)

	if _, err := a.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	msgs := h.Snapshot()
	result := msgs[2].Blocks[0].(llm.ToolResult)
	if !result.IsError {
		t.Error("ToolResult.IsError = false, want true for an unknown tool")
	}
	if result.ToolUseID != "call_1" {
		t.Errorf("ToolResult.ToolUseID = %q, want %q", result.ToolUseID, "call_1")
	}
}

func TestRunPropagatesProviderError(t *testing.T) {
	provider := &scriptedProvider{responses: nil}
	h := history.New()
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)

	if _, err := a.Run(context.Background(), "hi"); err == nil {
		t.Fatal("Run() error = nil, want the provider's error propagated")
	}
}

// echoTool is a minimal registrable tool for guard tests, which only care
// that a ToolUse gets answered, not what the answer is.
type echoTool struct{}

func (echoTool) Name() string                                             { return "echo_tool" }
func (echoTool) Description() string                                      { return "a no-op test tool" }
func (echoTool) Schema() json.RawMessage                                  { return json.RawMessage(`{"type":"object"}`) }
func (echoTool) Run(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil }

var _ tools.Tool = echoTool{}

func toolCallResponse(id string) *llm.Response {
	return toolCallResponseNamed(id, "echo_tool")
}

func toolCallResponseNamed(id, name string) *llm.Response {
	return &llm.Response{
		Message: llm.Message{
			Role:   llm.RoleAssistant,
			Blocks: []llm.Block{llm.ToolUse{ID: id, Name: name, Input: json.RawMessage(`{}`)}},
		},
		StopReason: "tool_use",
	}
}

func textResponse(text string) *llm.Response {
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: text}}},
		StopReason: "end_turn",
	}
}

func assertFinalMessageIsExplanation(t *testing.T, h *history.History) {
	t.Helper()
	msgs := h.Snapshot()
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant || len(last.Blocks) != 1 {
		t.Fatalf("final message = %+v, want a single-block assistant explanation", last)
	}
	if _, ok := last.Blocks[0].(llm.Text); !ok {
		t.Errorf("final message block = %T, want llm.Text", last.Blocks[0])
	}
}

// TestRunStopsAtMaxTurnsGuard is 4.2's AC for the max-turns guard: it trips
// exactly once and leaves valid history.
func TestRunStopsAtMaxTurnsGuard(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	// A model that would keep asking for tool calls forever; only 2
	// responses are scripted, so if the guard doesn't trip after turn 2,
	// the loop's 3rd Chat call fails with scriptedProvider's own error and
	// this test's assertions below catch it.
	provider := &scriptedProvider{responses: []*llm.Response{toolCallResponse("call_1"), toolCallResponse("call_2")}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{MaxTurns: 2}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "keep going forever")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if provider.calls != 2 {
		t.Errorf("provider was called %d times, want exactly 2", provider.calls)
	}
	if !strings.Contains(got, "maximum of 2 turn") {
		t.Errorf("Run() = %q, want an explanation mentioning the turn limit", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	assertFinalMessageIsExplanation(t, h)
}

// TestRunStopsAtTokenBudgetGuard is 4.2's AC for the token-budget guard.
func TestRunStopsAtTokenBudgetGuard(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	round := func(id string, tokens int) *llm.Response {
		resp := toolCallResponse(id)
		resp.Usage = llm.Usage{InputTokens: tokens}
		return resp
	}
	// 10 tokens/round: budget 15 is under cumulative 20 (after round 2) but
	// over 10 (after round 1), so the guard should trip after exactly 2
	// rounds, not 1.
	provider := &scriptedProvider{responses: []*llm.Response{round("call_1", 10), round("call_2", 10)}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{TokenBudget: 15}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "keep going forever")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if provider.calls != 2 {
		t.Errorf("provider was called %d times, want exactly 2", provider.calls)
	}
	if !strings.Contains(got, "token budget") {
		t.Errorf("Run() = %q, want an explanation mentioning the token budget", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	assertFinalMessageIsExplanation(t, h)
}

// TestRunStopsAtWallClockGuard is 4.2's AC for the wall-clock guard: a tool
// slower than the configured budget trips it after a single round.
func TestRunStopsAtWallClockGuard(t *testing.T) {
	reg := tools.NewRegistry()
	slow := &delayedTool{name: "slow_tool", delay: 30 * time.Millisecond, result: "ok", mu: &sync.Mutex{}, order: &[]string{}}
	if err := reg.Register(slow); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "slow_tool", Input: json.RawMessage(`{}`)}},
			},
			StopReason: "tool_use",
		},
	}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{WallClock: 15 * time.Millisecond}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "keep going forever")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if provider.calls != 1 {
		t.Errorf("provider was called %d times, want exactly 1", provider.calls)
	}
	if !strings.Contains(got, "wall-clock timeout") {
		t.Errorf("Run() = %q, want an explanation mentioning the wall-clock timeout", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	assertFinalMessageIsExplanation(t, h)
}

// TestRunStopsAtRepetitionGuard is 4.2's AC for repetition detection:
// CLAUDE.md's fixed threshold of 3 identical consecutive tool calls.
func TestRunStopsAtRepetitionGuard(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	identical := func(id string) *llm.Response {
		return &llm.Response{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: id, Name: "echo_tool", Input: json.RawMessage(`{"n":1}`)}},
			},
			StopReason: "tool_use",
		}
	}
	// Different IDs each round (as a real provider would produce) but the
	// same name+arguments — that's what "identical" means here, not ID
	// equality.
	provider := &scriptedProvider{responses: []*llm.Response{identical("call_1"), identical("call_2"), identical("call_3")}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "keep going forever")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if provider.calls != 3 {
		t.Errorf("provider was called %d times, want exactly 3", provider.calls)
	}
	if !strings.Contains(got, "repeated 3 times") {
		t.Errorf("Run() = %q, want an explanation mentioning the repetition", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	assertFinalMessageIsExplanation(t, h)
}

// TestRunDoesNotTripRepetitionGuardOnAlternatingCalls guards against a
// too-eager repetition counter (e.g. one that doesn't reset on a change).
func TestRunDoesNotTripRepetitionGuardOnAlternatingCalls(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	callWith := func(id string, n int) *llm.Response {
		return &llm.Response{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: id, Name: "echo_tool", Input: json.RawMessage(fmt.Sprintf(`{"n":%d}`, n))}},
			},
			StopReason: "tool_use",
		}
	}
	done := &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "finished"}}},
		StopReason: "end_turn",
	}
	provider := &scriptedProvider{responses: []*llm.Response{
		callWith("call_a", 1), callWith("call_b", 2), callWith("call_c", 1), callWith("call_d", 2), done,
	}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "alternate")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "finished" {
		t.Errorf("Run() = %q, want %q (repetition guard must not trip on alternating calls)", got, "finished")
	}
	if provider.calls != 5 {
		t.Errorf("provider was called %d times, want 5", provider.calls)
	}
}

// panicTool always panics — it stands in for a broken tool implementation.
type panicTool struct{}

func (panicTool) Name() string            { return "panic_tool" }
func (panicTool) Description() string     { return "a tool that always panics" }
func (panicTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (panicTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	panic("boom")
}

var _ tools.Tool = panicTool{}

// TestRunRecoversFromPanickingTool is 4.3's AC for the panicking-tool case:
// a valid paired result, and the loop continues.
func TestRunRecoversFromPanickingTool(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(panicTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		toolCallResponseNamed("call_1", "panic_tool"),
		textResponse("recovered"),
	}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 0, AutoApprove)

	got, err := a.Run(context.Background(), "call the panicking tool")
	if err != nil {
		t.Fatalf("Run() error = %v, want the loop to survive the panic and continue", err)
	}
	if got != "recovered" {
		t.Errorf("Run() = %q, want %q", got, "recovered")
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	result := h.Snapshot()[2].Blocks[0].(llm.ToolResult)
	if !result.IsError {
		t.Error("ToolResult.IsError = false, want true for a panicking tool")
	}
	if !strings.Contains(result.Content, "panicked") {
		t.Errorf("ToolResult.Content = %q, want it to mention the panic", result.Content)
	}
}

// hangingTool ignores ctx entirely and sleeps far longer than any test
// timeout — a genuine hang, not just a slow call, so the only thing that
// can end it is runTool giving up and moving on.
type hangingTool struct{}

func (hangingTool) Name() string            { return "hanging_tool" }
func (hangingTool) Description() string     { return "a tool that ignores ctx and never returns in time" }
func (hangingTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (hangingTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	time.Sleep(200 * time.Millisecond)
	return "too late", nil
}

var _ tools.Tool = hangingTool{}

// TestRunTimesOutHangingTool is 4.3's AC for the hanging-tool case: a valid
// paired result, and the loop continues, without waiting for the tool.
func TestRunTimesOutHangingTool(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(hangingTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		toolCallResponseNamed("call_1", "hanging_tool"),
		textResponse("gave up waiting"),
	}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 20*time.Millisecond, AutoApprove)

	start := time.Now()
	got, err := a.Run(context.Background(), "call the hanging tool")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "gave up waiting" {
		t.Errorf("Run() = %q, want %q", got, "gave up waiting")
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("Run() took %s, want well under the tool's 200ms sleep (the per-tool timeout must not block the loop)", elapsed)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	result := h.Snapshot()[2].Blocks[0].(llm.ToolResult)
	if !result.IsError {
		t.Error("ToolResult.IsError = false, want true for a timed-out tool")
	}
	if !strings.Contains(result.Content, "deadline exceeded") {
		t.Errorf("ToolResult.Content = %q, want it to mention the timeout", result.Content)
	}
}

// blockingTool is well-behaved: it blocks on ctx and returns as soon as ctx
// is cancelled, closing started first so the test knows the tool is
// actually in flight before it cancels.
type blockingTool struct {
	started chan struct{}
}

func (t *blockingTool) Name() string            { return "blocking_tool" }
func (t *blockingTool) Description() string     { return "a tool that blocks until ctx is done" }
func (t *blockingTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *blockingTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	close(t.started)
	<-ctx.Done()
	return "", ctx.Err()
}

var _ tools.Tool = (*blockingTool)(nil)

// TestRunHandlesCancellationMidTool is 4.3's AC for the cancelled-tool
// case: a valid paired result, and the loop exits cleanly (Run returns the
// cancellation error rather than hanging or corrupting history).
func TestRunHandlesCancellationMidTool(t *testing.T) {
	reg := tools.NewRegistry()
	bt := &blockingTool{started: make(chan struct{})}
	if err := reg.Register(bt); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{toolCallResponseNamed("call_1", "blocking_tool")}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 0, AutoApprove)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-bt.started
		cancel()
	}()

	_, err := a.Run(ctx, "call the blocking tool")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run() error = %v, want it to wrap context.Canceled", err)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	result := h.Snapshot()[2].Blocks[0].(llm.ToolResult)
	if !result.IsError {
		t.Error("ToolResult.IsError = false, want true for a cancelled tool")
	}
}

// approvalTool implements tools.ApprovalRequired; ran records whether Run
// was actually invoked, so tests can prove a denial skips it.
type approvalTool struct {
	ran bool
}

func (t *approvalTool) Name() string            { return "sensitive_tool" }
func (t *approvalTool) Description() string     { return "a tool that requires approval" }
func (t *approvalTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *approvalTool) NeedsApproval() bool     { return true }
func (t *approvalTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	t.ran = true
	return "done", nil
}

var _ tools.ApprovalRequired = (*approvalTool)(nil)

type fakeApprover struct {
	approve bool
	err     error
	asked   bool
}

func (f *fakeApprover) Approve(_ context.Context, _ string, _ json.RawMessage, _ string) (bool, error) {
	f.asked = true
	return f.approve, f.err
}

// TestRunDeniedApprovalSkipsToolAndContinues is 5.3's AC: a denied approval
// returns an error result and the loop continues.
func TestRunDeniedApprovalSkipsToolAndContinues(t *testing.T) {
	reg := tools.NewRegistry()
	tool := &approvalTool{}
	if err := reg.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		toolCallResponseNamed("call_1", "sensitive_tool"),
		textResponse("moved on"),
	}}

	h := history.New()
	approver := &fakeApprover{approve: false}
	a := New(provider, reg, h, "", 1024, Guards{}, 0, approver)

	got, err := a.Run(context.Background(), "do the sensitive thing")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "moved on" {
		t.Errorf("Run() = %q, want %q", got, "moved on")
	}
	if !approver.asked {
		t.Error("approver was never consulted")
	}
	if tool.ran {
		t.Error("tool.Run was called despite a denied approval")
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	result := h.Snapshot()[2].Blocks[0].(llm.ToolResult)
	if !result.IsError {
		t.Error("ToolResult.IsError = false, want true for a denied approval")
	}
}

func TestRunApprovedCallRunsTheTool(t *testing.T) {
	reg := tools.NewRegistry()
	tool := &approvalTool{}
	if err := reg.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		toolCallResponseNamed("call_1", "sensitive_tool"),
		textResponse("finished"),
	}}

	h := history.New()
	approver := &fakeApprover{approve: true}
	a := New(provider, reg, h, "", 1024, Guards{}, 0, approver)

	if _, err := a.Run(context.Background(), "do the sensitive thing"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !approver.asked {
		t.Error("approver was never consulted")
	}
	if !tool.ran {
		t.Error("tool.Run was never called despite an approved approval")
	}
}

func TestRunSkipsApprovalForToolsThatDontNeedIt(t *testing.T) {
	reg := tools.NewRegistry()
	if err := reg.Register(echoTool{}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{toolCallResponse("call_1"), textResponse("done")}}

	h := history.New()
	approver := &fakeApprover{approve: false}
	a := New(provider, reg, h, "", 1024, Guards{}, 0, approver)

	if _, err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if approver.asked {
		t.Error("approver was consulted for a tool that doesn't implement ApprovalRequired")
	}
}

// TestRunDeniedWriteFileLeavesFileByteIdentical is 5.4's AC exercised
// end-to-end through the loop's approval path, using the real
// tools.WriteFileTool rather than a fake.
func TestRunDeniedWriteFileLeavesFileByteIdentical(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "existing.txt")
	if err := os.WriteFile(path, []byte("original content"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	sandbox, err := tools.NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	writeTool, err := tools.NewWriteFileTool(sandbox)
	if err != nil {
		t.Fatalf("NewWriteFileTool() error = %v", err)
	}
	reg := tools.NewRegistry()
	if err := reg.Register(writeTool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	writeInput, _ := json.Marshal(tools.WriteFileInput{Path: "existing.txt", Content: "attacker-controlled overwrite"})
	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "write_file", Input: writeInput}},
			},
			StopReason: "tool_use",
		},
		textResponse("ok"),
	}}

	h := history.New()
	a := New(provider, reg, h, "", 1024, Guards{}, 0, &fakeApprover{approve: false})

	if _, err := a.Run(context.Background(), "overwrite the file"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != "original content" {
		t.Errorf("file contents = %q, want the original untouched after a denied approval", got)
	}
}

// statusRecordingRenderer wraps a ui.Renderer and records every
// ToolStarted/ToolFinished call it receives, so a test can prove
// RunStream actually reports tool status to a Renderer that asks for it.
type statusRecordingRenderer struct {
	ui.Renderer
	mu     sync.Mutex
	events []string
}

func (r *statusRecordingRenderer) ToolStarted(_, name string, _ json.RawMessage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, "start:"+name)
}

func (r *statusRecordingRenderer) ToolFinished(_, name, _ string, isError bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	status := "ok"
	if isError {
		status = "error"
	}
	r.events = append(r.events, fmt.Sprintf("finish:%s:%s", name, status))
}

var _ ui.ToolStatusReporter = (*statusRecordingRenderer)(nil)

// TestRunStreamDrivesScriptedConversationWithToolCall mirrors
// TestRunDrivesScriptedConversationWithParallelTools but through RunStream:
// the provider's Stream (not Chat) drives the round, and a ui.PlainRenderer
// does the rendering. The point is proving RunStream reaches the same
// final state as Run — valid history, right answer, right call count.
func TestRunStreamDrivesScriptedConversationWithToolCall(t *testing.T) {
	echo := &delayedTool{name: "echo_tool", delay: 0, result: "echoed", mu: &sync.Mutex{}, order: &[]string{}}
	reg := tools.NewRegistry()
	if err := reg.Register(echo); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message: llm.Message{
				Role:   llm.RoleAssistant,
				Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "echo_tool", Input: json.RawMessage(`{}`)}},
			},
			StopReason: "tool_use",
		},
		textResponse("done"),
	}}

	h := history.New()
	a := New(provider, reg, h, "you are a test agent", 1024, Guards{}, 0, AutoApprove)

	var buf bytes.Buffer
	renderer := &statusRecordingRenderer{Renderer: ui.PlainRenderer{Out: &buf}}
	got, err := a.RunStream(context.Background(), "please help", renderer)
	if err != nil {
		t.Fatalf("RunStream() error = %v", err)
	}
	if got != "done" {
		t.Errorf("RunStream() = %q, want %q", got, "done")
	}
	if provider.calls != 2 {
		t.Errorf("provider was called %d times, want 2", provider.calls)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	renderer.mu.Lock()
	events := append([]string(nil), renderer.events...)
	renderer.mu.Unlock()
	want := []string{"start:echo_tool", "finish:echo_tool:ok"}
	if diff := cmp.Diff(want, events); diff != "" {
		t.Errorf("tool status events (-want +got):\n%s", diff)
	}
}

// fakeRenderer lets a test script Render's return value directly, to
// exercise RunStream's error path without a real event stream.
type fakeRenderer struct {
	resp *llm.Response
	err  error
}

func (f *fakeRenderer) Render(context.Context, <-chan llm.Event) (*llm.Response, error) {
	return f.resp, f.err
}

func TestRunStreamPropagatesRenderError(t *testing.T) {
	provider := &scriptedProvider{responses: []*llm.Response{textResponse("unused")}}
	h := history.New()
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)

	wantErr := errors.New("render exploded")
	_, err := a.RunStream(context.Background(), "hi", &fakeRenderer{err: wantErr})
	if !errors.Is(err, wantErr) {
		t.Fatalf("RunStream() error = %v, want it to wrap %v", err, wantErr)
	}
}

func TestRunStreamStopsAtMaxTurnsGuard(t *testing.T) {
	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "missing_tool", Input: json.RawMessage(`{}`)}}},
			StopReason: "tool_use",
		},
		{
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.ToolUse{ID: "call_2", Name: "missing_tool", Input: json.RawMessage(`{}`)}}},
			StopReason: "tool_use",
		},
	}}

	h := history.New()
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{MaxTurns: 2}, 0, AutoApprove)

	var buf bytes.Buffer
	got, err := a.RunStream(context.Background(), "hi", ui.PlainRenderer{Out: &buf})
	if err != nil {
		t.Fatalf("RunStream() error = %v", err)
	}
	if !strings.Contains(got, "maximum of 2 turn") {
		t.Errorf("RunStream() = %q, want an explanation naming the max-turns guard", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}
