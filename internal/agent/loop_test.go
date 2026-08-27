package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// scriptedProvider returns responses in order, one per Chat call — the
// "scripted fake provider" 4.1's AC calls for.
type scriptedProvider struct {
	responses []*llm.Response
	calls     int
}

func (s *scriptedProvider) Name() string { return "scripted" }

func (s *scriptedProvider) Chat(_ context.Context, _ llm.Request) (*llm.Response, error) {
	if s.calls >= len(s.responses) {
		return nil, fmt.Errorf("scripted: no response scripted for call %d", s.calls+1)
	}
	resp := s.responses[s.calls]
	s.calls++
	return resp, nil
}

func (s *scriptedProvider) Stream(_ context.Context, _ llm.Request) (<-chan llm.Event, error) {
	return nil, errors.New("scripted: streaming not implemented")
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
	a := New(provider, reg, h, "you are a test agent", 1024)

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
	a := New(provider, tools.NewRegistry(), h, "", 1024)

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
	a := New(provider, tools.NewRegistry(), h, "", 1024)

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
	a := New(provider, tools.NewRegistry(), h, "", 1024)

	if _, err := a.Run(context.Background(), "hi"); err == nil {
		t.Fatal("Run() error = nil, want the provider's error propagated")
	}
}
