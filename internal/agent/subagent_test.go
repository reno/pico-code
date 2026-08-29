package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

func subAgentToolUse(t *testing.T, id, task string) llm.ToolUse {
	t.Helper()
	input, err := json.Marshal(SubAgentInput{Task: task})
	if err != nil {
		t.Fatalf("marshal SubAgentInput: %v", err)
	}
	return llm.ToolUse{ID: id, Name: "sub_agent", Input: input}
}

func subAgentCall(t *testing.T, id, task string) *llm.Response {
	t.Helper()
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{subAgentToolUse(t, id, task)}},
		StopReason: "tool_use",
	}
}

// TestSubAgentToolKeepsParentHistoryValidAndChildMessagesOut is half of
// 14.1's AC: the parent's history passes Validate() and never contains the
// child's own messages, only this tool's single flattened ToolResult.
func TestSubAgentToolKeepsParentHistoryValidAndChildMessagesOut(t *testing.T) {
	provider := &scriptedProvider{responses: []*llm.Response{
		subAgentCall(t, "call_1", "summarize something"),
		textResponse("child's own answer"),
		textResponse("all done"),
	}}

	h := history.New()
	parentReg := tools.NewRegistry()
	a := New(provider, parentReg, h, "", 1024, Guards{}, 0, AutoApprove)

	sa, err := NewSubAgentTool(provider, nil, "", 1024, 0, a)
	if err != nil {
		t.Fatalf("NewSubAgentTool() error = %v", err)
	}
	if err := parentReg.Register(sa); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	got, err := a.Run(context.Background(), "please delegate")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "all done" {
		t.Errorf("Run() = %q, want %q", got, "all done")
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	msgs := h.Snapshot()
	if len(msgs) != 4 {
		t.Fatalf("parent history has %d messages, want exactly 4 (user, tool_use, tool_result, final text) — a child message would inflate this", len(msgs))
	}
	tr, ok := msgs[2].Blocks[0].(llm.ToolResult)
	if !ok {
		t.Fatalf("msgs[2].Blocks[0] = %T, want llm.ToolResult", msgs[2].Blocks[0])
	}
	if tr.IsError {
		t.Errorf("ToolResult.IsError = true, want false for a sub-agent that finished normally")
	}
	if tr.Content != "child's own answer" {
		t.Errorf("ToolResult.Content = %q, want the child's final answer %q", tr.Content, "child's own answer")
	}
}

// TestSubAgentToolBudgetExhaustionReturnsErrorNotParentGuardTrip is 14.1's
// AC: a sub-agent exhausting its own budget comes back as
// ToolResult{IsError:true}, not as the parent's own guard tripping.
func TestSubAgentToolBudgetExhaustionReturnsErrorNotParentGuardTrip(t *testing.T) {
	childTrips := toolCallResponseNamed("child_call_1", "echo_tool")
	childTrips.Usage = llm.Usage{InputTokens: 150}

	provider := &scriptedProvider{responses: []*llm.Response{
		subAgentCall(t, "call_1", "do a lot of work"),
		childTrips,
		textResponse("the sub-agent ran out of budget, here's what I know"),
	}}

	h := history.New()
	parentReg := tools.NewRegistry()
	// Parent's own TokenBudget is 100 — high enough that its own round-1
	// usage (zero, since only the child's response carries Usage) never
	// trips it. This isolates the assertion to the child's derived budget.
	a := New(provider, parentReg, h, "", 1024, Guards{TokenBudget: 100}, 0, AutoApprove)

	sa, err := NewSubAgentTool(provider, []tools.Tool{echoTool{}}, "", 1024, 0, a)
	if err != nil {
		t.Fatalf("NewSubAgentTool() error = %v", err)
	}
	if err := parentReg.Register(sa); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	got, err := a.Run(context.Background(), "please delegate a big task")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if strings.Contains(got, "Stopping:") {
		t.Errorf("Run() = %q, want the parent's own guard to never trip", got)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	msgs := h.Snapshot()
	tr, ok := msgs[2].Blocks[0].(llm.ToolResult)
	if !ok {
		t.Fatalf("msgs[2].Blocks[0] = %T, want llm.ToolResult", msgs[2].Blocks[0])
	}
	if !tr.IsError {
		t.Error("ToolResult.IsError = false, want true: the sub-agent exhausted its own budget")
	}
	if !strings.Contains(tr.Content, "budget") {
		t.Errorf("ToolResult.Content = %q, want it to mention the exhausted budget", tr.Content)
	}
}

// TestSubAgentToolsRunConcurrentlyInOneAssistantMessage is 14.1's AC:
// several sub-agents in one assistant message run under the shared errgroup
// and are still answered correctly, in call order.
func TestSubAgentToolsRunConcurrentlyInOneAssistantMessage(t *testing.T) {
	// Both children get the identical scripted reply: which of the two
	// concurrent nested Agents claims which slot in the shared provider's
	// response list is not deterministic, so the test can't (and doesn't
	// need to) tell them apart by content — only that both come back
	// correctly paired to their own call ID, in the original call order.
	provider := &scriptedProvider{responses: []*llm.Response{
		{
			Message: llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{
				subAgentToolUse(t, "call_a", "task A"),
				subAgentToolUse(t, "call_b", "task B"),
			}},
			StopReason: "tool_use",
		},
		textResponse("delegated task complete"),
		textResponse("delegated task complete"),
		textResponse("both tasks are done"),
	}}

	h := history.New()
	parentReg := tools.NewRegistry()
	a := New(provider, parentReg, h, "", 1024, Guards{}, 0, AutoApprove)

	sa, err := NewSubAgentTool(provider, nil, "", 1024, 0, a)
	if err != nil {
		t.Fatalf("NewSubAgentTool() error = %v", err)
	}
	if err := parentReg.Register(sa); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	got, err := a.Run(context.Background(), "please delegate both tasks")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "both tasks are done" {
		t.Errorf("Run() = %q, want %q", got, "both tasks are done")
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	msgs := h.Snapshot()
	resultMsg := msgs[2]
	if len(resultMsg.Blocks) != 2 {
		t.Fatalf("tool result message has %d blocks, want 2", len(resultMsg.Blocks))
	}
	wantIDs := []string{"call_a", "call_b"}
	for i, want := range wantIDs {
		tr, ok := resultMsg.Blocks[i].(llm.ToolResult)
		if !ok {
			t.Fatalf("resultMsg.Blocks[%d] = %T, want llm.ToolResult", i, resultMsg.Blocks[i])
		}
		if tr.ToolUseID != want {
			t.Errorf("resultMsg.Blocks[%d].ToolUseID = %q, want %q (original call order, independent of completion order)", i, tr.ToolUseID, want)
		}
		if tr.IsError {
			t.Errorf("resultMsg.Blocks[%d].IsError = true, want false", i)
		}
		if tr.Content != "delegated task complete" {
			t.Errorf("resultMsg.Blocks[%d].Content = %q, want %q", i, tr.Content, "delegated task complete")
		}
	}
}

// TestSubAgentToolCancellationPropagatesToNestedAgent is 14.1's AC:
// cancelling the parent ctx tears the nested Agent down too — the same
// ctx chain, no separate wiring — leaving a valid paired error result and
// no hang. The package's TestMain (goleak.VerifyTestMain) is what actually
// proves no goroutine, nested or not, survives past this test.
func TestSubAgentToolCancellationPropagatesToNestedAgent(t *testing.T) {
	bt := &blockingTool{started: make(chan struct{})}
	childReg := []tools.Tool{bt}

	provider := &scriptedProvider{responses: []*llm.Response{
		subAgentCall(t, "call_1", "run the blocking tool"),
		toolCallResponseNamed("child_call_1", "blocking_tool"),
	}}

	h := history.New()
	parentReg := tools.NewRegistry()
	a := New(provider, parentReg, h, "", 1024, Guards{}, 0, AutoApprove)

	sa, err := NewSubAgentTool(provider, childReg, "", 1024, 0, a)
	if err != nil {
		t.Fatalf("NewSubAgentTool() error = %v", err)
	}
	if err := parentReg.Register(sa); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-bt.started
		cancel()
	}()

	_, err = a.Run(ctx, "call the sub-agent")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run() error = %v, want it to wrap context.Canceled", err)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	msgs := h.Snapshot()
	tr, ok := msgs[2].Blocks[0].(llm.ToolResult)
	if !ok {
		t.Fatalf("msgs[2].Blocks[0] = %T, want llm.ToolResult", msgs[2].Blocks[0])
	}
	if !tr.IsError {
		t.Error("ToolResult.IsError = false, want true for a cancelled sub-agent")
	}
}
