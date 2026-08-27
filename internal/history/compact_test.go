package history_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
)

// buildTurns returns n user/assistant turns; every third turn also
// includes a tool call and its result, so a compaction boundary has
// ToolUse/ToolResult pairs to potentially (and must not) split.
func buildTurns(n int) []llm.Message {
	var msgs []llm.Message
	for i := 0; i < n; i++ {
		msgs = append(msgs, textMsg(llm.RoleUser, fmt.Sprintf("turn %d question", i)))
		if i%3 == 0 {
			id := fmt.Sprintf("call_%d", i)
			msgs = append(msgs, toolUseMsg(llm.ToolUse{ID: id, Name: "read_file", Input: json.RawMessage(`{}`)}))
			msgs = append(msgs, toolResultMsg(llm.ToolResult{ToolUseID: id, Content: "result"}))
			msgs = append(msgs, textMsg(llm.RoleAssistant, fmt.Sprintf("turn %d answer after tool", i)))
		} else {
			msgs = append(msgs, textMsg(llm.RoleAssistant, fmt.Sprintf("turn %d answer", i)))
		}
	}
	return msgs
}

func TestCompactionBoundaryKeepsExactTurnCount(t *testing.T) {
	msgs := buildTurns(10)
	boundary := history.CompactionBoundary(msgs, 3)

	if !cmp.Equal(msgs[boundary], textMsg(llm.RoleUser, "turn 7 question")) {
		t.Fatalf("message at boundary %d = %+v, want the start of turn 7 (the 3 most recent of 10 turns: 7, 8, 9)", boundary, msgs[boundary])
	}
}

func TestCompactionBoundaryZeroWhenNotEnoughTurns(t *testing.T) {
	msgs := buildTurns(3)
	if boundary := history.CompactionBoundary(msgs, 5); boundary != 0 {
		t.Errorf("CompactionBoundary() = %d, want 0 when there are fewer turns than keepTurns", boundary)
	}
}

// TestCompact40TurnHistoryWithToolCallsStillValidates is 8.2's AC:
// compacting a 40-turn history with tool calls still passes Validate(),
// and the boundary never splits a ToolUse/ToolResult pair.
func TestCompact40TurnHistoryWithToolCallsStillValidates(t *testing.T) {
	h := history.New()
	for _, m := range buildTurns(40) {
		h.Append(m)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() on the uncompacted history error = %v", err)
	}

	if err := h.Compact(5, "summary of the first 35 turns"); err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() after Compact() error = %v", err)
	}

	msgs := h.Snapshot()
	if len(msgs) == 0 || msgs[0].Role != llm.RoleAssistant {
		t.Fatalf("messages[0] = %+v, want the synthetic assistant summary", msgs[0])
	}
	text, ok := msgs[0].Blocks[0].(llm.Text)
	if !ok || text.Text != "summary of the first 35 turns" {
		t.Errorf("messages[0] text = %+v, want the summary", msgs[0].Blocks[0])
	}
}

func TestCompactPreservesLastNTurnsVerbatim(t *testing.T) {
	all := buildTurns(10)
	h := history.New()
	for _, m := range all {
		h.Append(m)
	}

	if err := h.Compact(3, "elided"); err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	got := h.Snapshot()
	boundary := history.CompactionBoundary(all, 3)
	wantTail := all[boundary:]
	gotTail := got[1:] // [0] is the synthetic summary message
	if diff := cmp.Diff(wantTail, gotTail); diff != "" {
		t.Errorf("kept turns mismatch (-want +got):\n%s", diff)
	}
}

func TestCompactNoopsWhenNothingToCompact(t *testing.T) {
	h := history.New()
	for _, m := range buildTurns(2) {
		h.Append(m)
	}
	before := h.Snapshot()

	if err := h.Compact(5, "unused"); err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	if diff := cmp.Diff(before, h.Snapshot()); diff != "" {
		t.Errorf("Compact() changed history when there was nothing to compact (-before +after):\n%s", diff)
	}
}

func TestEstimateTokens(t *testing.T) {
	if got := history.EstimateTokens(nil); got != 0 {
		t.Errorf("EstimateTokens(nil) = %d, want 0", got)
	}
	if got := history.EstimateTokens(buildTurns(1)); got <= 0 {
		t.Errorf("EstimateTokens(1 turn) = %d, want a positive estimate", got)
	}
}
