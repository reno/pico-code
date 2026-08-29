package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

const helpGoldenPath = "testdata/golden/help.txt"

// TestHelpCommandMatchesGolden is 11.1's AC: /help renders commandTable
// aligned. Run with RECORD=1 to (re)write the golden file.
func TestHelpCommandMatchesGolden(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "help", ""); err != nil {
		t.Fatalf("handleCommand(help) error = %v", err)
	}
	got := out.String()

	if os.Getenv("RECORD") == "1" {
		if err := os.MkdirAll(filepath.Dir(helpGoldenPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(helpGoldenPath, []byte(got), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	want, err := os.ReadFile(helpGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to create it)", helpGoldenPath, err)
	}
	if diff := cmp.Diff(string(want), got); diff != "" {
		t.Errorf("/help output mismatch (-want +got):\n%s", diff)
	}
}

// TestCommandTableEntriesHaveSummaries guards against a bare entry slipping
// into commandTable with no help text, since /help's usefulness depends on
// every command documenting itself.
func TestCommandTableEntriesHaveSummaries(t *testing.T) {
	for _, c := range commandTable {
		if strings.TrimSpace(c.summary) == "" {
			t.Errorf("command %q has no summary", c.name)
		}
		if c.run == nil {
			t.Errorf("command %q has no run function", c.name)
		}
	}
}

// TestUnknownCommandSuggestsClosestMatch is 11.1's AC: an unknown command
// reports the closest match instead of a bare error.
func TestUnknownCommandSuggestsClosestMatch(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "hlep", ""); err != nil {
		t.Fatalf("handleCommand(hlep) error = %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "did you mean /help") {
		t.Errorf("output = %q, want it to suggest /help", got)
	}
}

// TestClearCommandWipesScrollbackNotHistory is 11.2's AC for /clear:
// history.Validate() still passes and the message count is unchanged, in
// both the plain REPL (which reacts to errClearScrollback by clearing the
// real terminal) and the TUI path (which reacts by resetting the
// transcript instead).
func TestClearCommandWipesScrollbackNotHistory(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	in := "hello\n/clear\n"
	runTurn := newTurnRunner(&config.Config{Stream: true}, ag, &out)
	if err := runREPL(context.Background(), strings.NewReader(in), &out, ag, h, noSession(t), runTurn); err != nil {
		t.Fatalf("runREPL() error = %v", err)
	}

	if err := h.Validate(); err != nil {
		t.Errorf("Validate() after /clear error = %v", err)
	}
	if got := len(h.Snapshot()); got != 2 {
		t.Errorf("history has %d message(s) after /clear, want 2 (the one turn, untouched)", got)
	}
	if got := out.String(); !strings.Contains(got, "\x1b[") {
		t.Errorf("output = %q, want it to contain an ANSI clear sequence", got)
	}

	output, err := runTUICommand(context.Background(), ag, h, noSession(t), "/clear", "clear", "")
	if !errors.Is(err, errClearScrollback) {
		t.Errorf("runTUICommand(/clear) error = %v, want errClearScrollback", err)
	}
	_ = output
	if got := len(h.Snapshot()); got != 2 {
		t.Errorf("history has %d message(s) after the TUI /clear, want 2 (untouched)", got)
	}
}

// toolHeavyHistory builds n turns, each a user question, a tool call, its
// result, and an assistant answer — "tool-heavy" per 11.2's AC for
// /compact, and long enough that CompactionBoundary finds something to cut
// with a small keepTurns.
func toolHeavyHistory(n int) *history.History {
	h := history.New()
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("call_%d", i)
		h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: fmt.Sprintf("question number %d, padded out so it has some real length to it", i)}}})
		h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.ToolUse{ID: id, Name: "read_file", Input: json.RawMessage(fmt.Sprintf(`{"path":"file_%d.go"}`, i))}}})
		h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.ToolResult{ToolUseID: id, Content: fmt.Sprintf("contents of file_%d.go, also padded to a realistic length", i)}}})
		h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: fmt.Sprintf("answer number %d, likewise padded out", i)}}})
	}
	return h
}

// TestCompactCommandLowersEstimateOnToolHeavyHistory is 11.2's AC for
// /compact: forcing it on a long tool-heavy history lowers the token
// estimate and still passes Validate().
func TestCompactCommandLowersEstimateOnToolHeavyHistory(t *testing.T) {
	h := toolHeavyHistory(8)
	before := history.EstimateTokens(h.Snapshot())

	provider := &fakeChatProvider{reply: "concise summary of the earlier turns"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	ag.SetCompactionPolicy(agent.CompactionPolicy{KeepTurns: 2})

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "compact", ""); err != nil {
		t.Fatalf("handleCommand(compact) error = %v", err)
	}

	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() after /compact error = %v", err)
	}
	after := history.EstimateTokens(h.Snapshot())
	if after >= before {
		t.Errorf("estimated tokens after /compact = %d, want less than before (%d)", after, before)
	}
	if got := out.String(); !strings.Contains(got, fmt.Sprintf("%d -> %d", before, after)) {
		t.Errorf("output = %q, want it to report %d -> %d", got, before, after)
	}
}

// TestCompactCommandNoopWhenNothingToCompact covers the case AC 11.2 leaves
// implicit: a short history has nothing eligible to summarize, so /compact
// must not call the provider or change history at all.
func TestCompactCommandNoopWhenNothingToCompact(t *testing.T) {
	h := history.New()
	h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}})
	before := h.Snapshot()

	provider := &fakeChatProvider{reply: "should never be requested"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	ag.SetCompactionPolicy(agent.CompactionPolicy{KeepTurns: 6})

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "compact", ""); err != nil {
		t.Fatalf("handleCommand(compact) error = %v", err)
	}
	if diff := cmp.Diff(before, h.Snapshot()); diff != "" {
		t.Errorf("history changed by a no-op /compact (-before +after):\n%s", diff)
	}
	if got := out.String(); !strings.Contains(got, "nothing to compact") {
		t.Errorf("output = %q, want it to report nothing to compact", got)
	}
}

// TestExitCommandSavesSessionBeforeSignalingExit is 11.2's AC for /exit:
// the session file is written before the command returns, and it leaves
// through the same path as Ctrl+D (runREPL returns nil, not an error, and
// any input after /exit is never processed).
func TestExitCommandSavesSessionBeforeSignalingExit(t *testing.T) {
	dir := t.TempDir()
	sess, err := newSession(dir, "work")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	h := history.New()
	provider := &fakeChatProvider{reply: "ok"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	in := "hello\n/exit\nshould never run\n"
	runTurn := newTurnRunner(&config.Config{Stream: true}, ag, &out)
	if err := runREPL(context.Background(), strings.NewReader(in), &out, ag, h, sess, runTurn); err != nil {
		t.Fatalf("runREPL() error = %v, want nil (the same clean return Ctrl+D produces)", err)
	}

	saved, err := history.Load(sess.path("work"))
	if err != nil {
		t.Fatalf("Load() session file error = %v, want /exit to have written it first", err)
	}
	if diff := cmp.Diff(h.Snapshot(), saved.Snapshot()); diff != "" {
		t.Errorf("saved session differs from in-memory history (-memory +saved):\n%s", diff)
	}
	if strings.Contains(out.String(), "should never run") {
		t.Error("input after /exit was processed, want runREPL to stop at /exit")
	}
}

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"help", "help", 0},
		{"hlep", "help", 2},
		{"sav", "save", 1},
		{"kitten", "sitting", 3},
	}
	for _, tt := range tests {
		if got := levenshtein(tt.a, tt.b); got != tt.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}
