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

// fakeSwitchableProvider adds llm.ModelSwitcher to fakeChatProvider, for
// testing /model without a real backend. unknownModel, if set, makes
// ValidateModel fail for that one name; every other name validates.
type fakeSwitchableProvider struct {
	*fakeChatProvider
	unknownModel string
	model        string
}

func (f *fakeSwitchableProvider) ValidateModel(_ context.Context, model string) error {
	if model == f.unknownModel {
		return fmt.Errorf("model %q not found", model)
	}
	return nil
}

func (f *fakeSwitchableProvider) SetModel(model string) {
	f.model = model
}

// readFileVia runs the registry's read_file tool for path, for assertions
// about which paths the sandbox currently accepts.
func readFileVia(t *testing.T, registry *tools.Registry, path string) (string, error) {
	t.Helper()
	tool, err := registry.Get("read_file")
	if err != nil {
		t.Fatalf("Get(read_file) error = %v", err)
	}
	input, err := json.Marshal(map[string]string{"path": path})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	return tool.Run(context.Background(), input)
}

// TestCdCommandRerootsWorkspaceSandbox is 11.3's AC for /cd: after /cd, a
// read_file reachable only from the new root succeeds and one under the
// old root is rejected.
func TestCdCommandRerootsWorkspaceSandbox(t *testing.T) {
	oldRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(oldRoot, "old.txt"), []byte("old"), 0o600); err != nil {
		t.Fatalf("WriteFile(old.txt) error = %v", err)
	}
	newRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(newRoot, "new.txt"), []byte("new"), 0o600); err != nil {
		t.Fatalf("WriteFile(new.txt) error = %v", err)
	}

	registry, err := buildRegistry(&config.Config{Workspace: oldRoot})
	if err != nil {
		t.Fatalf("buildRegistry() error = %v", err)
	}
	h := history.New()
	ag := agent.New(&fakeChatProvider{reply: "ok"}, registry, h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	if _, err := readFileVia(t, registry, "old.txt"); err != nil {
		t.Fatalf("read_file(old.txt) before /cd error = %v", err)
	}

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "cd", newRoot); err != nil {
		t.Fatalf("handleCommand(cd) error = %v", err)
	}
	if !strings.Contains(out.String(), "workspace root changed") {
		t.Errorf("output = %q, want a confirmation", out.String())
	}

	if _, err := readFileVia(t, registry, "new.txt"); err != nil {
		t.Errorf("read_file(new.txt) after /cd error = %v, want it reachable from the new root", err)
	}
	if _, err := readFileVia(t, registry, filepath.Join(oldRoot, "old.txt")); err == nil {
		t.Error("read_file(old.txt) after /cd error = nil, want the old root rejected as escaping the sandbox")
	}
}

// TestCdCommandRejectsMissingTarget is 11.3's other AC for /cd: a missing
// target is refused and the original root stays active.
func TestCdCommandRejectsMissingTarget(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "marker.txt"), []byte("marker"), 0o600); err != nil {
		t.Fatalf("WriteFile(marker.txt) error = %v", err)
	}
	registry, err := buildRegistry(&config.Config{Workspace: root})
	if err != nil {
		t.Fatalf("buildRegistry() error = %v", err)
	}
	h := history.New()
	ag := agent.New(&fakeChatProvider{reply: "ok"}, registry, h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "cd", filepath.Join(root, "does-not-exist")); err != nil {
		t.Fatalf("handleCommand(cd) error = %v", err)
	}
	if !strings.Contains(out.String(), "cd failed") {
		t.Errorf("output = %q, want a failure message", out.String())
	}
	if _, err := readFileVia(t, registry, "marker.txt"); err != nil {
		t.Errorf("read_file(marker.txt) after a refused /cd error = %v, want the original root still active", err)
	}
}

// TestModelCommandRejectsUnknownWithoutMutating is 11.3's AC for /model: a
// name the provider does not know errors without mutating the provider's
// model or the compaction policy.
func TestModelCommandRejectsUnknownWithoutMutating(t *testing.T) {
	h := history.New()
	provider := &fakeSwitchableProvider{fakeChatProvider: &fakeChatProvider{reply: "ok"}, unknownModel: "ghost-model", model: "original-model"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	ag.SetCompactionPolicy(agent.CompactionPolicy{ContextWindow: 500, TriggerFraction: 0.75, KeepTurns: 6})

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "model", "ghost-model"); err != nil {
		t.Fatalf("handleCommand(model) error = %v", err)
	}
	if !strings.Contains(out.String(), "model failed") {
		t.Errorf("output = %q, want a failure message", out.String())
	}
	if provider.model != "original-model" {
		t.Errorf("provider.model = %q, want it unchanged after a rejected /model", provider.model)
	}
	if got := ag.CompactionPolicy().ContextWindow; got != 500 {
		t.Errorf("CompactionPolicy().ContextWindow = %d, want 500 unchanged", got)
	}
}

// TestModelCommandErrorsWhenProviderCannotSwitch covers a provider that
// doesn't implement llm.ModelSwitcher at all (every fakeChatProvider in the
// rest of this file, and any future provider that never adds it).
func TestModelCommandErrorsWhenProviderCannotSwitch(t *testing.T) {
	h := history.New()
	ag := agent.New(&fakeChatProvider{reply: "ok"}, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "model", "some-model"); err != nil {
		t.Fatalf("handleCommand(model) error = %v", err)
	}
	if !strings.Contains(out.String(), "does not support switching models") {
		t.Errorf("output = %q, want a does-not-support message", out.String())
	}
}

// TestModelCommandSwitchingSmallerWindowTriggersCompactionNextTurn is
// 11.3's AC: switching to a smaller context window triggers the
// compaction check on the next turn.
func TestModelCommandSwitchingSmallerWindowTriggersCompactionNextTurn(t *testing.T) {
	h := toolHeavyHistory(8)
	before := history.EstimateTokens(h.Snapshot())

	provider := &fakeSwitchableProvider{fakeChatProvider: &fakeChatProvider{reply: "concise summary of the earlier turns"}, model: "old-model"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	ag.SetCompactionPolicy(agent.CompactionPolicy{ContextWindow: 1_000_000, TriggerFraction: 0.75, KeepTurns: 2})

	newWindow := before / 2
	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, noSession(t), "model", fmt.Sprintf("new-model %d", newWindow)); err != nil {
		t.Fatalf("handleCommand(model) error = %v", err)
	}
	if !strings.Contains(out.String(), "compaction will run on the next turn") {
		t.Errorf("output = %q, want a compaction warning", out.String())
	}
	if provider.model != "new-model" {
		t.Errorf("provider.model = %q, want %q", provider.model, "new-model")
	}

	if _, err := ag.Run(context.Background(), "one more question"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() after the next turn error = %v", err)
	}
	if after := history.EstimateTokens(h.Snapshot()); after >= before {
		t.Errorf("estimated tokens after the next turn = %d, want less than the pre-/model estimate (%d): the smaller window should have triggered compaction", after, before)
	}
}

// TestUsageCommandCostMatchesHandComputedFigure is 15.1's AC: a scripted
// run's total matches a hand-computed figure to the cent. claude-sonnet-4-5
// prices at $3.00/$15.00 per million input/output tokens; 100,000 of each
// costs 100000*3/1e6 + 100000*15/1e6 = 0.3 + 1.5 = $1.8000 exactly.
func TestUsageCommandCostMatchesHandComputedFigure(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok", usage: llm.Usage{InputTokens: 100_000, OutputTokens: 100_000}}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	if _, err := ag.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	sess := noSession(t)
	sess.model = "claude-sonnet-4-5"

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, sess, "usage", ""); err != nil {
		t.Fatalf("handleCommand(usage) error = %v", err)
	}
	if got := out.String(); !strings.Contains(got, "cumulative cost: $1.8000") {
		t.Errorf("handleCommand(usage) output = %q, want it to contain the hand-computed cost $1.8000", got)
	}
}

// TestUsageCommandOmitsCostForUnknownModel is 15.1's AC: an unknown model
// shows no cost and raises no error.
func TestUsageCommandOmitsCostForUnknownModel(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok", usage: llm.Usage{InputTokens: 100, OutputTokens: 100}}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	if _, err := ag.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	sess := noSession(t)
	sess.model = "not-a-real-model"

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, sess, "usage", ""); err != nil {
		t.Fatalf("handleCommand(usage) error = %v", err)
	}
	got := out.String()
	if strings.Contains(got, "cost") {
		t.Errorf("handleCommand(usage) output = %q, want no cost line for a model with no pricing entry", got)
	}
	if !strings.Contains(got, "cumulative: 100 input, 100 output") {
		t.Errorf("handleCommand(usage) output = %q, want token counts shown regardless of pricing", got)
	}
}

// fakeOllamaProvider is fakeChatProvider with Name() overridden to
// "ollama", so /usage's Ollama-reports-zero branch (15.1's AC) can be
// exercised without a real Ollama backend.
type fakeOllamaProvider struct {
	*fakeChatProvider
}

func (f *fakeOllamaProvider) Name() string { return "ollama" }

// TestUsageCommandOllamaReportsZeroCost is 15.1's AC: Ollama reports zero
// — an explicit $0.0000, not an omitted line, since Ollama is a known,
// supported provider that simply has no per-token price, unlike a model
// this table has never heard of.
func TestUsageCommandOllamaReportsZeroCost(t *testing.T) {
	provider := &fakeOllamaProvider{fakeChatProvider: &fakeChatProvider{reply: "ok", usage: llm.Usage{InputTokens: 100, OutputTokens: 100}}}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	if _, err := ag.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	sess := noSession(t)
	sess.model = "qwen3:8b"

	var out bytes.Buffer
	if _, err := handleCommand(context.Background(), &out, ag, h, sess, "usage", ""); err != nil {
		t.Fatalf("handleCommand(usage) error = %v", err)
	}
	if got := out.String(); !strings.Contains(got, "cumulative cost: $0.0000") {
		t.Errorf("handleCommand(usage) output = %q, want an explicit $0.0000 for Ollama", got)
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
