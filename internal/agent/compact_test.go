package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

const summarizationPromptGoldenPath = "testdata/golden/summarization_prompt.txt"

// TestSummarizationPromptGolden is 8.2's AC: a golden test pins the
// summarization prompt. Run with RECORD=1 to (re)write the golden file,
// mirroring the RECORD=1 convention CLAUDE.md's testing section already
// uses for provider fixtures.
func TestSummarizationPromptGolden(t *testing.T) {
	elided := []llm.Message{
		{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "read config.go and tell me the default port"}}},
		{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"config.go"}`)}}},
		{Role: llm.RoleUser, Blocks: []llm.Block{llm.ToolResult{ToolUseID: "call_1", Content: "const defaultPort = 8080"}}},
		{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "The default port is 8080."}}},
	}
	got := buildSummarizationPrompt(elided)

	if os.Getenv("RECORD") == "1" {
		if err := os.WriteFile(summarizationPromptGoldenPath, []byte(got), 0o600); err != nil {
			t.Fatalf("WriteFile(%s) error = %v", summarizationPromptGoldenPath, err)
		}
	}

	want, err := os.ReadFile(summarizationPromptGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to generate it)", summarizationPromptGoldenPath, err)
	}
	if got != string(want) {
		t.Errorf("buildSummarizationPrompt() mismatch against %s\ngot:\n%s\nwant:\n%s", summarizationPromptGoldenPath, got, want)
	}
}

// compactionScriptedProvider answers the first Chat call — maybeCompact's
// summarization request, which always happens before the turn's own — with
// summary, then falls through to the wrapped scriptedProvider's script for
// the turn itself.
type compactionScriptedProvider struct {
	*scriptedProvider
	summary    string
	summarized bool
}

func (p *compactionScriptedProvider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if !p.summarized {
		p.summarized = true
		return textResponse(p.summary), nil
	}
	return p.scriptedProvider.Chat(ctx, req)
}

// TestMaybeCompactTriggersAboveThresholdAndKeepsHistoryValid drives a
// history well past a tiny configured context window and checks that
// compaction fires, the result still passes Validate() (CLAUDE.md
// invariant 3 — the one this task is most likely to break, since
// compaction discards history wholesale), and the synthetic summary
// message is what the (extra) summarization call returned.
func TestMaybeCompactTriggersAboveThresholdAndKeepsHistoryValid(t *testing.T) {
	h := history.New()
	for i := 0; i < 10; i++ {
		h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "a fairly long question to pad out the estimated token count here"}}})
		h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "a fairly long answer to pad out the estimated token count here too"}}})
	}

	provider := &compactionScriptedProvider{
		scriptedProvider: &scriptedProvider{responses: []*llm.Response{textResponse("final reply")}},
		summary:          "everything before this was summarized",
	}
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)
	a.SetCompactionPolicy(CompactionPolicy{ContextWindow: 100, TriggerFraction: 0.5, KeepTurns: 2})

	if _, err := a.Run(context.Background(), "one more question"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if err := h.Validate(); err != nil {
		t.Fatalf("Validate() after compaction error = %v", err)
	}

	msgs := h.Snapshot()
	if len(msgs) == 0 {
		t.Fatal("history is empty after Run()")
	}
	first := msgs[0]
	text, ok := first.Blocks[0].(llm.Text)
	if !ok || text.Text != "everything before this was summarized" || first.Role != llm.RoleAssistant {
		t.Errorf("messages[0] = %+v, want the synthetic assistant summary", first)
	}
}

func TestMaybeCompactDoesNothingBelowThreshold(t *testing.T) {
	h := history.New()
	h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}})
	h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "hello"}}})
	before := h.Snapshot()

	provider := &scriptedProvider{responses: []*llm.Response{textResponse("ok")}}
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)
	a.SetCompactionPolicy(CompactionPolicy{ContextWindow: 1_000_000, TriggerFraction: 0.9, KeepTurns: 2})

	if _, err := a.Run(context.Background(), "another question"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got := h.Snapshot()
	if len(got) != len(before)+2 {
		t.Fatalf("history has %d messages, want %d (before + this turn's user/assistant pair, no compaction)", len(got), len(before)+2)
	}
	if got[0].Role != before[0].Role {
		t.Errorf("messages[0].Role = %q, want the original first message untouched (%q) since nothing should have been compacted", got[0].Role, before[0].Role)
	}
}

func TestMaybeCompactSkippedWhenSummarizationFails(t *testing.T) {
	h := history.New()
	for i := 0; i < 10; i++ {
		h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "a fairly long question to pad out the estimated token count here"}}})
		h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "a fairly long answer to pad out the estimated token count here too"}}})
	}
	before := h.Snapshot()

	provider := &alwaysErrorsProvider{err: errors.New("provider unavailable")}
	a := New(provider, tools.NewRegistry(), h, "", 1024, Guards{}, 0, AutoApprove)
	a.SetCompactionPolicy(CompactionPolicy{ContextWindow: 100, TriggerFraction: 0.5, KeepTurns: 2})

	_, err := a.Run(context.Background(), "one more question")
	if err == nil {
		t.Fatal("Run() error = nil, want the turn's own Chat call to fail too, since this fake provider always errors")
	}

	got := h.Snapshot()
	if len(got) != len(before)+1 {
		t.Fatalf("history has %d messages, want %d (before + the new user turn Run appends before calling Chat)", len(got), len(before)+1)
	}
	if diffMsg := cmpMessage(got[0], before[0]); diffMsg != "" {
		t.Errorf("messages[0] changed after a failed summarization call, want it untouched (compaction should be skipped, not partially applied): %s", diffMsg)
	}
}

func cmpMessage(a, b llm.Message) string {
	if a.Role != b.Role {
		return fmt.Sprintf("Role: got %q, want %q", a.Role, b.Role)
	}
	at, aok := a.Blocks[0].(llm.Text)
	bt, bok := b.Blocks[0].(llm.Text)
	if !aok || !bok || at != bt {
		return fmt.Sprintf("Blocks[0]: got %+v, want %+v", a.Blocks[0], b.Blocks[0])
	}
	return ""
}

// alwaysErrorsProvider errors on every call, used to make maybeCompact's
// summarization call fail deterministically.
type alwaysErrorsProvider struct {
	err error
}

func (p *alwaysErrorsProvider) Name() string { return "always-errors" }

func (p *alwaysErrorsProvider) Chat(context.Context, llm.Request) (*llm.Response, error) {
	return nil, p.err
}

func (p *alwaysErrorsProvider) Stream(context.Context, llm.Request) (<-chan llm.Event, error) {
	return nil, p.err
}
