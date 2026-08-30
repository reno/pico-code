package ollama

import (
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

func TestEstimateUsageLeavesReportedCountsAlone(t *testing.T) {
	req := llm.Request{System: "be concise", Messages: []llm.Message{
		{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello there"}}},
	}}
	reported := llm.Usage{InputTokens: 512, OutputTokens: 64}

	got := estimateUsage(req, "hi", reported)
	if got != reported {
		t.Errorf("estimateUsage() = %+v, want reported counts unchanged: %+v", got, reported)
	}
}

// TestEstimateUsageFillsInZeroCounts is 8.1's AC for Ollama: when a
// response omits its real counts, estimation still produces a non-zero
// usage so cumulative tracking keeps making progress.
func TestEstimateUsageFillsInZeroCounts(t *testing.T) {
	req := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "this prompt has more than a few characters in it"}}},
	}}

	got := estimateUsage(req, "a fairly short reply", llm.Usage{})
	if got.InputTokens <= 0 {
		t.Errorf("InputTokens = %d, want a positive estimate", got.InputTokens)
	}
	if got.OutputTokens <= 0 {
		t.Errorf("OutputTokens = %d, want a positive estimate", got.OutputTokens)
	}
}

func TestEstimateUsagePartialZeroOnlyFillsMissingField(t *testing.T) {
	req := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello"}}},
	}}

	got := estimateUsage(req, "a reply with several words in it", llm.Usage{InputTokens: 10})
	if got.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want the reported 10 preserved", got.InputTokens)
	}
	if got.OutputTokens <= 0 {
		t.Errorf("OutputTokens = %d, want a positive estimate for the omitted field", got.OutputTokens)
	}
}

func TestEstimateTokensEmptyStringIsZero(t *testing.T) {
	if got := estimateTokens(""); got != 0 {
		t.Errorf("estimateTokens(\"\") = %d, want 0", got)
	}
}
