package ollama

import (
	"strings"

	"github.com/reno/pico-code/internal/llm"
)

// estimateTokens approximates a token count from character length for use
// when Ollama's response omits real counts (prompt_eval_count/eval_count
// come back 0 whenever it skips evaluation, e.g. a fully cached prompt).
// ~4 characters per token is the same rough rule of thumb OpenAI's
// tokenizer docs use for English text — good enough for a fallback
// estimate, not for billing.
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	if n := len(s) / 4; n > 0 {
		return n
	}
	return 1
}

// estimateUsage fills in any zero field of reported with a character-based
// estimate derived from req's text and the response's own text, so
// cumulative usage tracking (8.1) keeps making progress against an Ollama
// version or model that omits real counts.
func estimateUsage(req llm.Request, responseText string, reported llm.Usage) llm.Usage {
	usage := reported
	if usage.InputTokens == 0 {
		usage.InputTokens = estimateTokens(requestText(req))
	}
	if usage.OutputTokens == 0 {
		usage.OutputTokens = estimateTokens(responseText)
	}
	return usage
}

func requestText(req llm.Request) string {
	var b strings.Builder
	b.WriteString(req.System)
	for _, m := range req.Messages {
		for _, blk := range m.Blocks {
			switch v := blk.(type) {
			case llm.Text:
				b.WriteString(v.Text)
			case llm.ToolResult:
				b.WriteString(v.Content)
			case llm.ToolUse:
				b.Write(v.Input)
			}
		}
	}
	return b.String()
}
