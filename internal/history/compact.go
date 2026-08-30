package history

import (
	"fmt"

	"github.com/reno/pico-code/internal/llm"
)

// CompactionBoundary returns the index of the first message that must
// survive compaction verbatim: the start of the keepTurns most recent user
// turns. A "turn" starts at a user message carrying a Text block — the
// message the agent loop appends for actual user input, as opposed to the
// user messages carrying only ToolResult blocks that continue the same
// turn. Because a genuine turn can only start once every ToolUse earlier in
// the conversation has already been answered (CLAUDE.md invariant 3), any
// boundary this returns is automatically safe to cut at: it can never land
// between a ToolUse and its ToolResult.
//
// It returns 0 (nothing to compact) if there are keepTurns or fewer turns.
func CompactionBoundary(messages []llm.Message, keepTurns int) int {
	if keepTurns <= 0 {
		return 0
	}
	found := 0
	for i := len(messages) - 1; i >= 0; i-- {
		if isTurnStart(messages[i]) {
			found++
			if found == keepTurns {
				return i
			}
		}
	}
	return 0
}

func isTurnStart(m llm.Message) bool {
	if m.Role != llm.RoleUser {
		return false
	}
	for _, b := range m.Blocks {
		if _, ok := b.(llm.Text); ok {
			return true
		}
	}
	return false
}

// Compact replaces every message before the boundary that keeps the last
// keepTurns user turns verbatim with a single synthetic assistant message
// containing summary. The synthetic message is Role assistant (not user)
// so it alternates cleanly with the kept messages, which always start at a
// genuine user turn. It refuses to leave history in a state that fails
// Validate() (CLAUDE.md invariant 3) rather than commit one — compaction is
// the one operation most likely to violate that invariant, since it
// discards history wholesale; the boundary construction above already
// makes that unreachable in practice, but this is the backstop.
func (h *History) Compact(keepTurns int, summary string) error {
	boundary := CompactionBoundary(h.messages, keepTurns)
	if boundary <= 0 {
		return nil
	}

	compacted := make([]llm.Message, 0, len(h.messages)-boundary+1)
	compacted = append(compacted, llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: summary}}})
	compacted = append(compacted, h.messages[boundary:]...)

	trial := &History{messages: compacted}
	if err := trial.Validate(); err != nil {
		return fmt.Errorf("history: refusing to compact into an invalid state: %w", err)
	}
	h.messages = compacted
	return nil
}

// EstimateTokens approximates messages' total size in tokens from
// character length (~4 characters per token, the same rough rule of thumb
// used for Ollama's own estimate in internal/llm/ollama) — good enough to
// decide when to compact, not for billing.
func EstimateTokens(messages []llm.Message) int {
	chars := 0
	for _, m := range messages {
		for _, b := range m.Blocks {
			switch v := b.(type) {
			case llm.Text:
				chars += len(v.Text)
			case llm.ToolUse:
				chars += len(v.Input)
			case llm.ToolResult:
				chars += len(v.Content)
			}
		}
	}
	if chars == 0 {
		return 0
	}
	if n := chars / 4; n > 0 {
		return n
	}
	return 1
}
