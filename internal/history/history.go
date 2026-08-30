// Package history stores the canonical conversation and enforces CLAUDE.md
// invariant 3: every ToolUse a model issues must be answered, in the very
// next message, by exactly one matching ToolResult per call, in call order.
package history

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/reno/pico-code/internal/llm"
)

// History is an ordered, append-only record of canonical messages.
type History struct {
	messages []llm.Message
}

// New returns an empty History.
func New() *History {
	return &History{}
}

// Append adds m to the end of history.
func (h *History) Append(m llm.Message) {
	h.messages = append(h.messages, m)
}

// Snapshot returns a copy of the current messages, safe for the caller to
// mutate or hand to a Request without aliasing History's internal slice.
func (h *History) Snapshot() []llm.Message {
	out := make([]llm.Message, len(h.messages))
	copy(out, h.messages)
	return out
}

// Validate enforces invariant 3. It walks the messages once: any assistant
// message's ToolUse blocks must be answered by the very next message — a
// user message containing exactly those ToolResult IDs, in the same order —
// and any ToolResult block must be preceded by such a ToolUse message, so a
// dropped, duplicated, reordered, or orphaned result is caught either way.
func (h *History) Validate() error {
	for i, m := range h.messages {
		calls := toolUseIDs(m)
		if len(calls) > 0 && m.Role != llm.RoleAssistant {
			return fmt.Errorf("history: message %d has ToolUse block(s) %v but role is %q, want %q", i, calls, m.Role, llm.RoleAssistant)
		}

		if hasToolResult(m) && !precededByToolUse(h.messages, i) {
			return fmt.Errorf("history: message %d has ToolResult block(s) with no preceding ToolUse to answer", i)
		}

		if len(calls) == 0 {
			continue
		}
		if i+1 >= len(h.messages) {
			return fmt.Errorf("history: message %d (assistant) has %d ToolUse block(s) %v with no following message", i, len(calls), calls)
		}
		if err := validatePairing(calls, h.messages[i+1]); err != nil {
			return fmt.Errorf("history: message %d ToolUse -> message %d: %w", i, i+1, err)
		}
	}
	return nil
}

// Save validates h and writes it to path as JSON. It refuses to write an
// invalid history rather than let a violating state reach disk.
func (h *History) Save(path string) error {
	if err := h.Validate(); err != nil {
		return fmt.Errorf("history: refusing to save invalid history: %w", err)
	}
	data, err := json.Marshal(h.messages)
	if err != nil {
		return fmt.Errorf("history: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("history: write %s: %w", path, err)
	}
	return nil
}

// Load reads a History previously written by Save and validates it before
// returning, so a corrupted or hand-edited file can never be replayed in a
// violating state.
func Load(path string) (*History, error) {
	msgs, err := loadMessages(path)
	if err != nil {
		return nil, err
	}
	return &History{messages: msgs}, nil
}

// LoadInto replaces h's messages with the ones stored at path, in place —
// unlike Load, which returns a new History — so callers holding a shared
// *History (e.g. an already-constructed Agent) can point it at a different
// session without reconstructing anything downstream. Like Load, it
// validates before committing: on error h is left completely untouched,
// never partially loaded.
func (h *History) LoadInto(path string) error {
	msgs, err := loadMessages(path)
	if err != nil {
		return err
	}
	h.messages = msgs
	return nil
}

// Reset clears h back to empty, in place.
func (h *History) Reset() {
	h.messages = nil
}

func loadMessages(path string) ([]llm.Message, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("history: read %s: %w", path, err)
	}
	var msgs []llm.Message
	if err := json.Unmarshal(data, &msgs); err != nil {
		return nil, fmt.Errorf("history: unmarshal %s: %w", path, err)
	}
	h := &History{messages: msgs}
	if err := h.Validate(); err != nil {
		return nil, fmt.Errorf("history: loaded invalid history from %s: %w", path, err)
	}
	return msgs, nil
}

func toolUseIDs(m llm.Message) []string {
	var ids []string
	for _, b := range m.Blocks {
		if tu, ok := b.(llm.ToolUse); ok {
			ids = append(ids, tu.ID)
		}
	}
	return ids
}

func hasToolResult(m llm.Message) bool {
	for _, b := range m.Blocks {
		if _, ok := b.(llm.ToolResult); ok {
			return true
		}
	}
	return false
}

func precededByToolUse(msgs []llm.Message, i int) bool {
	if i == 0 {
		return false
	}
	prev := msgs[i-1]
	return prev.Role == llm.RoleAssistant && len(toolUseIDs(prev)) > 0
}

// validatePairing checks that next is a user message containing exactly one
// ToolResult per entry in calls, in the same order.
func validatePairing(calls []string, next llm.Message) error {
	if next.Role != llm.RoleUser {
		return fmt.Errorf("want a user message with %d ToolResult block(s), got role %q", len(calls), next.Role)
	}

	results := make([]string, len(next.Blocks))
	for i, b := range next.Blocks {
		tr, ok := b.(llm.ToolResult)
		if !ok {
			return fmt.Errorf("want %d ToolResult block(s): block %d is a %T, not ToolResult", len(calls), i, b)
		}
		results[i] = tr.ToolUseID
	}

	if len(results) < len(calls) {
		return fmt.Errorf("missing ToolResult(s) for %v: got %d result(s), want %d", calls[len(results):], len(results), len(calls))
	}
	if len(results) > len(calls) {
		return fmt.Errorf("unexpected extra ToolResult(s) %v: got %d result(s), want %d", results[len(calls):], len(results), len(calls))
	}
	for i, id := range calls {
		if results[i] != id {
			return fmt.Errorf("position %d: want ToolResult for %q (call order), got %q", i, id, results[i])
		}
	}
	return nil
}
