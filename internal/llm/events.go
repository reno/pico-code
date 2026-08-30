package llm

import (
	"context"
	"encoding/json"
)

// Event is one item in a Provider's Stream. It is sealed to this package
// (via the unexported isEvent method), mirroring Block, so a consumer's
// type switch is exhaustive by construction.
type Event interface {
	isEvent()
}

// TextDelta is an incremental chunk of assistant text.
type TextDelta struct {
	Text string
}

func (TextDelta) isEvent() {}

// ThinkingDelta is an incremental chunk of the model's reasoning trace,
// emitted (if the provider produces one at all) before any TextDelta for
// the same message (16.2).
type ThinkingDelta struct {
	Text string
}

func (ThinkingDelta) isEvent() {}

// ToolUseStart announces a new tool call; its arguments follow as
// ToolUseArgsDelta events and finish with a matching ToolUseDone.
type ToolUseStart struct {
	ID   string
	Name string
}

func (ToolUseStart) isEvent() {}

// ToolUseArgsDelta is a raw fragment of a tool call's JSON arguments.
// Fragments must only be concatenated, never parsed individually — CLAUDE.md:
// "Accumulate the raw string and json.Unmarshal exactly once at block stop."
type ToolUseArgsDelta struct {
	ID      string
	Partial string
}

func (ToolUseArgsDelta) isEvent() {}

// ToolUseDone finalizes a tool call with its fully assembled input.
type ToolUseDone struct {
	ID    string
	Input json.RawMessage
}

func (ToolUseDone) isEvent() {}

// MessageDone marks the end of the assistant's turn.
type MessageDone struct {
	StopReason string
	Usage      Usage
}

func (MessageDone) isEvent() {}

// Error carries a terminal stream failure. A Provider sends at most one,
// always as its last event before the channel closes.
type Error struct {
	Err error
}

func (Error) isEvent() {}

// StreamEvents runs produce in its own goroutine and returns the channel it
// sends to, closing that channel exactly once when produce returns —
// whether that's because it finished normally or because send reported ctx
// was cancelled. Every Provider.Stream implementation should build its
// channel through this rather than managing goroutine/channel lifecycle
// itself, so the "closed exactly once, exits on cancel" contract lives in
// one place.
func StreamEvents(ctx context.Context, produce func(send func(Event) bool)) <-chan Event {
	ch := make(chan Event)
	go func() {
		defer close(ch)
		produce(func(e Event) bool {
			select {
			case ch <- e:
				return true
			case <-ctx.Done():
				return false
			}
		})
	}()
	return ch
}
