// Package ui renders an agent turn for a person: it turns a Provider's
// event stream (and the agent loop's tool-execution outcomes, which never
// appear on that stream since running a tool is loop policy, not provider
// transport) into terminal output. Two implementations exist: Plain, the
// default for piped/non-interactive use, and a bubbletea TUI.
package ui

import (
	"context"
	"encoding/json"

	"github.com/reno/pico-code/internal/llm"
)

// Renderer turns one turn's provider event stream into user-facing output,
// returning the assembled Response for the agent loop to append to
// history — the same contract llm.CollectStream has, plus the rendering
// side effect.
type Renderer interface {
	Render(ctx context.Context, events <-chan llm.Event) (*llm.Response, error)
}

// ToolStatusReporter is an optional Renderer capability for showing a tool
// call's progress. The agent loop calls it, when implemented, around
// actually running a tool — a step that happens after Render returns, so it
// cannot be expressed as an llm.Event.
type ToolStatusReporter interface {
	ToolStarted(id, name string, input json.RawMessage)
	ToolFinished(id, name, output string, isError bool)
}

// SubToolStatusReporter is an optional ToolStatusReporter capability for a
// tool call made by a nested loop (a sub-agent) rather than the top-level
// agent loop. parentID names the top-level ToolUse — the sub_agent call —
// it nests under, so a renderer can group it under that call instead of
// interleaving it into the flat transcript the way a top-level call is.
type SubToolStatusReporter interface {
	SubToolStarted(parentID, id, name string, input json.RawMessage)
	SubToolFinished(parentID, id, name, output string, isError bool)
}
