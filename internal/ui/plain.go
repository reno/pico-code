package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/reno/pico-code/internal/llm"
)

// PlainRenderer writes assistant text to Out as it arrives and nothing
// else beyond a couple of one-line tool status markers — no ANSI escapes,
// no cursor movement — so it is safe piped, redirected, or read by another
// program. It is the default whenever stdout is not a terminal or --tui is
// off.
type PlainRenderer struct {
	Out io.Writer
}

// Render implements Renderer. It reconstructs Message.Blocks the same way
// llm.CollectStream does, writing each TextDelta's text to Out as it
// arrives rather than only after the fact.
func (p PlainRenderer) Render(ctx context.Context, events <-chan llm.Event) (*llm.Response, error) {
	var blocks []llm.Block
	var text strings.Builder
	textOpen := false
	wroteAny := false
	names := map[string]string{}

	flushText := func() {
		if textOpen {
			blocks = append(blocks, llm.Text{Text: text.String()})
			text.Reset()
			textOpen = false
		}
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case e, ok := <-events:
			if !ok {
				return nil, fmt.Errorf("ui: event stream closed before a MessageDone event")
			}
			switch v := e.(type) {
			case llm.TextDelta:
				textOpen = true
				text.WriteString(v.Text)
				if _, err := fmt.Fprint(p.Out, v.Text); err != nil {
					return nil, fmt.Errorf("ui: write output: %w", err)
				}
				wroteAny = true
			case llm.ToolUseStart:
				flushText()
				names[v.ID] = v.Name
			case llm.ToolUseArgsDelta:
				// Informational only; ToolUseDone carries the finished value.
			case llm.ToolUseDone:
				blocks = append(blocks, llm.ToolUse{ID: v.ID, Name: names[v.ID], Input: v.Input})
			case llm.MessageDone:
				flushText()
				if wroteAny {
					if _, err := fmt.Fprintln(p.Out); err != nil {
						return nil, fmt.Errorf("ui: write output: %w", err)
					}
				}
				return &llm.Response{
					Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
					StopReason: v.StopReason,
					Usage:      v.Usage,
				}, nil
			case llm.Error:
				return nil, v.Err
			default:
				return nil, fmt.Errorf("ui: unknown event type %T", e)
			}
		}
	}
}

// ToolStarted implements ToolStatusReporter.
func (p PlainRenderer) ToolStarted(_, name string, _ json.RawMessage) {
	_, _ = fmt.Fprintf(p.Out, "→ %s\n", name)
}

// ToolFinished implements ToolStatusReporter.
func (p PlainRenderer) ToolFinished(_, name, _ string, isError bool) {
	if isError {
		_, _ = fmt.Fprintf(p.Out, "✗ %s\n", name)
		return
	}
	_, _ = fmt.Fprintf(p.Out, "✓ %s\n", name)
}

var (
	_ Renderer           = PlainRenderer{}
	_ ToolStatusReporter = PlainRenderer{}
)
