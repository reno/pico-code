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

	// wroteAny records whether the most recent Render call actually wrote
	// any TextDelta text to Out, read back via WroteAny() by a caller
	// (cmd/pico's streaming turn runner) that needs to know whether a
	// non-empty final Response.Message text was already shown live or
	// still needs printing — true for an ordinary reply (all its text
	// streamed as it arrived), false when the round produced no TextDelta
	// events at all, e.g. a guard-trip or empty-reply explanation (16.1's
	// --think can trigger the latter by exhausting the budget on
	// thinking) synthesized by the agent loop after Render already
	// returned, never itself passed through this renderer.
	wroteAny bool
}

// WroteAny reports whether the most recent Render call wrote any text to
// Out.
func (p *PlainRenderer) WroteAny() bool {
	return p.wroteAny
}

// Render implements Renderer. It reconstructs Message.Blocks the same way
// llm.CollectStream does, writing each TextDelta's text to Out as it
// arrives rather than only after the fact. ThinkingDelta text is folded
// into the reconstructed Message the same way, but never written to Out:
// no thinking text ever reaches piped stdout (16.3).
func (p *PlainRenderer) Render(ctx context.Context, events <-chan llm.Event) (*llm.Response, error) {
	p.wroteAny = false
	var blocks []llm.Block
	var thinking strings.Builder
	var text strings.Builder
	thinkingOpen := false
	textOpen := false
	names := map[string]string{}

	flushThinking := func() {
		if thinkingOpen {
			blocks = append(blocks, llm.Thinking{Text: thinking.String()})
			thinking.Reset()
			thinkingOpen = false
		}
	}
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
			case llm.ThinkingDelta:
				thinkingOpen = true
				thinking.WriteString(v.Text)
			case llm.TextDelta:
				flushThinking()
				textOpen = true
				text.WriteString(v.Text)
				if _, err := fmt.Fprint(p.Out, v.Text); err != nil {
					return nil, fmt.Errorf("ui: write output: %w", err)
				}
				p.wroteAny = true
			case llm.ToolUseStart:
				flushThinking()
				flushText()
				names[v.ID] = v.Name
			case llm.ToolUseArgsDelta:
				// Informational only; ToolUseDone carries the finished value.
			case llm.ToolUseDone:
				blocks = append(blocks, llm.ToolUse{ID: v.ID, Name: names[v.ID], Input: v.Input})
			case llm.MessageDone:
				flushThinking()
				flushText()
				if p.wroteAny {
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
func (p *PlainRenderer) ToolStarted(_, name string, _ json.RawMessage) {
	_, _ = fmt.Fprintf(p.Out, "→ %s\n", name)
}

// ToolFinished implements ToolStatusReporter.
func (p *PlainRenderer) ToolFinished(_, name, _ string, isError bool) {
	if isError {
		_, _ = fmt.Fprintf(p.Out, "✗ %s\n", name)
		return
	}
	_, _ = fmt.Fprintf(p.Out, "✓ %s\n", name)
}

var (
	_ Renderer           = (*PlainRenderer)(nil)
	_ ToolStatusReporter = (*PlainRenderer)(nil)
)
