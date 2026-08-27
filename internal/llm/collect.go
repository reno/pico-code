package llm

import (
	"context"
	"fmt"
	"strings"
)

// CollectStream drains ch into a Response, reconstructing Message.Blocks in
// the same shape a non-streaming call would produce: TextDelta events
// accumulate into the currently open Text block (a new one starts whenever
// a ToolUseStart/ToolUseDone pair intervenes), and each ToolUseDone
// contributes one ToolUse block using the Name its ToolUseStart announced.
// It returns an error if ch yields an Error event, closes before a
// MessageDone arrives, or ctx is cancelled first.
func CollectStream(ctx context.Context, ch <-chan Event) (*Response, error) {
	var blocks []Block
	var text strings.Builder
	textOpen := false
	names := map[string]string{}

	flushText := func() {
		if textOpen {
			blocks = append(blocks, Text{Text: text.String()})
			text.Reset()
			textOpen = false
		}
	}

	for {
		select {
		case e, ok := <-ch:
			if !ok {
				return nil, fmt.Errorf("llm: stream closed before a MessageDone event")
			}
			switch v := e.(type) {
			case TextDelta:
				textOpen = true
				text.WriteString(v.Text)
			case ToolUseStart:
				flushText()
				names[v.ID] = v.Name
			case ToolUseArgsDelta:
				// Informational only; the adapter accumulates internally
				// and hands back the finished value in ToolUseDone.
			case ToolUseDone:
				blocks = append(blocks, ToolUse{ID: v.ID, Name: names[v.ID], Input: v.Input})
			case MessageDone:
				flushText()
				return &Response{
					Message:    Message{Role: RoleAssistant, Blocks: blocks},
					StopReason: v.StopReason,
					Usage:      v.Usage,
				}, nil
			case Error:
				return nil, v.Err
			default:
				return nil, fmt.Errorf("llm: unknown event type %T", e)
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
