package llm

import (
	"context"
	"fmt"
	"strings"
)

// CollectStream drains ch into a Response, reconstructing Message.Blocks in
// the same shape a non-streaming call would produce: ThinkingDelta and
// TextDelta events each accumulate into their own currently open block (a
// new one starts whenever a ToolUseStart/ToolUseDone pair intervenes), with
// any Thinking block always flushed ahead of the Text block that follows
// it, and each ToolUseDone contributes one ToolUse block using the Name its
// ToolUseStart announced. It returns an error if ch yields an Error event,
// closes before a MessageDone arrives, or ctx is cancelled first.
func CollectStream(ctx context.Context, ch <-chan Event) (*Response, error) {
	var blocks []Block
	var thinking strings.Builder
	var text strings.Builder
	thinkingOpen := false
	textOpen := false
	names := map[string]string{}

	flushThinking := func() {
		if thinkingOpen {
			blocks = append(blocks, Thinking{Text: thinking.String()})
			thinking.Reset()
			thinkingOpen = false
		}
	}
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
			case ThinkingDelta:
				thinkingOpen = true
				thinking.WriteString(v.Text)
			case TextDelta:
				flushThinking()
				textOpen = true
				text.WriteString(v.Text)
			case ToolUseStart:
				flushThinking()
				flushText()
				names[v.ID] = v.Name
			case ToolUseArgsDelta:
				// Informational only; the adapter accumulates internally
				// and hands back the finished value in ToolUseDone.
			case ToolUseDone:
				blocks = append(blocks, ToolUse{ID: v.ID, Name: names[v.ID], Input: v.Input})
			case MessageDone:
				flushThinking()
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
