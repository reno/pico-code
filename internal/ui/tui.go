package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/reno/pico-code/internal/llm"
)

// The bubbletea messages driving Model.Update. They are unexported: only
// TUIRenderer and TUIApprover in this package construct them, sending each
// through the *tea.Program the caller owns — cmd/pico never touches these
// directly, only wires Model, TUIRenderer, and TUIApprover together.
type (
	turnStartedMsg   struct{ cancel context.CancelFunc }
	textDeltaMsg     struct{ text string }
	thinkingDeltaMsg struct{ text string }
	toolStartedMsg   struct {
		id, name string
		input    json.RawMessage
	}
	toolFinishedMsg struct {
		id, name, output string
		isError          bool
	}
	subToolStartedMsg struct {
		parentID, id, name string
		input              json.RawMessage
	}
	subToolFinishedMsg struct {
		parentID, id, name, output string
		isError                    bool
	}
	turnDoneMsg struct {
		text string
		err  error
	}
	// turnTickMsg drives the status line's once-per-second elapsed-time
	// display. It self-schedules (see tickEvery) only while a turn is in
	// flight, so it stops on its own once the turn ends.
	turnTickMsg        struct{}
	commandOutputMsg   struct{ text string }
	clearScrollbackMsg struct{}
	approvalRequestMsg struct {
		toolName string
		input    json.RawMessage
		preview  string
		resp     chan<- bool
	}
)

// TUIRenderer implements Renderer and ToolStatusReporter by forwarding
// every event to a *tea.Program as a message, so Model.Update can render
// it — the same block-reconstruction PlainRenderer does, plus the side
// effect of a Send instead of a Fprint.
type TUIRenderer struct {
	Program *tea.Program
}

// Render implements Renderer. ThinkingDelta text is folded into the
// reconstructed Message the same way TextDelta is, ahead of the Text
// block, and forwarded as its own message so Model can render it as a
// separate, collapsible block (16.3) rather than mixed into the reply.
func (r *TUIRenderer) Render(ctx context.Context, events <-chan llm.Event) (*llm.Response, error) {
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
				r.Program.Send(thinkingDeltaMsg{text: v.Text})
			case llm.TextDelta:
				flushThinking()
				textOpen = true
				text.WriteString(v.Text)
				r.Program.Send(textDeltaMsg{text: v.Text})
			case llm.ToolUseStart:
				flushThinking()
				flushText()
				names[v.ID] = v.Name
			case llm.ToolUseArgsDelta:
			case llm.ToolUseDone:
				blocks = append(blocks, llm.ToolUse{ID: v.ID, Name: names[v.ID], Input: v.Input})
			case llm.MessageDone:
				flushThinking()
				flushText()
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
func (r *TUIRenderer) ToolStarted(id, name string, input json.RawMessage) {
	r.Program.Send(toolStartedMsg{id: id, name: name, input: input})
}

// ToolFinished implements ToolStatusReporter.
func (r *TUIRenderer) ToolFinished(id, name, output string, isError bool) {
	r.Program.Send(toolFinishedMsg{id: id, name: name, output: output, isError: isError})
}

// SubToolStarted implements SubToolStatusReporter.
func (r *TUIRenderer) SubToolStarted(parentID, id, name string, input json.RawMessage) {
	r.Program.Send(subToolStartedMsg{parentID: parentID, id: id, name: name, input: input})
}

// SubToolFinished implements SubToolStatusReporter.
func (r *TUIRenderer) SubToolFinished(parentID, id, name, output string, isError bool) {
	r.Program.Send(subToolFinishedMsg{parentID: parentID, id: id, name: name, output: output, isError: isError})
}

var (
	_ Renderer              = (*TUIRenderer)(nil)
	_ ToolStatusReporter    = (*TUIRenderer)(nil)
	_ SubToolStatusReporter = (*TUIRenderer)(nil)
)

// TUIApprover implements the agent.Approver contract (matched structurally
// to avoid this package importing internal/agent) by showing a modal in
// the TUI and blocking until it's answered or ctx is cancelled. Approve
// runs on the agent loop's tool goroutine, never on the tea.Program's own
// goroutine, so blocking here does not freeze the UI.
type TUIApprover struct {
	Program *tea.Program
}

// Approve sends an approval request to the running Model and waits for its
// answer.
func (a *TUIApprover) Approve(ctx context.Context, toolName string, input json.RawMessage, preview string) (bool, error) {
	resp := make(chan bool, 1)
	a.Program.Send(approvalRequestMsg{toolName: toolName, input: input, preview: preview, resp: resp})
	select {
	case ok := <-resp:
		return ok, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// TurnStarted announces a turn's start (and its cancel func, for Ctrl+C) to
// the running Model, letting cmd/pico's driver loop do so without
// constructing the unexported message type itself. TurnDone is its
// counterpart for a turn's end.
func TurnStarted(program *tea.Program, cancel context.CancelFunc) {
	program.Send(turnStartedMsg{cancel: cancel})
}

// TurnDone announces a turn's result, mirroring TurnStarted.
func TurnDone(program *tea.Program, text string, err error) {
	program.Send(turnDoneMsg{text: text, err: err})
}

// CommandOutput appends text (a slash command's result, e.g. /usage or
// /save) to the transcript. Unlike a turn, a command never touches the
// spinner/streaming state — cmd/pico's driver loop handles it inline,
// without ever sending TurnStarted for it.
func CommandOutput(program *tea.Program, text string) {
	program.Send(commandOutputMsg{text: text})
}

// ClearScrollback wipes the transcript /clear renders to (never the agent's
// history — that lives in internal/history, untouched here), for the
// /clear command in the TUI.
func ClearScrollback(program *tea.Program) {
	program.Send(clearScrollbackMsg{})
}
