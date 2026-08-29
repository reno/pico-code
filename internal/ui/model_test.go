package ui

import (
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func newTestModel() Model {
	m := NewModel(make(chan string, 1), BannerInfo{})
	rm, _ := m.handleResize(tea.WindowSizeMsg{Width: 80, Height: 24})
	return rm
}

func update(m Model, msg tea.Msg) Model {
	next, _ := m.Update(msg)
	return next.(Model)
}

// TestModelStateTransitionsIdleStreamingToolIdle is 7.2's AC: the model
// layer has unit tests for the idle -> streaming -> tool -> idle cycle.
func TestModelStateTransitionsIdleStreamingToolIdle(t *testing.T) {
	m := newTestModel()
	if m.state != stateIdle {
		t.Fatalf("initial state = %v, want stateIdle", m.state)
	}

	m = update(m, turnStartedMsg{cancel: func() {}})
	if m.state != stateStreaming {
		t.Fatalf("after turnStartedMsg state = %v, want stateStreaming", m.state)
	}

	m = update(m, textDeltaMsg{text: "hello"})
	if m.state != stateStreaming {
		t.Fatalf("after textDeltaMsg state = %v, want stateStreaming", m.state)
	}
	if got := m.liveText; got != "hello" {
		t.Errorf("liveText = %q, want %q", got, "hello")
	}

	m = update(m, toolStartedMsg{id: "t1", name: "read_file"})
	if m.state != stateToolRunning {
		t.Fatalf("after toolStartedMsg state = %v, want stateToolRunning", m.state)
	}

	m = update(m, toolFinishedMsg{id: "t1", name: "read_file", output: "contents", isError: false})
	if m.state != stateStreaming {
		t.Fatalf("after the only running tool finishes, state = %v, want stateStreaming", m.state)
	}

	m = update(m, turnDoneMsg{text: "done"})
	if m.state != stateIdle {
		t.Fatalf("after turnDoneMsg state = %v, want stateIdle", m.state)
	}
	if m.cancel != nil {
		t.Error("cancel should be cleared once the turn is done")
	}
}

// TestModelAccumulatesMultipleTextDeltasAcrossSeparateUpdateCalls is a
// regression test for a real streaming crash: liveText used to be a
// strings.Builder value field, and Update has a value receiver, so each
// message (exactly like bubbletea's real dispatch, mirrored by the update()
// helper) runs against a fresh copy of Model. A Builder panics if written
// to from more than one copy of the struct it lives in — which is exactly
// what any turn streaming more than one text chunk did.
func TestModelAccumulatesMultipleTextDeltasAcrossSeparateUpdateCalls(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})
	m = update(m, textDeltaMsg{text: "hel"})
	m = update(m, textDeltaMsg{text: "lo "})
	m = update(m, textDeltaMsg{text: "world"})

	if got := m.liveText; got != "hello world" {
		t.Fatalf("liveText = %q, want %q", got, "hello world")
	}
}

func TestModelToolRunningPersistsUntilLastParallelToolFinishes(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})
	m = update(m, toolStartedMsg{id: "t1", name: "a"})
	m = update(m, toolStartedMsg{id: "t2", name: "b"})
	if m.state != stateToolRunning {
		t.Fatalf("state = %v, want stateToolRunning", m.state)
	}

	m = update(m, toolFinishedMsg{id: "t1", name: "a", output: "ok"})
	if m.state != stateToolRunning {
		t.Fatalf("state = %v, want stateToolRunning while t2 is still running", m.state)
	}

	m = update(m, toolFinishedMsg{id: "t2", name: "b", output: "ok"})
	if m.state != stateStreaming {
		t.Fatalf("state = %v, want stateStreaming once every tool has finished", m.state)
	}
}

func TestModelApprovalRequestSuspendsAndRestoresPriorState(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})

	resp := make(chan bool, 1)
	m = update(m, approvalRequestMsg{toolName: "run_command", resp: resp})
	if m.state != stateApproval {
		t.Fatalf("state = %v, want stateApproval", m.state)
	}
	if m.approval == nil || m.approval.toolName != "run_command" {
		t.Fatalf("approval = %+v, want a pending request for run_command", m.approval)
	}

	m = update(m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")})
	select {
	case ok := <-resp:
		if !ok {
			t.Error("resp = false, want true for a 'y' answer")
		}
	default:
		t.Fatal("expected an answer to be sent on resp")
	}
	if m.state != stateStreaming {
		t.Fatalf("state after answering = %v, want it restored to stateStreaming", m.state)
	}
}

func TestModelApprovalQueueServesOneAtATime(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})

	resp1 := make(chan bool, 1)
	resp2 := make(chan bool, 1)
	m = update(m, approvalRequestMsg{toolName: "first", resp: resp1})
	m = update(m, approvalRequestMsg{toolName: "second", resp: resp2})

	if m.approval.toolName != "first" {
		t.Fatalf("active approval = %q, want %q", m.approval.toolName, "first")
	}

	m = update(m, tea.KeyMsg{Type: tea.KeyEsc})
	if ok := <-resp1; ok {
		t.Error("Esc should deny, got true")
	}
	if m.approval == nil || m.approval.toolName != "second" {
		t.Fatalf("active approval after answering first = %+v, want %q next", m.approval, "second")
	}
	if m.state != stateApproval {
		t.Fatalf("state = %v, want stateApproval while a second request is queued", m.state)
	}
}

func TestModelTurnDoneWithErrorReturnsToIdleAndRecordsErr(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})
	wantErr := errors.New("boom")
	m = update(m, turnDoneMsg{err: wantErr})

	if m.state != stateIdle {
		t.Fatalf("state = %v, want stateIdle", m.state)
	}
	if !errors.Is(m.err, wantErr) {
		t.Errorf("err = %v, want %v", m.err, wantErr)
	}
}

func TestModelCtrlDQuitsAndCancelsAnyInFlightTurn(t *testing.T) {
	m := newTestModel()
	cancelled := false
	m = update(m, turnStartedMsg{cancel: func() { cancelled = true }})

	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlD})
	if !cancelled {
		t.Error("Ctrl+D during a turn should cancel it")
	}
	if cmd == nil {
		t.Fatal("Ctrl+D should return a quit command")
	}
}

func TestModelCtrlCCancelsTurnWithoutQuitting(t *testing.T) {
	m := newTestModel()
	cancelled := false
	m = update(m, turnStartedMsg{cancel: func() { cancelled = true }})

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = next.(Model)
	if !cancelled {
		t.Error("Ctrl+C during a turn should cancel it")
	}
	if cmd != nil {
		t.Error("Ctrl+C should not quit the program")
	}
	if m.state != stateStreaming {
		t.Errorf("state after Ctrl+C = %v, want it to stay stateStreaming (cancellation arrives via turnDoneMsg)", m.state)
	}
}

func TestModelEnterSubmitsInputOnlyWhenIdle(t *testing.T) {
	ch := make(chan string, 1)
	m := NewModel(ch, BannerInfo{})
	m, _ = m.handleResize(tea.WindowSizeMsg{Width: 80, Height: 24})
	m.textarea.SetValue("hello agent")

	m = update(m, tea.KeyMsg{Type: tea.KeyEnter})

	select {
	case got := <-ch:
		if got != "hello agent" {
			t.Errorf("submitted = %q, want %q", got, "hello agent")
		}
	default:
		t.Fatal("expected Enter in stateIdle to submit the textarea's value")
	}
	if m.textarea.Value() != "" {
		t.Errorf("textarea value after submit = %q, want it cleared", m.textarea.Value())
	}
}

// TestModelCommandOutputAppendsToTranscriptWithoutChangingState proves a
// slash command's output (sent via CommandOutput/commandOutputMsg) shows
// up in the transcript without going through the turn state machine — no
// spinner, no state change, since cmd/pico's driver never sends
// turnStartedMsg for a command.
func TestModelCommandOutputAppendsToTranscriptWithoutChangingState(t *testing.T) {
	m := newTestModel()
	if m.state != stateIdle {
		t.Fatalf("initial state = %v, want stateIdle", m.state)
	}

	m = update(m, commandOutputMsg{text: "> /usage\ncumulative: 5 input, 2 output token(s)\n"})

	if m.state != stateIdle {
		t.Errorf("state after commandOutputMsg = %v, want it to stay stateIdle", m.state)
	}
	if !strings.Contains(m.completed, "cumulative: 5 input, 2 output token(s)") {
		t.Errorf("completed = %q, want it to contain the command output", m.completed)
	}
}
