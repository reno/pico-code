package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/reno/pico-code/internal/llm"
)

func newTestModel() Model {
	m := NewModel(make(chan string, 1), BannerInfo{}, "dark")
	rm, _ := m.handleResize(tea.WindowSizeMsg{Width: 80, Height: 24})
	return rm
}

// TestNewModelDefaultsEmptyGlamourStyleToDark documents the fallback for
// callers that don't resolve a style themselves.
func TestNewModelDefaultsEmptyGlamourStyleToDark(t *testing.T) {
	m := NewModel(make(chan string, 1), BannerInfo{}, "")
	if m.glamourStyle != "dark" {
		t.Errorf("glamourStyle = %q, want the \"dark\" fallback for an empty style", m.glamourStyle)
	}
}

// TestHandleResizeUsesResolvedStyleNotAutoDetect is a regression test: the
// renderer used to be built with glamour.WithAutoStyle(), which queries the
// terminal's background color at render time. That query races with
// bubbletea's own raw-mode input reader once the program has taken over the
// terminal, leaking stray escape bytes onto the screen on every resize.
// handleResize must build the renderer from the style resolved once at
// NewModel time instead, for both possible values.
func TestHandleResizeUsesResolvedStyleNotAutoDetect(t *testing.T) {
	for _, style := range []string{"dark", "light"} {
		m := NewModel(make(chan string, 1), BannerInfo{}, style)
		updated, _ := m.handleResize(tea.WindowSizeMsg{Width: 80, Height: 24})
		if updated.renderer == nil {
			t.Errorf("style %q: renderer is nil after handleResize, want glamour.WithStandardStyle to succeed", style)
		}
		if updated.glamourStyle != style {
			t.Errorf("style %q: glamourStyle = %q after resize, want it unchanged", style, updated.glamourStyle)
		}
	}
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
	m := NewModel(ch, BannerInfo{}, "dark")
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

// TestModelStatusLineTracksElapsedAndTokens is 10.1's AC: a scripted
// tick/event sequence renders the documented status line, and it resets
// cleanly between turns instead of accumulating.
func TestModelStatusLineTracksElapsedAndTokens(t *testing.T) {
	m := newTestModel()
	if got := m.View(); !strings.Contains(got, "> ") {
		t.Fatalf("idle view = %q, want the plain prompt with no timer", got)
	}

	m = update(m, turnStartedMsg{cancel: func() {}})
	m = update(m, turnTickMsg{})
	m = update(m, turnTickMsg{})
	m = update(m, textDeltaMsg{text: strings.Repeat("x", 4800)}) // ~1200 estimated tokens

	want := fmt.Sprintf("%s… (2s · ↑ 1.2k tokens · esc to interrupt)", m.currentWord)
	if got := m.statusText(); got != want {
		t.Fatalf("statusText = %q, want %q", got, want)
	}

	m = update(m, turnDoneMsg{text: "done"})
	if got := m.View(); !strings.Contains(got, "> ") || strings.Contains(got, "esc to interrupt") {
		t.Fatalf("idle view after turn = %q, want the plain prompt with no timer", got)
	}

	// A second turn must not carry over the first turn's elapsed time or
	// token count.
	m = update(m, turnStartedMsg{cancel: func() {}})
	if want := fmt.Sprintf("%s… (0s · ↑ 0 tokens · esc to interrupt)", m.currentWord); m.statusText() != want {
		t.Fatalf("statusText at the start of a new turn = %q, want %q", m.statusText(), want)
	}
	m = update(m, turnTickMsg{})
	if want := fmt.Sprintf("%s… (1s · ↑ 0 tokens · esc to interrupt)", m.currentWord); m.statusText() != want {
		t.Fatalf("statusText = %q, want %q (no leakage from the previous turn)", m.statusText(), want)
	}
}

// TestModelTurnTickStopsOnceIdle proves a tick arriving after the turn has
// already ended (a race between the timer and turnDoneMsg) is a no-op
// rather than resurrecting the timer or corrupting state.
func TestModelTurnTickStopsOnceIdle(t *testing.T) {
	m := newTestModel()
	m = update(m, turnStartedMsg{cancel: func() {}})
	m = update(m, turnDoneMsg{text: "done"})

	next, cmd := m.Update(turnTickMsg{})
	m = next.(Model)
	if cmd != nil {
		t.Error("turnTickMsg while idle should not reschedule another tick")
	}
	if m.turnElapsed != 0 {
		t.Errorf("turnElapsed = %d, want 0 once idle", m.turnElapsed)
	}
}

// TestModelEscInterruptsTurnWithoutQuitting proves the status line's "esc
// to interrupt" hint is backed by real behavior: Esc during a turn cancels
// it the same way Ctrl+C does, and does not quit the program.
func TestModelEscInterruptsTurnWithoutQuitting(t *testing.T) {
	m := newTestModel()
	cancelled := false
	m = update(m, turnStartedMsg{cancel: func() { cancelled = true }})

	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = next.(Model)
	if !cancelled {
		t.Error("Esc during a turn should cancel it")
	}
	if cmd != nil {
		t.Error("Esc should not quit the program")
	}
	if m.state != stateStreaming {
		t.Errorf("state after Esc = %v, want it to stay stateStreaming (cancellation arrives via turnDoneMsg)", m.state)
	}
}

// TestTurnWordRotatesOnBoundariesAndNeverRepeats is 10.2's AC: with a fixed
// rand.Source, the word changes exactly on rotation boundaries, stays
// stable across every re-render in between, and never repeats itself on a
// rotation (including the boundary between two turns).
func TestTurnWordRotatesOnBoundariesAndNeverRepeats(t *testing.T) {
	m := newTestModel()
	m.wordRand = rand.New(rand.NewSource(1))

	m = update(m, turnStartedMsg{cancel: func() {}})
	seen := []string{m.currentWord}
	if seen[0] == "" {
		t.Fatal("currentWord after turnStartedMsg is empty, want a word from the vocabulary")
	}

	for rotation := 0; rotation < 5; rotation++ {
		stable := m.currentWord
		for tick := 1; tick < wordRotationTicks; tick++ {
			m = update(m, turnTickMsg{})
			if m.currentWord != stable {
				t.Fatalf("rotation %d tick %d: word changed to %q before the rotation boundary (want it to stay %q)", rotation, tick, m.currentWord, stable)
			}
			// A re-render from an unrelated message must not perturb the word.
			m = update(m, textDeltaMsg{text: "x"})
			if m.currentWord != stable {
				t.Fatalf("rotation %d: textDeltaMsg changed the word from %q to %q between rotations", rotation, stable, m.currentWord)
			}
		}
		m = update(m, turnTickMsg{}) // the rotation boundary itself
		if m.currentWord == stable {
			t.Fatalf("rotation %d: word %q did not change at the rotation boundary", rotation, stable)
		}
		seen = append(seen, m.currentWord)
	}

	for i := 1; i < len(seen); i++ {
		if seen[i] == seen[i-1] {
			t.Fatalf("word %q repeated back to back at rotation %d: %v", seen[i], i, seen)
		}
	}

	// Ending the turn and starting a new one must not repeat the last word
	// either — "never the same word twice in a row" spans turn boundaries.
	last := m.currentWord
	m = update(m, turnDoneMsg{text: "done"})
	m = update(m, turnStartedMsg{cancel: func() {}})
	if m.currentWord == last {
		t.Fatalf("first word of the new turn (%q) repeated the previous turn's last word", m.currentWord)
	}
}

// TestPlainRendererEmitsNoStatusLine is the other half of 10.2's AC: the
// non-TTY plain renderer never emits a status line, spinner, or turn word
// at all — only the assistant's own text and one-line tool markers.
func TestPlainRendererEmitsNoStatusLine(t *testing.T) {
	var buf strings.Builder
	p := PlainRenderer{Out: &buf}
	ch := make(chan llm.Event)
	go send(ch,
		llm.ToolUseStart{ID: "t1", Name: "read_file"},
		llm.TextDelta{Text: "hello"},
		llm.ToolUseDone{ID: "t1", Input: json.RawMessage(`{}`)},
		llm.MessageDone{StopReason: "end_turn"},
	)

	if _, err := p.Render(context.Background(), ch); err != nil {
		t.Fatalf("Render: %v", err)
	}
	got := buf.String()
	if strings.Contains(got, "esc to interrupt") {
		t.Errorf("plain output = %q, want no status-line artifacts", got)
	}
	for _, w := range turnWords {
		if strings.Contains(got, w) {
			t.Errorf("plain output = %q, contains turn word %q — the plain renderer must never render one", got, w)
		}
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

// TestModelClearScrollbackWipesTranscriptWithoutChangingState is 11.2's AC
// for /clear in the TUI: it resets the rendered transcript and nothing else
// (the agent's history lives outside Model entirely, so there's nothing
// here that could touch it).
func TestModelClearScrollbackWipesTranscriptWithoutChangingState(t *testing.T) {
	m := newTestModel()
	m = update(m, commandOutputMsg{text: "> /usage\ncumulative: 5 input, 2 output token(s)\n"})
	if !strings.Contains(m.completed, "cumulative") {
		t.Fatalf("completed = %q, want the earlier command output present before clearing", m.completed)
	}

	m = update(m, clearScrollbackMsg{})

	if m.completed != "" {
		t.Errorf("completed = %q after clearScrollbackMsg, want empty", m.completed)
	}
	if m.state != stateIdle {
		t.Errorf("state after clearScrollbackMsg = %v, want it to stay stateIdle", m.state)
	}
}
