package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/llm"
)

func send(ch chan llm.Event, events ...llm.Event) {
	for _, e := range events {
		ch <- e
	}
	close(ch)
}

// TestPlainRendererWritesTextAsItStreamsWithNoANSI is 7.1's AC: piped
// output is clean and ANSI-free.
func TestPlainRendererWritesTextAsItStreamsWithNoANSI(t *testing.T) {
	ch := make(chan llm.Event)
	go send(ch,
		llm.TextDelta{Text: "hel"},
		llm.TextDelta{Text: "lo"},
		llm.MessageDone{StopReason: "end_turn", Usage: llm.Usage{InputTokens: 3, OutputTokens: 2}},
	)

	var buf bytes.Buffer
	resp, err := (&PlainRenderer{Out: &buf}).Render(context.Background(), ch)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}

	if got := buf.String(); got != "hello\n" {
		t.Errorf("Out = %q, want %q", got, "hello\n")
	}
	if strings.ContainsRune(buf.String(), '\x1b') {
		t.Errorf("Out contains an ANSI escape byte: %q", buf.String())
	}

	want := &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: "hello"}}},
		StopReason: "end_turn",
		Usage:      llm.Usage{InputTokens: 3, OutputTokens: 2},
	}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("Response mismatch (-want +got):\n%s", diff)
	}
}

// TestPlainRendererStaysSilentOnThinkingDelta is 16.3's AC: a scripted
// ThinkingDelta produces zero output — no thinking text ever reaches piped
// stdout — while still folding into the reconstructed Message ahead of
// Text, the same shape the non-streaming path would produce.
func TestPlainRendererStaysSilentOnThinkingDelta(t *testing.T) {
	ch := make(chan llm.Event)
	go send(ch,
		llm.ThinkingDelta{Text: "7 times 8 is 56."},
		llm.TextDelta{Text: "56"},
		llm.MessageDone{StopReason: "end_turn"},
	)

	var buf bytes.Buffer
	resp, err := (&PlainRenderer{Out: &buf}).Render(context.Background(), ch)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}

	if got := buf.String(); got != "56\n" {
		t.Errorf("Out = %q, want %q (no thinking text at all)", got, "56\n")
	}

	want := []llm.Block{llm.Thinking{Text: "7 times 8 is 56."}, llm.Text{Text: "56"}}
	if diff := cmp.Diff(want, resp.Message.Blocks); diff != "" {
		t.Errorf("Blocks mismatch (-want +got):\n%s", diff)
	}
}

func TestPlainRendererReconstructsToolUseBlock(t *testing.T) {
	ch := make(chan llm.Event)
	go send(ch,
		llm.TextDelta{Text: "checking"},
		llm.ToolUseStart{ID: "t1", Name: "get_weather"},
		llm.ToolUseArgsDelta{ID: "t1", Partial: `{"location"`},
		llm.ToolUseArgsDelta{ID: "t1", Partial: `:"Paris"}`},
		llm.ToolUseDone{ID: "t1", Input: json.RawMessage(`{"location":"Paris"}`)},
		llm.MessageDone{StopReason: "tool_use"},
	)

	var buf bytes.Buffer
	resp, err := (&PlainRenderer{Out: &buf}).Render(context.Background(), ch)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}

	want := []llm.Block{
		llm.Text{Text: "checking"},
		llm.ToolUse{ID: "t1", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
	}
	if diff := cmp.Diff(want, resp.Message.Blocks); diff != "" {
		t.Errorf("Blocks mismatch (-want +got):\n%s", diff)
	}
	if got := buf.String(); got != "checking\n" {
		t.Errorf("Out = %q, want %q", got, "checking\n")
	}
}

func TestPlainRendererPropagatesErrorEvent(t *testing.T) {
	wantErr := errors.New("boom")
	ch := make(chan llm.Event)
	go send(ch, llm.Error{Err: wantErr})

	_, err := (&PlainRenderer{Out: &bytes.Buffer{}}).Render(context.Background(), ch)
	if !errors.Is(err, wantErr) {
		t.Fatalf("Render() error = %v, want it to wrap %v", err, wantErr)
	}
}

func TestPlainRendererErrorsIfChannelClosesWithoutMessageDone(t *testing.T) {
	ch := make(chan llm.Event)
	go send(ch, llm.TextDelta{Text: "hi"})

	_, err := (&PlainRenderer{Out: &bytes.Buffer{}}).Render(context.Background(), ch)
	if err == nil {
		t.Fatal("Render() error = nil, want an error for a stream closed without MessageDone")
	}
}

func TestPlainRendererRespectsCancellation(t *testing.T) {
	ch := make(chan llm.Event) // never sends
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(10*time.Millisecond, cancel)

	start := time.Now()
	_, err := (&PlainRenderer{Out: &bytes.Buffer{}}).Render(ctx, ch)
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Render() error = %v, want it to wrap context.Canceled", err)
	}
	if elapsed > time.Second {
		t.Fatalf("Render() took %s after cancellation, want a prompt return", elapsed)
	}
}

// TestPlainRendererWroteAnyTracksEachRenderCallIndependently is a
// regression test for cmd/pico's streaming fallback print: a caller needs
// to know, after each Render call, whether that specific round actually
// streamed any text — a guard-trip or empty-reply explanation (16.1's
// --think can trigger the latter) is synthesized by the agent loop after
// Render already returned, so it never flows through TextDelta at all.
// WroteAny must reset per call, not latch true forever after one round
// that did stream text.
func TestPlainRendererWroteAnyTracksEachRenderCallIndependently(t *testing.T) {
	var buf bytes.Buffer
	r := &PlainRenderer{Out: &buf}

	ch1 := make(chan llm.Event)
	go send(ch1, llm.TextDelta{Text: "hi"}, llm.MessageDone{StopReason: "end_turn"})
	if _, err := r.Render(context.Background(), ch1); err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	if !r.WroteAny() {
		t.Error("WroteAny() = false after a round with a TextDelta, want true")
	}

	ch2 := make(chan llm.Event)
	go send(ch2, llm.MessageDone{StopReason: "length"})
	if _, err := r.Render(context.Background(), ch2); err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	if r.WroteAny() {
		t.Error("WroteAny() = true after a round with no TextDelta at all, want false")
	}
}

func TestPlainRendererToolStatusHasNoANSI(t *testing.T) {
	var buf bytes.Buffer
	r := PlainRenderer{Out: &buf}
	r.ToolStarted("t1", "read_file", json.RawMessage(`{"path":"x"}`))
	r.ToolFinished("t1", "read_file", "contents", false)
	r.ToolFinished("t1", "read_file", "boom", true)

	if strings.ContainsRune(buf.String(), '\x1b') {
		t.Errorf("tool status output contains an ANSI escape byte: %q", buf.String())
	}
}

// TestPlainRendererToolFinishedShowsErrorMessage guards against the tool
// failure content (e.g. a schema validation message naming a missing
// required parameter) being computed for the model's ToolResult but never
// reaching the human watching a piped or non-TUI session.
func TestPlainRendererToolFinishedShowsErrorMessage(t *testing.T) {
	var buf bytes.Buffer
	r := PlainRenderer{Out: &buf}
	r.ToolFinished("t1", "read_file", `tools: invalid input for "read_file": input: missing required field "path"`, true)

	if !strings.Contains(buf.String(), `missing required field "path"`) {
		t.Errorf("ToolFinished(isError=true) output = %q, want it to contain the failure message", buf.String())
	}
}
