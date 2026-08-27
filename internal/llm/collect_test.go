package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestCollectStreamReconstructsBlockBoundaries(t *testing.T) {
	ch := StreamEvents(context.Background(), func(send func(Event) bool) {
		send(TextDelta{Text: "Let me check. "})
		send(ToolUseStart{ID: "call_1", Name: "get_weather"})
		send(ToolUseArgsDelta{ID: "call_1", Partial: `{"loc`})
		send(ToolUseArgsDelta{ID: "call_1", Partial: `ation":"Paris"}`})
		send(ToolUseDone{ID: "call_1", Input: json.RawMessage(`{"location":"Paris"}`)})
		send(TextDelta{Text: "One moment."})
		send(MessageDone{StopReason: "tool_use", Usage: Usage{InputTokens: 10, OutputTokens: 5}})
	})

	got, err := CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	want := &Response{
		Message: Message{
			Role: RoleAssistant,
			Blocks: []Block{
				Text{Text: "Let me check. "},
				ToolUse{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
				Text{Text: "One moment."},
			},
		},
		StopReason: "tool_use",
		Usage:      Usage{InputTokens: 10, OutputTokens: 5},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("CollectStream() mismatch (-want +got):\n%s", diff)
	}
}

func TestCollectStreamPropagatesErrorEvent(t *testing.T) {
	boom := errors.New("boom")
	ch := StreamEvents(context.Background(), func(send func(Event) bool) {
		send(TextDelta{Text: "partial"})
		send(Error{Err: boom})
	})

	_, err := CollectStream(context.Background(), ch)
	if !errors.Is(err, boom) {
		t.Fatalf("CollectStream() error = %v, want wrapping %v", err, boom)
	}
}

func TestCollectStreamErrorsIfChannelClosesWithoutMessageDone(t *testing.T) {
	ch := StreamEvents(context.Background(), func(send func(Event) bool) {
		send(TextDelta{Text: "partial"})
	})

	_, err := CollectStream(context.Background(), ch)
	if err == nil {
		t.Fatal("CollectStream() error = nil, want an error for a stream that closed early")
	}
}

func TestCollectStreamRespectsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	block := make(chan struct{})
	ch := StreamEvents(context.Background(), func(_ func(Event) bool) {
		<-block
	})

	cancel()
	_, err := CollectStream(ctx, ch)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CollectStream() error = %v, want wrapping context.Canceled", err)
	}
	close(block)
}
