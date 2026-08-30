package llm

import (
	"context"
	"testing"
	"time"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// TestStreamEventsConsumedToCompletion is half of 6.1's AC: a fake stream
// consumed to completion closes its channel exactly once, with every event
// delivered in order.
func TestStreamEventsConsumedToCompletion(t *testing.T) {
	done := make(chan struct{})
	ch := StreamEvents(context.Background(), func(send func(Event) bool) {
		defer close(done)
		send(TextDelta{Text: "a"})
		send(TextDelta{Text: "b"})
		send(MessageDone{StopReason: "end_turn"})
	})

	var got []Event
	for e := range ch {
		got = append(got, e)
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("producer goroutine did not finish")
	}

	if len(got) != 3 {
		t.Fatalf("got %d events, want 3", len(got))
	}
	if _, ok := got[2].(MessageDone); !ok {
		t.Errorf("last event = %T, want MessageDone", got[2])
	}

	if _, ok := <-ch; ok {
		t.Error("channel yielded another value after range completed")
	}
}

// TestStreamEventsCancelledMidFlight is the other half of 6.1's AC: a
// producer that would otherwise send forever stops and the channel closes
// once ctx is cancelled, with no leaked goroutine (verified by TestMain's
// goleak.VerifyTestMain, and directly here via the finished handshake) and
// no send on a closed channel (the race detector would catch that).
func TestStreamEventsCancelledMidFlight(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	finished := make(chan struct{})

	ch := StreamEvents(ctx, func(send func(Event) bool) {
		defer close(finished)
		close(started)
		for {
			if !send(TextDelta{Text: "x"}) {
				return
			}
		}
	})

	<-started
	<-ch
	<-ch
	cancel()

	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("producer goroutine did not exit after ctx cancellation")
	}

	for {
		if _, ok := <-ch; !ok {
			break
		}
	}
}
