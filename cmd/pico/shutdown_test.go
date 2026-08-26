package main

import (
	"context"
	"os"
	"testing"
	"time"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestShutdownContextFirstSignalCancels(t *testing.T) {
	sigCh := make(chan os.Signal, 2)
	ctx, stop := newShutdownContext(context.Background(), sigCh, func(int) {
		t.Fatal("exit should not be called on a single signal")
	})
	defer stop()

	sigCh <- os.Interrupt

	select {
	case <-ctx.Done():
	case <-time.After(200 * time.Millisecond):
		t.Fatal("context was not canceled within 200ms of the first signal")
	}
}

func TestShutdownContextSecondSignalForceExits(t *testing.T) {
	sigCh := make(chan os.Signal, 2)
	exited := make(chan int, 1)
	ctx, stop := newShutdownContext(context.Background(), sigCh, func(code int) {
		exited <- code
	})
	defer stop()

	sigCh <- os.Interrupt
	<-ctx.Done()
	sigCh <- os.Interrupt

	select {
	case code := <-exited:
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("exit was not called within 200ms of the second signal")
	}
}

func TestShutdownContextStopWithoutSignalLeaksNothing(t *testing.T) {
	sigCh := make(chan os.Signal, 2)
	_, stop := newShutdownContext(context.Background(), sigCh, func(int) {
		t.Fatal("exit should not be called")
	})
	stop()
}

// fakeSleepTool stands in for a real tool (built in phase 5) that blocks on
// I/O; it proves ctx cancellation reaches a running operation the way
// invariant 6 requires, before any real tool exists to test it against.
func fakeSleepTool(ctx context.Context) time.Duration {
	start := time.Now()
	select {
	case <-time.After(5 * time.Second):
	case <-ctx.Done():
	}
	return time.Since(start)
}

func TestShutdownDuringFakeToolReturnsQuickly(t *testing.T) {
	sigCh := make(chan os.Signal, 2)
	ctx, stop := newShutdownContext(context.Background(), sigCh, func(int) {
		t.Fatal("exit should not be called on a single signal")
	})
	defer stop()

	done := make(chan time.Duration, 1)
	go func() { done <- fakeSleepTool(ctx) }()

	time.Sleep(10 * time.Millisecond)
	sigCh <- os.Interrupt

	select {
	case elapsed := <-done:
		if elapsed > 200*time.Millisecond {
			t.Fatalf("fake tool returned after %s, want <= 200ms", elapsed)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("fake tool did not return within 200ms of Ctrl+C")
	}
}
