package main

import (
	"context"
	"os"
	"sync"
)

// newShutdownContext returns a context canceled the first time a value
// arrives on sig, and calls exit(1) if a second arrives while the caller is
// still shutting down — one Ctrl+C asks the agent loop to stop, two insist.
// sig and exit are injected so tests can drive shutdown without touching
// process signals or actually exiting.
//
// The returned stop func must be called once the caller is done (normal
// completion or test cleanup); it is what lets the internal goroutine exit
// without a second signal ever arriving.
func newShutdownContext(parent context.Context, sig <-chan os.Signal, exit func(int)) (ctx context.Context, stop context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	unregister := make(chan struct{})
	var once sync.Once
	stopFn := func() {
		once.Do(func() { close(unregister) })
		cancel()
	}

	go func() {
		select {
		case <-sig:
		case <-unregister:
			return
		}
		cancel()
		select {
		case <-sig:
			exit(1)
		case <-unregister:
		}
	}()

	return ctx, stopFn
}
