// Command pico is the pico code CLI agent.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	ctx, stop := newShutdownContext(context.Background(), sigCh, os.Exit)
	defer stop()

	if err := newRootCmd(os.Getenv).ExecuteContext(ctx); err != nil {
		os.Exit(1)
	}
}
