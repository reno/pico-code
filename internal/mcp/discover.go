package mcp

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

// Discovered is one server's outcome from Discover: either a live Client
// with the tools it advertised, or an Err explaining why it never became
// available.
type Discovered struct {
	Config ServerConfig
	Client *Client
	Tools  []ToolInfo
	Err    error
}

// Discover starts every server in cfgs concurrently — "lazily", in that
// nothing runs until Discover is actually called — each independently
// bounded by timeout across both the handshake and tools/list. A server
// that never answers in time is killed and reported via Err with a logged
// warning, never left to block the others or the caller: Discover returns
// once every server has settled, and no single server can hold that up
// past timeout.
func Discover(ctx context.Context, cfgs []ServerConfig, timeout time.Duration) []Discovered {
	results := make([]Discovered, len(cfgs))
	var wg sync.WaitGroup
	for i, cfg := range cfgs {
		wg.Add(1)
		go func(i int, cfg ServerConfig) {
			defer wg.Done()
			results[i] = discoverOne(ctx, cfg, timeout)
		}(i, cfg)
	}
	wg.Wait()
	return results
}

func discoverOne(ctx context.Context, cfg ServerConfig, timeout time.Duration) Discovered {
	client, err := Start(ctx, cfg, timeout)
	if err != nil {
		slog.WarnContext(ctx, "mcp: server abandoned", "server", cfg.Name, "error", err)
		return Discovered{Config: cfg, Err: err}
	}

	lctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	tools, err := client.ListTools(lctx)
	if err != nil {
		slog.WarnContext(ctx, "mcp: server abandoned", "server", cfg.Name, "error", err)
		_ = client.Close()
		return Discovered{Config: cfg, Err: err}
	}

	return Discovered{Config: cfg, Client: client, Tools: tools}
}
