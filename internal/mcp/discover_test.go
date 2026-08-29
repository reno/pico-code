package mcp

import (
	"context"
	"testing"
	"time"
)

// TestDiscoverGoodServer proves the straightforward path: a server that
// handshakes and lists tools cleanly comes back with no Err.
func TestDiscoverGoodServer(t *testing.T) {
	cfgs := []ServerConfig{helperConfig(t, "good", "ok")}
	results := Discover(context.Background(), cfgs, 2*time.Second)
	t.Cleanup(func() {
		for _, d := range results {
			if d.Client != nil {
				_ = d.Client.Close()
			}
		}
	})

	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Err != nil {
		t.Fatalf("results[0].Err = %v, want nil", results[0].Err)
	}
	if len(results[0].Tools) != 1 || results[0].Tools[0].Name != "echo" {
		t.Errorf("results[0].Tools = %v, want one tool named echo", results[0].Tools)
	}
}

// TestDiscoverAbandonsHangingServerWithoutBlockingOthers is 13.1's AC in
// full: a server that never answers is abandoned after its own timeout,
// and — because every server in cfgs is started concurrently, each with
// its own independent deadline — a hanging server never delays a good
// one's result or Discover's overall return.
func TestDiscoverAbandonsHangingServerWithoutBlockingOthers(t *testing.T) {
	cfgs := []ServerConfig{
		helperConfig(t, "good", "ok"),
		helperConfig(t, "hangs-on-handshake", "hang"),
		helperConfig(t, "hangs-on-tools-list", "no_tools"),
	}

	start := time.Now()
	results := Discover(context.Background(), cfgs, 200*time.Millisecond)
	elapsed := time.Since(start)
	t.Cleanup(func() {
		for _, d := range results {
			if d.Client != nil {
				_ = d.Client.Close()
			}
		}
	})

	if elapsed > 2*time.Second {
		t.Fatalf("Discover() took %s, want it bounded by the per-server timeout regardless of how many servers hang", elapsed)
	}
	if len(results) != 3 {
		t.Fatalf("len(results) = %d, want 3", len(results))
	}

	good, handshakeHang, toolsHang := results[0], results[1], results[2]
	if good.Err != nil {
		t.Errorf("good server: Err = %v, want nil", good.Err)
	}
	if handshakeHang.Err == nil {
		t.Error("server that never handshakes: Err = nil, want an error")
	}
	if handshakeHang.Client != nil {
		t.Error("server that never handshakes: Client != nil, want it abandoned with no live Client")
	}
	if toolsHang.Err == nil {
		t.Error("server that hangs on tools/list: Err = nil, want an error")
	}
	if toolsHang.Client != nil {
		t.Error("server that hangs on tools/list: Client != nil, want it closed and abandoned")
	}
}

// TestDiscoverEmpty proves Discover on no configured servers returns
// immediately with an empty result rather than blocking.
func TestDiscoverEmpty(t *testing.T) {
	results := Discover(context.Background(), nil, time.Second)
	if len(results) != 0 {
		t.Errorf("len(results) = %d, want 0", len(results))
	}
}
