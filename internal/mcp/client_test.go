package mcp

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

// helperConfig builds a ServerConfig whose "command" is this test binary
// itself, re-exec'd with -test.run=TestHelperProcess so it runs as a fake
// MCP server instead of the real test suite (see helper_process_test.go).
func helperConfig(t *testing.T, name, mode string) ServerConfig {
	t.Helper()
	return ServerConfig{
		Name:    name,
		Command: os.Args[0],
		Args:    []string{"-test.run=TestHelperProcess"},
		Env:     []string{"MCP_HELPER_PROCESS=1", "MCP_HELPER_MODE=" + mode},
	}
}

// TestStartAndListTools is 13.1's discovery AC: a fake server is
// discovered (a completed initialize handshake) and its tools listed.
func TestStartAndListTools(t *testing.T) {
	cfg := helperConfig(t, "fake", "ok")
	client, err := Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	tools, err := client.ListTools(context.Background())
	if err != nil {
		t.Fatalf("ListTools() error = %v", err)
	}
	want := []ToolInfo{{
		Name:        "echo",
		Description: "echoes input",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
	}}
	if diff := cmp.Diff(want, tools); diff != "" {
		t.Errorf("ListTools() mismatch (-want +got):\n%s", diff)
	}
}

// TestStartTimesOutOnHangingServer is 13.1's abandonment AC: a server that
// never answers initialize is abandoned once timeout elapses, promptly
// and with an error, rather than hanging Start forever.
func TestStartTimesOutOnHangingServer(t *testing.T) {
	cfg := helperConfig(t, "hangs", "hang")

	start := time.Now()
	_, err := Start(context.Background(), cfg, 200*time.Millisecond)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Start() error = nil, want an error for a server that never answers")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("Start() took %s after its handshake timeout, want a prompt return", elapsed)
	}
}

// TestStartMalformedResponseErrors proves a response that isn't valid
// JSON-RPC surfaces as an error rather than a panic or a silent hang.
func TestStartMalformedResponseErrors(t *testing.T) {
	cfg := helperConfig(t, "malformed", "bad_json")
	_, err := Start(context.Background(), cfg, 2*time.Second)
	if err == nil {
		t.Fatal("Start() error = nil, want an error for a malformed initialize response")
	}
}

// TestStartCtxCancelledStopsPromptly is CLAUDE.md invariant 6: cancelling
// ctx must abort an in-flight operation, here the handshake itself.
func TestStartCtxCancelledStopsPromptly(t *testing.T) {
	cfg := helperConfig(t, "hangs", "hang")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	_, err := Start(ctx, cfg, 5*time.Second)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Start() error = nil, want an error for an already-cancelled ctx")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("Start() took %s after ctx cancellation, want a prompt return", elapsed)
	}
}

// TestCloseTerminatesProcess proves Close actually reaps the child rather
// than leaving it running or leaking a zombie: Close returning at all
// (rather than hanging on Wait) is the signal the process exited.
func TestCloseTerminatesProcess(t *testing.T) {
	cfg := helperConfig(t, "fake", "ok")
	client, err := Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- client.Close() }()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Close() did not return within 2s, want the process reaped promptly")
	}
}
