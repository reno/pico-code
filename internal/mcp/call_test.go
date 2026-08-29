package mcp

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestCallReturnsToolResult(t *testing.T) {
	cfg := helperConfig(t, "fake", "ok")
	client, err := Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	result, err := client.Call(context.Background(), "echo", json.RawMessage(`{"text":"hi"}`))
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	want := &CallToolResult{Content: []ContentBlock{{Type: "text", Text: "hi"}}, IsError: false}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Call() mismatch (-want +got):\n%s", diff)
	}
}

func TestCallServerReportedError(t *testing.T) {
	cfg := helperConfig(t, "fake", "call_errors")
	client, err := Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	result, err := client.Call(context.Background(), "echo", json.RawMessage(`{"text":"hi"}`))
	if err != nil {
		t.Fatalf("Call() error = %v, want a normal JSON-RPC result with IsError set, per the MCP spec's convention for a tool execution failure", err)
	}
	if !result.IsError {
		t.Error("result.IsError = false, want true")
	}
}

// TestCallServerKilledMidCallErrors is 13.2's AC: a server killed mid-call
// produces an error rather than hanging Call forever or panicking. The
// helper process exits without responding the instant it receives
// tools/call, which is what a real crash looks like on the wire — the
// pending read hits EOF.
func TestCallServerKilledMidCallErrors(t *testing.T) {
	cfg := helperConfig(t, "fake", "crash_on_call")
	client, err := Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	callCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	_, err = client.Call(callCtx, "echo", json.RawMessage(`{"text":"hi"}`))
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Call() error = nil, want an error for a server that died mid-call")
	}
	if elapsed > time.Second {
		t.Fatalf("Call() took %s after the server crashed, want the closed pipe to unblock it promptly", elapsed)
	}
}
