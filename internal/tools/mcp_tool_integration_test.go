package tools

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/mcp"
)

// TestMCPToolServerKilledMidCallProducesValidPairedErrorResult is 13.2's
// AC end to end, with a real subprocess: a real mcp.Client connects to a
// real (fake) server, tools/list discovers "get_weather", it's wrapped
// and registered exactly as production wiring would, and the server dies
// the instant it receives tools/call. Registry.Run — the same call the
// agent loop makes for any tool — must return a plain (string, error)
// rather than hang or panic, so the loop's own already-tested machinery
// (invariant 4) can turn it into ToolResult{IsError:true} and continue.
// This is also invariant 5 in action: nothing about how the loop calls
// Registry.Run changes for a tool that happens to live behind a process
// boundary.
func TestMCPToolServerKilledMidCallProducesValidPairedErrorResult(t *testing.T) {
	cfg := mcp.ServerConfig{
		Name:    "weather-server",
		Command: os.Args[0],
		Args:    []string{"-test.run=TestMCPHelperProcess"},
		Env:     []string{"MCP_TOOL_HELPER_PROCESS=1"},
	}
	client, err := mcp.Start(context.Background(), cfg, 2*time.Second)
	if err != nil {
		t.Fatalf("mcp.Start() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	remoteTools, err := client.ListTools(context.Background())
	if err != nil {
		t.Fatalf("ListTools() error = %v", err)
	}
	if len(remoteTools) != 1 {
		t.Fatalf("len(remoteTools) = %d, want 1", len(remoteTools))
	}

	reg := NewRegistry()
	tool := NewMCPTool(client, cfg.Name, remoteTools[0])
	if err := reg.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	callCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	_, runErr := reg.Run(callCtx, tool.Name(), json.RawMessage(`{"location":"Paris"}`))
	elapsed := time.Since(start)

	if runErr == nil {
		t.Fatal("Registry.Run() error = nil, want an error when the server dies mid-call")
	}
	if elapsed > time.Second {
		t.Fatalf("Registry.Run() took %s after the server crashed, want a prompt return", elapsed)
	}

	// This is exactly what the agent loop does with any Tool.Run error
	// (loop.go, unmodified by this task) — spelled out here so the
	// resulting ToolResult's validity is asserted, not just implied.
	result := llm.ToolResult{ToolUseID: "call_1", Content: runErr.Error(), IsError: true}
	if result.Content == "" {
		t.Error("resulting ToolResult.Content is empty, want the error's message")
	}
}
