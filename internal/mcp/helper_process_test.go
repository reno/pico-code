package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

// TestHelperProcess is not a real test. It is the re-exec entry point a
// test spawns (as os.Args[0] with -test.run=TestHelperProcess) to act as
// a fake MCP server speaking JSON-RPC over its own stdin/stdout — the same
// subprocess-helper technique Go's own os/exec tests use, so the suite
// never depends on an external binary or script. During a normal `go test`
// run MCP_HELPER_PROCESS is unset, so this is a silent no-op test.
// Behavior is selected by MCP_HELPER_MODE so one binary covers every
// server scenario the suite needs.
func TestHelperProcess(_ *testing.T) {
	if os.Getenv("MCP_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	switch os.Getenv("MCP_HELPER_MODE") {
	case "ok":
		runOKHelperServer()
	case "hang":
		time.Sleep(10 * time.Second)
	case "bad_json":
		fmt.Println("not json")
	case "no_tools":
		runNoToolsHelperServer()
	}
}

// runOKHelperServer answers initialize and tools/list correctly, and
// silently drops notifications/initialized (a notification, never
// answered).
func runOKHelperServer() {
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			return
		}
		var req struct {
			ID     int64  `json:"id"`
			Method string `json:"method"`
		}
		if err := json.Unmarshal(line, &req); err != nil {
			return
		}
		switch req.Method {
		case "initialize":
			fmt.Printf("{\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":{\"protocolVersion\":%q,\"capabilities\":{},\"serverInfo\":{\"name\":\"fake\",\"version\":\"1.0\"}}}\n", req.ID, protocolVersion)
		case "tools/list":
			fmt.Printf(`{"jsonrpc":"2.0","id":%d,"result":{"tools":[{"name":"echo","description":"echoes input","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}}`+"\n", req.ID)
		case "notifications/initialized":
			// No response — it's a notification.
		}
	}
}

// runNoToolsHelperServer answers initialize but then hangs on tools/list,
// so Discover's per-step timeout (not just the handshake step) is what
// gets exercised.
func runNoToolsHelperServer() {
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			return
		}
		var req struct {
			ID     int64  `json:"id"`
			Method string `json:"method"`
		}
		if err := json.Unmarshal(line, &req); err != nil {
			return
		}
		if req.Method == "initialize" {
			fmt.Printf("{\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":{\"protocolVersion\":%q,\"capabilities\":{},\"serverInfo\":{\"name\":\"fake\",\"version\":\"1.0\"}}}\n", req.ID, protocolVersion)
			continue
		}
		// tools/list (or anything else): never answer.
		time.Sleep(10 * time.Second)
	}
}
