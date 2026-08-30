package tools

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// TestMCPHelperProcess is not a real test. It is the re-exec entry point a
// test spawns (as os.Args[0] with -test.run=TestMCPHelperProcess) to act
// as a fake MCP server whose tools/call handler crashes without
// responding — reproducing "server killed mid-call" for real, at the
// process level, rather than through a fake mcpCaller. Mirrors
// internal/mcp's own TestHelperProcess; duplicated rather than shared
// because Go's re-exec trick needs the helper compiled into the same test
// binary as the test that spawns it.
func TestMCPHelperProcess(_ *testing.T) {
	if os.Getenv("MCP_TOOL_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

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
			fmt.Printf(`{"jsonrpc":"2.0","id":%d,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1.0"}}}`+"\n", req.ID)
		case "tools/list":
			fmt.Printf(`{"jsonrpc":"2.0","id":%d,"result":{"tools":[{"name":"get_weather","description":"gets weather","inputSchema":{"type":"object"}}]}}`+"\n", req.ID)
		case "tools/call":
			// Dies without responding — a real process killed mid-call.
			os.Exit(1)
		}
	}
}
