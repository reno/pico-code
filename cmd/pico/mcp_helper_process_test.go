package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// TestMCPHelperProcess is not a real test. It is the re-exec entry point a
// test spawns (as os.Args[0] with -test.run=TestMCPHelperProcess) to act
// as a fake MCP server — the same subprocess-helper technique
// internal/mcp's and internal/tools' own suites use, duplicated here
// because Go's re-exec trick needs the helper compiled into the same test
// binary as the test that spawns it. During a normal `go test` run
// MCP_HELPER_PROCESS is unset, so this is a silent no-op test.
func TestMCPHelperProcess(_ *testing.T) {
	if os.Getenv("MCP_HELPER_PROCESS") != "1" {
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
			fmt.Printf(`{"jsonrpc":"2.0","id":%d,"result":{"tools":[{"name":"get_weather","description":"gets the weather","inputSchema":{"type":"object"}},{"name":"get_forecast","description":"gets the forecast","inputSchema":{"type":"object"}}]}}`+"\n", req.ID)
		}
	}
}
