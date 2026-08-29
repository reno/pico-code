package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/reno/pico-code/internal/mcp"
)

// fakeCaller scripts one Call() outcome per test, without spawning a real
// MCP server subprocess — the unit-test half of 13.2's coverage, leaving
// the real-process "server killed mid-call" scenario to
// mcp_tool_helper_process_test.go's integration test.
type fakeCaller struct {
	gotTool string
	gotArgs json.RawMessage

	result *mcp.CallToolResult
	err    error
}

func (f *fakeCaller) Call(_ context.Context, tool string, arguments json.RawMessage) (*mcp.CallToolResult, error) {
	f.gotTool = tool
	f.gotArgs = arguments
	return f.result, f.err
}

func testToolInfo() mcp.ToolInfo {
	return mcp.ToolInfo{
		Name:        "get_weather",
		Description: "Get the current weather for a location.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}
}

func TestMCPToolNameIsNamespaced(t *testing.T) {
	tool := NewMCPTool(&fakeCaller{}, "weather-server", testToolInfo())
	if got, want := tool.Name(), "mcp__weather-server__get_weather"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

func TestMCPToolDescriptionAndSchemaPassThrough(t *testing.T) {
	info := testToolInfo()
	tool := NewMCPTool(&fakeCaller{}, "weather-server", info)
	if tool.Description() != info.Description {
		t.Errorf("Description() = %q, want %q", tool.Description(), info.Description)
	}
	if string(tool.Schema()) != string(info.InputSchema) {
		t.Errorf("Schema() = %s, want the server's schema byte-identical: %s", tool.Schema(), info.InputSchema)
	}
}

func TestMCPToolRunFlattensTextContentAndForwardsArguments(t *testing.T) {
	caller := &fakeCaller{result: &mcp.CallToolResult{
		Content: []mcp.ContentBlock{{Type: "text", Text: "15C, cloudy"}},
	}}
	tool := NewMCPTool(caller, "weather-server", testToolInfo())

	out, err := tool.Run(context.Background(), json.RawMessage(`{"location":"Paris"}`))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if out != "15C, cloudy" {
		t.Errorf("Run() = %q, want %q", out, "15C, cloudy")
	}
	if caller.gotTool != "get_weather" {
		t.Errorf("Call() was invoked with tool = %q, want the unnamespaced remote name %q", caller.gotTool, "get_weather")
	}
	if string(caller.gotArgs) != `{"location":"Paris"}` {
		t.Errorf("Call() was invoked with arguments = %s, want the input passed through unchanged", caller.gotArgs)
	}
}

func TestMCPToolRunFlattensMultipleAndNonTextBlocks(t *testing.T) {
	caller := &fakeCaller{result: &mcp.CallToolResult{
		Content: []mcp.ContentBlock{
			{Type: "text", Text: "line one"},
			{Type: "image"},
			{Type: "text", Text: "line two"},
		},
	}}
	tool := NewMCPTool(caller, "srv", testToolInfo())

	out, err := tool.Run(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	want := "line one\n[image content omitted]\nline two"
	if out != want {
		t.Errorf("Run() = %q, want %q", out, want)
	}
}

func TestMCPToolRunTruncatesLargeOutput(t *testing.T) {
	huge := strings.Repeat("x", defaultByteBudget*2)
	caller := &fakeCaller{result: &mcp.CallToolResult{Content: []mcp.ContentBlock{{Type: "text", Text: huge}}}}
	tool := NewMCPTool(caller, "srv", testToolInfo())

	out, err := tool.Run(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(out) > defaultByteBudget {
		t.Errorf("len(Run()) = %d, want at most the %d-byte budget", len(out), defaultByteBudget)
	}
	if !strings.Contains(out, "elided") {
		t.Errorf("Run() = %q, want the elision marker for output over budget", out)
	}
}

// TestMCPToolRunServerReportedErrorBecomesGoError is 13.2's AC: a remote
// failure is data, not a crash — here the MCP-spec convention of the
// server answering tools/call normally but with isError:true.
func TestMCPToolRunServerReportedErrorBecomesGoError(t *testing.T) {
	caller := &fakeCaller{result: &mcp.CallToolResult{
		Content: []mcp.ContentBlock{{Type: "text", Text: "location not found"}},
		IsError: true,
	}}
	tool := NewMCPTool(caller, "srv", testToolInfo())

	_, err := tool.Run(context.Background(), json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("Run() error = nil, want an error when the server reports isError")
	}
	if !strings.Contains(err.Error(), "location not found") {
		t.Errorf("Run() error = %v, want it to carry the server's error content", err)
	}
}

// TestMCPToolRunTransportFailureBecomesGoError proves a transport-level
// Call failure (the connection dying, a timeout, ...) also surfaces as an
// ordinary error rather than a panic — this is the scenario a killed
// server produces, exercised for real (with a real subprocess) in
// mcp_tool_helper_process_test.go.
func TestMCPToolRunTransportFailureBecomesGoError(t *testing.T) {
	caller := &fakeCaller{err: errors.New("mcp: connection lost")}
	tool := NewMCPTool(caller, "srv", testToolInfo())

	_, err := tool.Run(context.Background(), json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("Run() error = nil, want an error when the transport call fails")
	}
}

// TestMCPToolDuplicateNamespacedNameRejected is 13.2's AC: duplicate
// detection covers namespaced names.
func TestMCPToolDuplicateNamespacedNameRejected(t *testing.T) {
	reg := NewRegistry()
	first := NewMCPTool(&fakeCaller{}, "weather-server", testToolInfo())
	second := NewMCPTool(&fakeCaller{}, "weather-server", testToolInfo())

	if err := reg.Register(first); err != nil {
		t.Fatalf("Register(first) error = %v", err)
	}
	err := reg.Register(second)
	if !errors.Is(err, ErrDuplicateTool) {
		t.Fatalf("Register(second) error = %v, want wrapping ErrDuplicateTool", err)
	}
}

// TestMCPToolSameRemoteNameDifferentServersDoNotCollide proves the
// namespace actually does its job: the same tool name advertised by two
// different servers produces two distinct, both-registrable names.
func TestMCPToolSameRemoteNameDifferentServersDoNotCollide(t *testing.T) {
	reg := NewRegistry()
	a := NewMCPTool(&fakeCaller{}, "server-a", testToolInfo())
	b := NewMCPTool(&fakeCaller{}, "server-b", testToolInfo())

	if err := reg.Register(a); err != nil {
		t.Fatalf("Register(a) error = %v", err)
	}
	if err := reg.Register(b); err != nil {
		t.Fatalf("Register(b) error = %v, want no collision with a differently-namespaced tool", err)
	}
}
