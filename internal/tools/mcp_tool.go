package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/reno/pico-code/internal/mcp"
)

// mcpCaller is the one capability MCPTool needs from an MCP connection.
// Declared here, by the consumer, rather than reused from mcp.Client's own
// method set directly (CLAUDE.md's "interfaces are declared by the
// consumer, kept to what the consumer uses") — this is also what lets a
// test fake a server's behavior without spawning a real subprocess.
type mcpCaller interface {
	Call(ctx context.Context, tool string, arguments json.RawMessage) (*mcp.CallToolResult, error)
}

// MCPTool wraps one remote tool a server advertised via tools/list into
// the local Tool interface, so the Registry and the agent loop never have
// to know a tool's implementation crosses a process boundary — CLAUDE.md
// invariant 5, tools own side effects, the loop stays generic.
type MCPTool struct {
	client      mcpCaller
	server      string
	remoteName  string
	description string
	schema      json.RawMessage
}

// NewMCPTool wraps info, one entry from server's tools/list, using client
// for tools/call.
func NewMCPTool(client mcpCaller, server string, info mcp.ToolInfo) *MCPTool {
	return &MCPTool{
		client:      client,
		server:      server,
		remoteName:  info.Name,
		description: info.Description,
		schema:      info.InputSchema,
	}
}

// Name implements Tool. Namespacing as mcp__<server>__<tool> means tools
// from two different servers — or an MCP tool and a same-named built-in
// one — can never collide in the Registry, which detects the collision
// that would otherwise happen (CLAUDE.md's "duplicate name is always a
// wiring mistake") on whatever the final, namespaced name turns out to
// be.
func (t *MCPTool) Name() string { return MCPToolName(t.server, t.remoteName) }

// MCPToolName returns the namespaced Registry name an MCPTool for
// server/remoteName uses. Exported so a caller managing a server's own
// lifecycle (reconnecting it, tearing it down) can compute the same name
// to unregister a stale entry, without duplicating the "mcp__server__tool"
// format and risking it drifting from Name's.
func MCPToolName(server, remoteName string) string {
	return fmt.Sprintf("mcp__%s__%s", server, remoteName)
}

// Description implements Tool.
func (t *MCPTool) Description() string { return t.description }

// NeedsApproval implements tools.ApprovalRequired: a remote server's tool
// can have side effects at least as unbounded as run_command or write_file
// — network calls, shell execution, filesystem access on its own machine —
// but none of the sandboxing this process applies to its own built-ins, so
// every call goes through the same approval gate.
func (t *MCPTool) NeedsApproval() bool { return true }

// Schema implements Tool. The server's JSON Schema is passed through
// untouched: unlike an LLM provider adapter narrowing a schema to its own
// SDK shape, this client has no basis for reshaping an externally
// authored schema it doesn't control and isn't guaranteed to fully
// understand.
func (t *MCPTool) Schema() json.RawMessage { return t.schema }

// Run calls the remote tool via tools/call and flattens its result to a
// string through the same truncation budget every built-in tool uses
// (truncateBytes, §5.2). A remote failure — the call erroring, the
// connection dying mid-call, or the server itself reporting isError —
// always becomes a returned Go error, never a panic: CLAUDE.md invariant
// 4, a failing tool is data for the model, not control flow, and the
// agent loop already turns any Tool.Run error into
// ToolResult{IsError:true} without needing to know this one crossed a
// process boundary.
func (t *MCPTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	result, err := t.client.Call(ctx, t.remoteName, input)
	if err != nil {
		return "", fmt.Errorf("mcp tool %q: %w", t.Name(), err)
	}

	text := truncateBytes(flattenContent(result.Content), defaultByteBudget)
	if result.IsError {
		return "", fmt.Errorf("mcp tool %q: %s", t.Name(), text)
	}
	return text, nil
}

// flattenContent joins a CallToolResult's content blocks into one string:
// a text block contributes its text verbatim, and any other MCP content
// type (image, resource, ...) contributes a placeholder naming its type,
// so a block this client doesn't render is never silently dropped.
func flattenContent(blocks []mcp.ContentBlock) string {
	parts := make([]string, len(blocks))
	for i, b := range blocks {
		if b.Type == "text" {
			parts[i] = b.Text
		} else {
			parts[i] = fmt.Sprintf("[%s content omitted]", b.Type)
		}
	}
	return strings.Join(parts, "\n")
}

var (
	_ Tool             = (*MCPTool)(nil)
	_ ApprovalRequired = (*MCPTool)(nil)
)
