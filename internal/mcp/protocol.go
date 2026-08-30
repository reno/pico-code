package mcp

import (
	"encoding/json"
	"fmt"
)

// protocolVersion is the MCP protocol version this client negotiates
// during initialize. There is exactly one client in this codebase, so a
// single hardcoded version (rather than a negotiated range) keeps the
// handshake simple; bump it by hand if a server ever rejects it.
const protocolVersion = "2024-11-05"

// clientName and clientVersion identify pico code to a server in
// initialize's clientInfo. Kept local to this package rather than
// imported from cmd/pico: this is wire metadata for the MCP handshake,
// not application versioning, and internal/mcp must not depend on cmd/.
const (
	clientName    = "pico-code"
	clientVersion = "0.1.0"
)

// rpcRequest is a JSON-RPC 2.0 request: one line of newline-delimited
// JSON on the server's stdin, per MCP's stdio transport.
type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcNotification is a JSON-RPC 2.0 notification: no ID, and the server
// must never reply to it.
type rpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcResponse is a JSON-RPC 2.0 response, decoded generically since its
// Result shape depends on which method the matching request named.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// rpcError is a JSON-RPC 2.0 error object.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("mcp: server error %d: %s", e.Code, e.Message)
}

// initializeParams is initialize's request payload. Capabilities is sent
// as an empty object: this client doesn't yet declare support for any
// optional MCP capability (sampling, roots, ...), only the baseline
// tools/list and tools/call 13.1/13.2 need.
type initializeParams struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ClientInfo      clientInfo      `json:"clientInfo"`
}

type clientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// initializeResult is initialize's response payload. Only ProtocolVersion
// is actually consulted (to fail fast on a mismatch); Capabilities and
// ServerInfo are kept as raw JSON since nothing in 13.1 needs their
// contents yet.
type initializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ServerInfo      json.RawMessage `json:"serverInfo"`
}

// ToolInfo is one tool a server advertised via tools/list.
type ToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

type listToolsResult struct {
	Tools []ToolInfo `json:"tools"`
}

// callToolParams is tools/call's request payload.
type callToolParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// CallToolResult is tools/call's response payload. IsError distinguishes a
// tool-execution failure the server still reports through a normal
// JSON-RPC result (per the MCP spec, execution errors use this flag, not
// a JSON-RPC error object — that's reserved for protocol-level failures
// like an unknown tool name) from success.
type CallToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError"`
}

// ContentBlock is one entry in a CallToolResult's Content array. Only Text
// is populated for Type == "text"; other MCP content types (image,
// resource, ...) carry no payload here, so a caller that flattens Content
// to a string can still name what it dropped instead of silently losing
// it.
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}
