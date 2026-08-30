// Package llm defines the canonical, provider-agnostic message and request
// types every adapter translates to and from. The shape is Anthropic's: a
// message is a role plus a list of content blocks. This package must never
// import a provider SDK — adapters live in their own subpackages and depend
// on this one, never the other way around.
package llm

import (
	"encoding/json"
	"fmt"
)

// Role identifies who produced a Message.
type Role string

// The two roles a canonical Message can carry. The system prompt is not a
// message role — it lives on Request.System, matching how every provider
// actually sends it on the wire.
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Block is one piece of content within a Message. It is sealed to this
// package (via the unexported isBlock method) so every adapter can exhaust a
// type switch over exactly Text, ToolUse, ToolResult, and Thinking and know
// no other case will ever appear.
type Block interface {
	isBlock()
}

// Text is a plain-text content block.
type Text struct {
	Text string
}

func (Text) isBlock() {}

// Thinking is a model's reasoning trace, produced ahead of its Text reply
// when Request.Think asked for one (16.1). Not every provider supports it —
// an adapter that doesn't never produces this block, rather than producing
// an empty one.
type Thinking struct {
	Text string
}

func (Thinking) isBlock() {}

// ToolUse is a model-issued request to call a tool. Input is kept as raw
// JSON because the loop validates and decodes it against the tool's schema
// (phase 1.4) rather than this package guessing a shape.
type ToolUse struct {
	ID    string
	Name  string
	Input json.RawMessage
}

func (ToolUse) isBlock() {}

// ToolResult answers a ToolUse by ID. Every ToolUse in an assistant message
// must be followed by exactly one ToolResult with a matching ToolUseID, per
// CLAUDE.md invariant 3; IsError distinguishes a tool failure (still a
// successful round trip to the model) from a successful result.
type ToolResult struct {
	ToolUseID string
	Content   string
	IsError   bool
}

func (ToolResult) isBlock() {}

// Message is a single turn: a role plus the ordered content blocks that make
// it up.
type Message struct {
	Role   Role
	Blocks []Block
}

// Usage reports token counts for a single request/response exchange.
// CacheWriteTokens and CacheReadTokens are 0 for a provider or request that
// never touched prompt caching (every non-Anthropic adapter, and any
// Anthropic request before 15.2's cache_control breakpoints existed) —
// there is no separate "unsupported" state to distinguish from "not used
// this time."
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheWriteTokens int
	CacheReadTokens  int
}

// ToolDefinition is the provider-agnostic shape of a tool advertised in a
// Request. It carries only what an adapter needs to build a provider's tool
// schema on the wire — internal/tools owns validation and execution and is
// not imported here to keep the dependency direction one-way.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// Request is everything a Provider needs to produce one Response.
type Request struct {
	System        string
	Messages      []Message
	Tools         []ToolDefinition
	MaxTokens     int
	Temperature   float64
	StopSequences []string

	// Think asks the provider to produce a Thinking block ahead of its
	// reply, when it supports one (16.1). A provider with no equivalent
	// ignores it — never an error, per CLAUDE.md invariant 2's "neutral
	// Request field" pattern rather than widening Provider itself.
	Think bool
}

// Response is a Provider's answer to a Request.
type Response struct {
	Message    Message
	StopReason string
	Usage      Usage
}

// wireBlock is the JSON envelope for a Block: a "type" discriminator plus
// the union of every block kind's fields. It exists only to give Message a
// stable, self-describing wire format that round-trips through save/load in
// internal/history.
type wireBlock struct {
	Type string `json:"type"`

	Text string `json:"text,omitempty"`

	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

func toWireBlock(b Block) (wireBlock, error) {
	switch v := b.(type) {
	case Text:
		return wireBlock{Type: "text", Text: v.Text}, nil
	case Thinking:
		return wireBlock{Type: "thinking", Text: v.Text}, nil
	case ToolUse:
		return wireBlock{Type: "tool_use", ID: v.ID, Name: v.Name, Input: v.Input}, nil
	case ToolResult:
		return wireBlock{Type: "tool_result", ToolUseID: v.ToolUseID, Content: v.Content, IsError: v.IsError}, nil
	default:
		return wireBlock{}, fmt.Errorf("llm: unknown block type %T", b)
	}
}

func (wb wireBlock) toBlock() (Block, error) {
	switch wb.Type {
	case "text":
		return Text{Text: wb.Text}, nil
	case "thinking":
		return Thinking{Text: wb.Text}, nil
	case "tool_use":
		return ToolUse{ID: wb.ID, Name: wb.Name, Input: wb.Input}, nil
	case "tool_result":
		return ToolResult{ToolUseID: wb.ToolUseID, Content: wb.Content, IsError: wb.IsError}, nil
	default:
		return nil, fmt.Errorf("llm: unknown block type %q", wb.Type)
	}
}

// MarshalJSON implements json.Marshaler so a Message round-trips through
// save/load without callers needing to know about wireBlock.
func (m Message) MarshalJSON() ([]byte, error) {
	wire := struct {
		Role   Role        `json:"role"`
		Blocks []wireBlock `json:"blocks"`
	}{Role: m.Role, Blocks: make([]wireBlock, len(m.Blocks))}

	for i, b := range m.Blocks {
		wb, err := toWireBlock(b)
		if err != nil {
			return nil, fmt.Errorf("marshal message: %w", err)
		}
		wire.Blocks[i] = wb
	}
	return json.Marshal(wire)
}

// UnmarshalJSON implements json.Unmarshaler, the inverse of MarshalJSON.
func (m *Message) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role   Role        `json:"role"`
		Blocks []wireBlock `json:"blocks"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("unmarshal message: %w", err)
	}

	blocks := make([]Block, len(wire.Blocks))
	for i, wb := range wire.Blocks {
		b, err := wb.toBlock()
		if err != nil {
			return fmt.Errorf("unmarshal message: %w", err)
		}
		blocks[i] = b
	}

	m.Role = wire.Role
	m.Blocks = blocks
	return nil
}
