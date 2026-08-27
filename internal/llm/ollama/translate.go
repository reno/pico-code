// Package ollama adapts Ollama's native /api/chat endpoint to the canonical
// llm.Provider interface. Native /api/chat is used rather than the
// OpenAI-compatible /v1 endpoint: num_ctx (a required knob — Ollama
// silently truncates context without it) is a native option with no /v1
// equivalent, and CLAUDE.md's provider gotchas already describe /api/chat's
// id/argument quirks as the ones to design around.
package ollama

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"

	"github.com/reno/pico-code/internal/llm"
)

// toRequest translates a canonical Request into an Ollama ChatRequest.
// num_ctx always has an explicit value (CLAUDE.md: Ollama silently
// truncates context otherwise) — it comes from config via the caller, never
// left unset here.
func toRequest(req llm.Request, model string, numCtx int) (*api.ChatRequest, error) {
	stream := false
	ar := &api.ChatRequest{
		Model:   model,
		Stream:  &stream,
		Options: map[string]any{"num_ctx": numCtx},
	}
	if req.Temperature != 0 {
		ar.Options["temperature"] = req.Temperature
	}

	var msgs []api.Message
	if req.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: req.System})
	}
	for i, m := range req.Messages {
		converted, err := toMessages(m)
		if err != nil {
			return nil, fmt.Errorf("message %d: %w", i, err)
		}
		msgs = append(msgs, converted...)
	}
	ar.Messages = msgs

	if len(req.Tools) > 0 {
		tools := make(api.Tools, len(req.Tools))
		for i, td := range req.Tools {
			params, err := toParameters(td.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("tool %q: %w", td.Name, err)
			}
			tools[i] = api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        td.Name,
					Description: td.Description,
					Parameters:  params,
				},
			}
		}
		ar.Tools = tools
	}

	return ar, nil
}

// toMessages expands one canonical Message into the Ollama messages it maps
// to. Most of the time that's one; a user message answering N parallel
// tool calls expands to N role:"tool" messages, since Ollama represents
// each tool result as its own message rather than Anthropic's array of
// content blocks within one message.
func toMessages(m llm.Message) ([]api.Message, error) {
	switch m.Role {
	case llm.RoleUser:
		return toUserMessages(m.Blocks)
	case llm.RoleAssistant:
		return toAssistantMessages(m.Blocks)
	default:
		return nil, fmt.Errorf("unsupported role %q", m.Role)
	}
}

func toUserMessages(blocks []llm.Block) ([]api.Message, error) {
	var text strings.Builder
	var toolMsgs []api.Message
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(v.Text)
		case llm.ToolResult:
			toolMsgs = append(toolMsgs, api.Message{Role: "tool", Content: v.Content, ToolCallID: v.ToolUseID})
		default:
			return nil, fmt.Errorf("unsupported block type %T in user message", b)
		}
	}
	if text.Len() == 0 {
		return toolMsgs, nil
	}
	return append([]api.Message{{Role: "user", Content: text.String()}}, toolMsgs...), nil
}

func toAssistantMessages(blocks []llm.Block) ([]api.Message, error) {
	msg := api.Message{Role: "assistant"}
	var text strings.Builder
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(v.Text)
		case llm.ToolUse:
			args, err := toToolCallArguments(v.Input)
			if err != nil {
				return nil, fmt.Errorf("tool_use %s: %w", v.ID, err)
			}
			msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
				ID:       v.ID,
				Function: api.ToolCallFunction{Name: v.Name, Arguments: args},
			})
		default:
			return nil, fmt.Errorf("unsupported block type %T in assistant message", b)
		}
	}
	msg.Content = text.String()
	return []api.Message{msg}, nil
}

func toToolCallArguments(input json.RawMessage) (api.ToolCallFunctionArguments, error) {
	if len(input) == 0 {
		return api.NewToolCallFunctionArguments(), nil
	}
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal(input, &args); err != nil {
		return api.ToolCallFunctionArguments{}, fmt.Errorf("parse tool_use input: %w", err)
	}
	return args, nil
}

// toParameters lifts a tool's generated JSON Schema (internal/tools'
// GenerateSchema output) into Ollama's ToolFunctionParameters shape.
func toParameters(raw json.RawMessage) (api.ToolFunctionParameters, error) {
	var parsed struct {
		Properties json.RawMessage `json:"properties"`
		Required   []string        `json:"required"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return api.ToolFunctionParameters{}, fmt.Errorf("parse input schema: %w", err)
	}

	props := api.NewToolPropertiesMap()
	if len(parsed.Properties) > 0 {
		if err := json.Unmarshal(parsed.Properties, props); err != nil {
			return api.ToolFunctionParameters{}, fmt.Errorf("parse input schema properties: %w", err)
		}
	}

	return api.ToolFunctionParameters{
		Type:       "object",
		Required:   parsed.Required,
		Properties: props,
	}, nil
}

// fromResponse translates a non-streaming ChatResponse into the canonical
// Response. req is only used to estimate token counts when resp's own are
// 0 (see estimateUsage) — translation of the message itself doesn't need it.
func fromResponse(req llm.Request, resp *api.ChatResponse) (*llm.Response, error) {
	blocks, err := fromMessage(resp.Message)
	if err != nil {
		return nil, err
	}
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: resp.DoneReason,
		Usage: estimateUsage(req, resp.Message.Content, llm.Usage{
			InputTokens:  resp.PromptEvalCount,
			OutputTokens: resp.EvalCount,
		}),
	}, nil
}

func fromMessage(m api.Message) ([]llm.Block, error) {
	var blocks []llm.Block
	if m.Content != "" {
		blocks = append(blocks, llm.Text{Text: m.Content})
	}
	for i, tc := range m.ToolCalls {
		id := tc.ID
		if id == "" {
			// CLAUDE.md: Ollama's native /api/chat sometimes omits an id
			// entirely — synthesize one. Unique within this message is
			// enough: it only ever has to match the ToolResult the loop
			// generates from this same ToolUse.
			id = fmt.Sprintf("ollama_call_%d", i)
		}
		input, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("tool_call %d arguments: %w", i, err)
		}
		blocks = append(blocks, llm.ToolUse{ID: id, Name: tc.Function.Name, Input: input})
	}
	return blocks, nil
}
