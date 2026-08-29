package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/reno/pico-code/internal/llm"
)

// chatRequest is the wire shape of a POST to /chat/completions, kept to the
// subset every OpenAI-compatible backend this adapter targets (OpenAI,
// vLLM, LM Studio, Ollama's /v1) actually implements.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Tools       []toolParam   `json:"tools,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Stop        []string      `json:"stop,omitempty"`

	// Stream and StreamOptions are only set by Stream, never by Chat — the
	// zero values (false, nil) both omitempty away, so toRequest's output
	// is identical for both callers until Stream overrides them.
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

// streamOptions.IncludeUsage asks a streaming response to add a trailing
// chunk carrying Usage — without it, most compatible backends never report
// token counts for a streamed call at all.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatMessage is one entry in chatRequest.Messages or chatResponse's
// choice.Message. ToolCallID is only ever set on a role:"tool" message
// (CLAUDE.md: the OpenAI-compatible shape correlates results by
// tool_call_id plus message ordering, unlike Anthropic's tool_use_id on a
// content block).
type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// toolCall is one entry in an assistant message's flat tool_calls array —
// CLAUDE.md invariant 1's "OpenAI-shaped flat tool_calls format" is exactly
// this shape, in contrast to Anthropic's ToolUse content block.
type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

// toolCallFunction carries a call's name and arguments. Outgoing, Arguments
// is always the JSON-encoded string the spec documents. Incoming, it is
// decoded with responseToolCallFunction instead: CLAUDE.md's "arguments
// accepted as an object or a JSON-encoded string" gotcha applies here too,
// since not every compatible backend follows the string-only spec.
type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// toolParam advertises one tool in chatRequest.Tools.
type toolParam struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

// toolFunction's Parameters is the tool's generated JSON Schema
// (internal/tools' GenerateSchema output) passed through untouched: unlike
// Anthropic's and Ollama's SDK-typed params, OpenAI's "parameters" field
// takes a full JSON Schema object directly, so no reshaping is needed.
type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// toRequest translates a canonical Request into a chatRequest for model.
func toRequest(req llm.Request, model string) (*chatRequest, error) {
	cr := &chatRequest{
		Model:     model,
		MaxTokens: req.MaxTokens,
	}
	if req.Temperature != 0 {
		cr.Temperature = req.Temperature
	}
	if len(req.StopSequences) > 0 {
		cr.Stop = req.StopSequences
	}
	// req.Think (16.1) is intentionally ignored here: chatRequest has no
	// equivalent field, and Request.Think is defined to be a no-op for a
	// provider that doesn't support it, not an error.

	var msgs []chatMessage
	if req.System != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: req.System})
	}
	for i, m := range req.Messages {
		converted, err := toMessages(m)
		if err != nil {
			return nil, fmt.Errorf("message %d: %w", i, err)
		}
		msgs = append(msgs, converted...)
	}
	cr.Messages = msgs

	if len(req.Tools) > 0 {
		tools := make([]toolParam, len(req.Tools))
		for i, td := range req.Tools {
			tools[i] = toolParam{
				Type: "function",
				Function: toolFunction{
					Name:        td.Name,
					Description: td.Description,
					Parameters:  td.InputSchema,
				},
			}
		}
		cr.Tools = tools
	}

	return cr, nil
}

// toMessages expands one canonical Message into the chatMessages it maps
// to. A user message answering N parallel tool calls expands to N
// role:"tool" messages, one per ToolResult — the flat shape has no
// equivalent of Anthropic's array of content blocks within one message.
func toMessages(m llm.Message) ([]chatMessage, error) {
	switch m.Role {
	case llm.RoleUser:
		return toUserMessages(m.Blocks)
	case llm.RoleAssistant:
		return toAssistantMessages(m.Blocks)
	default:
		return nil, fmt.Errorf("unsupported role %q", m.Role)
	}
}

func toUserMessages(blocks []llm.Block) ([]chatMessage, error) {
	var text strings.Builder
	var toolMsgs []chatMessage
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(v.Text)
		case llm.ToolResult:
			toolMsgs = append(toolMsgs, chatMessage{Role: "tool", Content: v.Content, ToolCallID: v.ToolUseID})
		default:
			return nil, fmt.Errorf("unsupported block type %T in user message", b)
		}
	}
	if text.Len() == 0 {
		return toolMsgs, nil
	}
	return append([]chatMessage{{Role: "user", Content: text.String()}}, toolMsgs...), nil
}

func toAssistantMessages(blocks []llm.Block) ([]chatMessage, error) {
	msg := chatMessage{Role: "assistant"}
	var text strings.Builder
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(v.Text)
		case llm.ToolUse:
			args := string(v.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, toolCall{
				ID:       v.ID,
				Type:     "function",
				Function: toolCallFunction{Name: v.Name, Arguments: args},
			})
		default:
			return nil, fmt.Errorf("unsupported block type %T in assistant message", b)
		}
	}
	msg.Content = text.String()
	return []chatMessage{msg}, nil
}

// chatResponse is the wire shape of a non-streaming /chat/completions
// response.
type chatResponse struct {
	Choices []responseChoice `json:"choices"`
	Usage   responseUsage    `json:"usage"`
}

type responseChoice struct {
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// responseMessage mirrors chatMessage but decodes tool call arguments with
// responseToolCall, since incoming arguments may not follow the
// string-only spec (CLAUDE.md's argument-shape gotcha).
type responseMessage struct {
	Content   string             `json:"content"`
	ToolCalls []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       string                   `json:"id"`
	Function responseToolCallFunction `json:"function"`
}

type responseToolCallFunction struct {
	Name string `json:"name"`
	// Arguments is decoded as raw JSON rather than string: some
	// OpenAI-compatible backends emit the documented JSON-encoded string,
	// others emit the argument object directly. decodeArguments below
	// normalizes either into the object bytes canonical ToolUse.Input
	// expects.
	Arguments json.RawMessage `json:"arguments"`
}

type responseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// fromResponse translates a non-streaming chatResponse into the canonical
// Response. It errors on zero choices: every real backend returns at least
// one for a successful call, so an empty array signals a shape this
// adapter doesn't understand rather than a valid empty answer.
func fromResponse(resp *chatResponse) (*llm.Response, error) {
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("response has no choices")
	}
	choice := resp.Choices[0]

	blocks, err := fromMessage(choice.Message)
	if err != nil {
		return nil, err
	}

	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: choice.FinishReason,
		Usage: llm.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}, nil
}

func fromMessage(m responseMessage) ([]llm.Block, error) {
	var blocks []llm.Block
	if m.Content != "" {
		blocks = append(blocks, llm.Text{Text: m.Content})
	}
	for i, tc := range m.ToolCalls {
		id := tc.ID
		if id == "" {
			// CLAUDE.md: the canonical format always carries an ID, even
			// when the backend's own response omits one.
			id = fmt.Sprintf("openai_call_%d", i)
		}
		input, err := decodeArguments(tc.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("tool_call %d arguments: %w", i, err)
		}
		blocks = append(blocks, llm.ToolUse{ID: id, Name: tc.Function.Name, Input: input})
	}
	return blocks, nil
}

// decodeArguments normalizes a tool call's arguments into the JSON object
// bytes ToolUse.Input expects, whether raw arrived as a JSON-encoded string
// (the documented shape) or as the object itself (a quirk some compatible
// backends produce — same idea as Ollama's normalizeToolCallArguments, but
// applied while decoding rather than by rewriting the wire bytes, since
// this adapter owns its own response struct).
func decodeArguments(raw json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return json.RawMessage("{}"), nil
	}
	if trimmed[0] != '"' {
		return trimmed, nil
	}

	var encoded string
	if err := json.Unmarshal(trimmed, &encoded); err != nil {
		return nil, fmt.Errorf("parse arguments string: %w", err)
	}
	if encoded == "" {
		return json.RawMessage("{}"), nil
	}
	if !json.Valid([]byte(encoded)) {
		return nil, fmt.Errorf("arguments string is not valid JSON once decoded: %q", encoded)
	}
	return json.RawMessage(encoded), nil
}
