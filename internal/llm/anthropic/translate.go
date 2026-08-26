// Package anthropic adapts the Anthropic Messages API to the canonical
// llm.Provider interface. It is the only place that imports
// anthropic-sdk-go; internal/llm itself never does (CLAUDE.md invariant 1).
package anthropic

import (
	"encoding/json"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/reno/pico-code/internal/llm"
)

// toParams translates a canonical Request into the SDK's request params for
// the given model. CLAUDE.md forbids hardcoding an Anthropic model
// constant, so model always comes from the caller (ultimately config).
func toParams(req llm.Request, model string) (sdk.MessageNewParams, error) {
	params := sdk.MessageNewParams{
		Model:     sdk.Model(model),
		MaxTokens: int64(req.MaxTokens),
	}

	if req.System != "" {
		params.System = []sdk.TextBlockParam{{Text: req.System}}
	}
	if req.Temperature != 0 {
		params.Temperature = param.NewOpt(req.Temperature)
	}
	if len(req.StopSequences) > 0 {
		params.StopSequences = req.StopSequences
	}

	msgs := make([]sdk.MessageParam, len(req.Messages))
	for i, m := range req.Messages {
		blocks, err := toContentBlocks(m.Blocks)
		if err != nil {
			return sdk.MessageNewParams{}, fmt.Errorf("message %d: %w", i, err)
		}
		switch m.Role {
		case llm.RoleUser:
			msgs[i] = sdk.NewUserMessage(blocks...)
		case llm.RoleAssistant:
			msgs[i] = sdk.NewAssistantMessage(blocks...)
		default:
			return sdk.MessageNewParams{}, fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
	}
	params.Messages = msgs

	if len(req.Tools) > 0 {
		tools := make([]sdk.ToolUnionParam, len(req.Tools))
		for i, td := range req.Tools {
			schema, err := toInputSchema(td.InputSchema)
			if err != nil {
				return sdk.MessageNewParams{}, fmt.Errorf("tool %q: %w", td.Name, err)
			}
			tp := sdk.ToolParam{Name: td.Name, InputSchema: schema}
			if td.Description != "" {
				tp.Description = param.NewOpt(td.Description)
			}
			tools[i] = sdk.ToolUnionParam{OfTool: &tp}
		}
		params.Tools = tools
	}

	return params, nil
}

func toContentBlocks(blocks []llm.Block) ([]sdk.ContentBlockParamUnion, error) {
	out := make([]sdk.ContentBlockParamUnion, len(blocks))
	for i, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			out[i] = sdk.NewTextBlock(v.Text)
		case llm.ToolUse:
			out[i] = sdk.NewToolUseBlock(v.ID, v.Input, v.Name)
		case llm.ToolResult:
			out[i] = sdk.NewToolResultBlock(v.ToolUseID, v.Content, v.IsError)
		default:
			return nil, fmt.Errorf("unsupported block type %T", b)
		}
	}
	return out, nil
}

// toInputSchema lifts a tool's generated JSON Schema (internal/tools'
// GenerateSchema output: {"type":"object","properties":...,"required":...})
// into the SDK's narrower ToolInputSchemaParam shape.
func toInputSchema(raw json.RawMessage) (sdk.ToolInputSchemaParam, error) {
	var parsed struct {
		Properties any      `json:"properties"`
		Required   []string `json:"required"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return sdk.ToolInputSchemaParam{}, fmt.Errorf("parse input schema: %w", err)
	}
	return sdk.ToolInputSchemaParam{
		Properties: parsed.Properties,
		Required:   parsed.Required,
	}, nil
}

// fromResponse translates an SDK Message (the result of a non-streaming
// call) into the canonical Response.
func fromResponse(msg *sdk.Message) (*llm.Response, error) {
	blocks := make([]llm.Block, len(msg.Content))
	for i, cb := range msg.Content {
		b, err := fromContentBlock(cb)
		if err != nil {
			return nil, fmt.Errorf("content block %d: %w", i, err)
		}
		blocks[i] = b
	}

	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: string(msg.StopReason),
		Usage: llm.Usage{
			InputTokens:  int(msg.Usage.InputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
	}, nil
}

func fromContentBlock(cb sdk.ContentBlockUnion) (llm.Block, error) {
	switch cb.Type {
	case "text":
		return llm.Text{Text: cb.Text}, nil
	case "tool_use":
		return llm.ToolUse{ID: cb.ID, Name: cb.Name, Input: cb.Input}, nil
	default:
		return nil, fmt.Errorf("unsupported content block type %q", cb.Type)
	}
}
