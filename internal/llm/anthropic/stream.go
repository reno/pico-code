package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/reno/pico-code/internal/llm"
)

// streamAccumulator tracks the per-block state needed to translate
// Anthropic's SSE events into canonical llm.Events: which content-block
// Index is which tool call's ID, its raw (unparsed) input_json_delta
// fragments, and the running usage/stop-reason that only becomes complete
// once message_delta and message_stop arrive.
type streamAccumulator struct {
	toolID  map[int64]string
	toolBuf map[int64]*bytes.Buffer

	usage      llm.Usage
	stopReason string
}

func newStreamAccumulator() *streamAccumulator {
	return &streamAccumulator{toolID: map[int64]string{}, toolBuf: map[int64]*bytes.Buffer{}}
}

// handle translates one SSE event into zero or more canonical events.
func (a *streamAccumulator) handle(event sdk.MessageStreamEventUnion) ([]llm.Event, error) {
	switch event.Type {
	case "message_start":
		ms := event.AsMessageStart()
		a.usage.InputTokens = int(ms.Message.Usage.InputTokens)
		a.usage.OutputTokens = int(ms.Message.Usage.OutputTokens)
		return nil, nil

	case "content_block_start":
		cbs := event.AsContentBlockStart()
		if cbs.ContentBlock.Type != "tool_use" {
			return nil, nil
		}
		a.toolID[cbs.Index] = cbs.ContentBlock.ID
		a.toolBuf[cbs.Index] = &bytes.Buffer{}
		return []llm.Event{llm.ToolUseStart{ID: cbs.ContentBlock.ID, Name: cbs.ContentBlock.Name}}, nil

	case "content_block_delta":
		cbd := event.AsContentBlockDelta()
		switch cbd.Delta.Type {
		case "text_delta":
			return []llm.Event{llm.TextDelta{Text: cbd.Delta.Text}}, nil
		case "input_json_delta":
			id, ok := a.toolID[cbd.Index]
			if !ok {
				return nil, fmt.Errorf("anthropic: input_json_delta for unknown block index %d", cbd.Index)
			}
			a.toolBuf[cbd.Index].WriteString(cbd.Delta.PartialJSON)
			return []llm.Event{llm.ToolUseArgsDelta{ID: id, Partial: cbd.Delta.PartialJSON}}, nil
		default:
			return nil, nil
		}

	case "content_block_stop":
		cbs := event.AsContentBlockStop()
		id, ok := a.toolID[cbs.Index]
		if !ok {
			return nil, nil
		}
		raw := a.toolBuf[cbs.Index].Bytes()
		if len(raw) == 0 {
			raw = []byte("{}")
		}
		if !json.Valid(raw) {
			return nil, fmt.Errorf("anthropic: tool_use %s: accumulated input is not valid JSON: %s", id, raw)
		}
		return []llm.Event{llm.ToolUseDone{ID: id, Input: json.RawMessage(raw)}}, nil

	case "message_delta":
		md := event.AsMessageDelta()
		a.stopReason = string(md.Delta.StopReason)
		if md.Usage.OutputTokens > 0 {
			a.usage.OutputTokens = int(md.Usage.OutputTokens)
		}
		if md.Usage.InputTokens > 0 {
			a.usage.InputTokens = int(md.Usage.InputTokens)
		}
		return nil, nil

	case "message_stop":
		return []llm.Event{llm.MessageDone{StopReason: a.stopReason, Usage: a.usage}}, nil

	default:
		return nil, nil
	}
}

// Stream sends req and translates the SSE response into canonical events
// via llm.StreamEvents, which owns the close-exactly-once/exit-on-cancel
// channel contract.
func (p *Provider) Stream(ctx context.Context, req llm.Request) (<-chan llm.Event, error) {
	params, err := toParams(req, p.model)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}

	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		stream := p.client.Messages.NewStreaming(ctx, params)
		defer func() { _ = stream.Close() }()

		acc := newStreamAccumulator()
		for stream.Next() {
			events, err := acc.handle(stream.Current())
			if err != nil {
				send(llm.Error{Err: fmt.Errorf("anthropic: %w", err)})
				return
			}
			for _, e := range events {
				if !send(e) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			send(llm.Error{Err: mapError(err)})
		}
	}), nil
}
