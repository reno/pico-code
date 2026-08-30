package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/recordutil"
)

// streamChunk is one SSE "data:" payload from a streaming /chat/completions
// response.
type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Usage   *responseUsage `json:"usage"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type streamDelta struct {
	Content   string           `json:"content"`
	ToolCalls []streamToolCall `json:"tool_calls"`
}

// streamToolCall is one entry in a delta's flat tool_calls array, keyed by
// Index rather than ID: only the chunk that introduces a call carries its
// ID and function name, every later fragment for the same call repeats
// just Index and a piece of Function.Arguments — CLAUDE.md's "per-index
// argument accumulation".
type streamToolCall struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id"`
	Function streamToolCallFunction `json:"function"`
}

type streamToolCallFunction struct {
	Name string `json:"name"`
	// Arguments is always a fragment of the documented JSON-encoded
	// string in streaming mode — unlike the non-streaming response, there
	// is no object-vs-string ambiguity here, since a partial JSON value
	// can only ever be represented as partial text.
	Arguments string `json:"arguments"`
}

// streamAccumulator tracks the per-index state needed to translate a
// stream of chatCompletion chunks into canonical llm.Events: which
// tool_calls array index is which call's ID, its accumulated (unparsed)
// arguments fragments, and the running usage/stop-reason that only becomes
// final once a finish_reason and (often a separate, later) usage-bearing
// chunk have both arrived.
type streamAccumulator struct {
	toolOrder []int
	toolID    map[int]string
	toolBuf   map[int]*bytes.Buffer

	finished   bool
	stopReason string
	usage      llm.Usage
}

func newStreamAccumulator() *streamAccumulator {
	return &streamAccumulator{toolID: map[int]string{}, toolBuf: map[int]*bytes.Buffer{}}
}

// handle translates one chunk into zero or more canonical events. Usage is
// captured whenever present, even after finished is set: CLAUDE.md's
// required stream_options.include_usage often lands its usage on a
// trailing chunk with an empty Choices array, sent after the chunk that
// carried finish_reason.
func (a *streamAccumulator) handle(chunk streamChunk) ([]llm.Event, error) {
	if chunk.Usage != nil {
		a.usage = llm.Usage{InputTokens: chunk.Usage.PromptTokens, OutputTokens: chunk.Usage.CompletionTokens}
	}
	if a.finished || len(chunk.Choices) == 0 {
		return nil, nil
	}
	choice := chunk.Choices[0]

	var events []llm.Event
	if choice.Delta.Content != "" {
		events = append(events, llm.TextDelta{Text: choice.Delta.Content})
	}

	for _, tc := range choice.Delta.ToolCalls {
		if _, seen := a.toolID[tc.Index]; !seen {
			id := tc.ID
			if id == "" {
				// CLAUDE.md: the canonical format always carries an ID,
				// even when a backend's own chunk omits one.
				id = fmt.Sprintf("openai_call_%d", tc.Index)
			}
			a.toolID[tc.Index] = id
			a.toolBuf[tc.Index] = &bytes.Buffer{}
			a.toolOrder = append(a.toolOrder, tc.Index)
			events = append(events, llm.ToolUseStart{ID: id, Name: tc.Function.Name})
		}
		if tc.Function.Arguments != "" {
			a.toolBuf[tc.Index].WriteString(tc.Function.Arguments)
			events = append(events, llm.ToolUseArgsDelta{ID: a.toolID[tc.Index], Partial: tc.Function.Arguments})
		}
	}

	if choice.FinishReason != "" {
		a.finished = true
		a.stopReason = choice.FinishReason
		for _, idx := range a.toolOrder {
			raw := a.toolBuf[idx].Bytes()
			if len(raw) == 0 {
				raw = []byte("{}")
			}
			if !json.Valid(raw) {
				return nil, fmt.Errorf("tool_call %s: accumulated arguments are not valid JSON: %s", a.toolID[idx], raw)
			}
			events = append(events, llm.ToolUseDone{ID: a.toolID[idx], Input: json.RawMessage(raw)})
		}
	}

	return events, nil
}

// Stream sends req to POST {baseURL}/chat/completions with streaming
// enabled and translates the SSE response into canonical events via
// llm.StreamEvents, which owns the close-exactly-once/exit-on-cancel
// channel contract.
func (p *Provider) Stream(ctx context.Context, req llm.Request) (<-chan llm.Event, error) {
	cr, err := toRequest(req, p.model)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	cr.Stream = true
	cr.StreamOptions = &streamOptions{IncludeUsage: true}

	body, err := json.Marshal(cr)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	recordutil.LogBytes(ctx, "openai: stream request", body, p.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		defer func() { _ = res.Body.Close() }()
		respBody, _ := io.ReadAll(res.Body)
		return nil, mapError(res.StatusCode, respBody)
	}

	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		defer func() { _ = res.Body.Close() }()

		acc := newStreamAccumulator()
		scanner := bufio.NewScanner(res.Body)
		scanner.Buffer(nil, 1024*1024)

	readLoop:
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				break readLoop
			}

			var chunk streamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				send(llm.Error{Err: fmt.Errorf("openai: decode stream chunk: %w", err)})
				return
			}
			events, err := acc.handle(chunk)
			if err != nil {
				send(llm.Error{Err: fmt.Errorf("openai: %w", err)})
				return
			}
			for _, e := range events {
				if !send(e) {
					return
				}
			}
		}
		if err := scanner.Err(); err != nil {
			send(llm.Error{Err: fmt.Errorf("openai: read stream: %w", err)})
			return
		}
		if !acc.finished {
			return
		}
		recordutil.LogJSON(ctx, "openai: stream response", struct {
			StopReason string    `json:"stop_reason"`
			Usage      llm.Usage `json:"usage"`
		}{acc.stopReason, acc.usage}, p.apiKey)
		send(llm.MessageDone{StopReason: acc.stopReason, Usage: acc.usage})
	}), nil
}
