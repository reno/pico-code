package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"

	"github.com/reno/pico-code/internal/llm"
)

// Stream sends req to /api/chat with streaming enabled and translates each
// NDJSON line into canonical events. Unlike Anthropic, Ollama doesn't
// fragment a tool call's arguments across lines — each line's tool_calls
// are already complete — so a tool call becomes an immediate
// ToolUseStart/ToolUseDone pair rather than accumulating deltas.
func (p *Provider) Stream(ctx context.Context, req llm.Request) (<-chan llm.Event, error) {
	if err := p.checkToolSupport(ctx, req); err != nil {
		return nil, err
	}

	ar, err := toRequest(req, p.model, p.numCtx)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	stream := true
	ar.Stream = &stream

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		defer func() { _ = res.Body.Close() }()
		respBody, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("ollama: request failed with status %d: %s", res.StatusCode, bytes.TrimSpace(respBody))
	}

	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		defer func() { _ = res.Body.Close() }()

		toolCallSeq := 0
		var respText strings.Builder
		scanner := bufio.NewScanner(res.Body)
		scanner.Buffer(nil, 1024*1024)

		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}

			normalized, err := normalizeToolCallArguments(line)
			if err != nil {
				send(llm.Error{Err: fmt.Errorf("ollama: %w", err)})
				return
			}
			var chunk api.ChatResponse
			if err := json.Unmarshal(normalized, &chunk); err != nil {
				send(llm.Error{Err: fmt.Errorf("ollama: decode stream chunk: %w", err)})
				return
			}

			if chunk.Message.Content != "" {
				respText.WriteString(chunk.Message.Content)
				if !send(llm.TextDelta{Text: chunk.Message.Content}) {
					return
				}
			}

			for _, tc := range chunk.Message.ToolCalls {
				id := tc.ID
				if id == "" {
					id = fmt.Sprintf("ollama_call_%d", toolCallSeq)
				}
				toolCallSeq++

				input, err := json.Marshal(tc.Function.Arguments)
				if err != nil {
					send(llm.Error{Err: fmt.Errorf("ollama: tool call arguments: %w", err)})
					return
				}
				if !send(llm.ToolUseStart{ID: id, Name: tc.Function.Name}) {
					return
				}
				if !send(llm.ToolUseDone{ID: id, Input: input}) {
					return
				}
			}

			if chunk.Done {
				send(llm.MessageDone{
					StopReason: chunk.DoneReason,
					Usage: estimateUsage(req, respText.String(), llm.Usage{
						InputTokens:  chunk.PromptEvalCount,
						OutputTokens: chunk.EvalCount,
					}),
				})
				return
			}
		}
		if err := scanner.Err(); err != nil {
			send(llm.Error{Err: fmt.Errorf("ollama: read stream: %w", err)})
		}
	}), nil
}
