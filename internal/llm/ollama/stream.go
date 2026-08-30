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
	"github.com/reno/pico-code/internal/llm/recordutil"
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

	recordutil.LogBytes(ctx, "ollama: stream request", body)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnreachable, err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		defer func() { _ = res.Body.Close() }()
		respBody, _ := io.ReadAll(res.Body)
		return nil, mapError(res.StatusCode, respBody)
	}

	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		defer func() { _ = res.Body.Close() }()

		toolCallSeq := 0
		var respText strings.Builder
		// held back and re-checked against narratedToolCall once the full
		// response is in, instead of being sent immediately as TextDelta,
		// for as long as the accumulated text still looks like it could be
		// one (see maybeNarratedCall below). Flushed unchanged the moment
		// that stops being true, so an ordinary reply pays no cost.
		var pending []string
		maybeNarratedCall := true
		flushPending := func() bool {
			for _, text := range pending {
				if !send(llm.TextDelta{Text: text}) {
					return false
				}
			}
			pending = nil
			return true
		}

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

			if len(chunk.Message.ToolCalls) > 0 && maybeNarratedCall {
				// Real structured tool_calls showed up after all, so
				// whatever text preceded them was never a narrated call —
				// release it as ordinary text.
				maybeNarratedCall = false
				if !flushPending() {
					return
				}
			}

			if chunk.Message.Thinking != "" {
				if !send(llm.ThinkingDelta{Text: chunk.Message.Thinking}) {
					return
				}
			}

			if chunk.Message.Content != "" {
				respText.WriteString(chunk.Message.Content)
				if trimmed := strings.TrimSpace(respText.String()); maybeNarratedCall && trimmed != "" && !strings.HasPrefix(trimmed, "{") {
					// Once we've seen a first non-whitespace character that
					// isn't '{', this can no longer become a bare JSON
					// object — give up the candidacy for good.
					maybeNarratedCall = false
				}
				if maybeNarratedCall {
					pending = append(pending, chunk.Message.Content)
				} else {
					if !flushPending() {
						return
					}
					if !send(llm.TextDelta{Text: chunk.Message.Content}) {
						return
					}
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
				recordutil.LogBytes(ctx, "ollama: stream response", line)
				if maybeNarratedCall {
					if name, input, ok := narratedToolCall(respText.String(), req.Tools); ok {
						if !send(llm.ToolUseStart{ID: "ollama_call_0", Name: name}) {
							return
						}
						if !send(llm.ToolUseDone{ID: "ollama_call_0", Input: input}) {
							return
						}
					} else if !flushPending() {
						return
					}
				}
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
