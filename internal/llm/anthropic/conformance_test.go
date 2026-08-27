package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/conformance"
)

// TestConformance is 6.4's AC: the shared conformance suite passes for the
// Anthropic adapter unchanged.
func TestConformance(t *testing.T) {
	conformance.Run(t, conformance.Harness{
		Name: "anthropic",
		NewProvider: func(baseURL string) llm.Provider {
			return testProvider(baseURL)
		},
		Server:      conformanceServer,
		WantText:    "hello",
		WantToolUse: llm.ToolUse{Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
	})
}

func conformanceServer(t *testing.T, scenario conformance.Scenario) *httptest.Server {
	t.Helper()
	switch scenario {
	case conformance.ScenarioSimpleText:
		return jsonServer(`{
			"id": "msg_ct1", "type": "message", "role": "assistant", "model": "claude-test-model",
			"content": [{"type": "text", "text": "hello"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 5, "output_tokens": 2}
		}`)
	case conformance.ScenarioToolCall:
		return jsonServer(`{
			"id": "msg_ct2", "type": "message", "role": "assistant", "model": "claude-test-model",
			"content": [{"type": "tool_use", "id": "toolu_ct1", "name": "get_weather", "input": {"location":"Paris"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 8, "output_tokens": 4}
		}`)
	case conformance.ScenarioSimpleTextStream:
		return sseServer(
			sseEvent("message_start", map[string]any{"type": "message_start", "message": map[string]any{
				"id": "msg_ct1", "type": "message", "role": "assistant", "model": "claude-test-model",
				"content": []any{}, "usage": map[string]any{"input_tokens": 5, "output_tokens": 1},
			}}),
			sseEvent("content_block_start", map[string]any{"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "text", "text": ""}}),
			sseEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "text_delta", "text": "hello"}}),
			sseEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}),
			sseEvent("message_delta", map[string]any{"type": "message_delta",
				"delta": map[string]any{"stop_reason": "end_turn"}, "usage": map[string]any{"output_tokens": 2}}),
			sseEvent("message_stop", map[string]any{"type": "message_stop"}),
		)
	case conformance.ScenarioToolCallStream:
		return sseServer(
			sseEvent("message_start", map[string]any{"type": "message_start", "message": map[string]any{
				"id": "msg_ct2", "type": "message", "role": "assistant", "model": "claude-test-model",
				"content": []any{}, "usage": map[string]any{"input_tokens": 8, "output_tokens": 1},
			}}),
			sseEvent("content_block_start", map[string]any{"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "tool_use", "id": "toolu_ct1", "name": "get_weather", "input": map[string]any{}}}),
			sseEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"location":"Paris"}`}}),
			sseEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}),
			sseEvent("message_delta", map[string]any{"type": "message_delta",
				"delta": map[string]any{"stop_reason": "tool_use"}, "usage": map[string]any{"output_tokens": 4}}),
			sseEvent("message_stop", map[string]any{"type": "message_stop"}),
		)
	case conformance.ScenarioServerError:
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`))
		}))
	case conformance.ScenarioHang:
		// Bounded, not truly infinite: a cancelled client doesn't always
		// close its TCP connection quickly enough for the server to
		// observe via r.Context(), which would otherwise hang
		// httptest.Server.Close() waiting for this handler to return. What
		// this scenario actually tests is the client (our provider)
		// returning promptly on its own ctx cancellation, not the
		// server's.
		return httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
			case <-time.After(150 * time.Millisecond):
			}
		}))
	default:
		t.Fatalf("conformanceServer: unhandled scenario %v", scenario)
		return nil
	}
}

func jsonServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
}

func sseServer(events ...string) *httptest.Server {
	body := strings.Join(events, "")
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
}

func sseEvent(name string, data any) string {
	b, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("event: %s\ndata: %s\n\n", name, string(b))
}
