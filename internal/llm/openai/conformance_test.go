package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/conformance"
)

// TestConformance is 12.2's AC: the shared conformance suite (6.4) passes
// for this adapter unchanged, and the only diff needed to add it is this
// one file — no provider branches inside the suite itself.
func TestConformance(t *testing.T) {
	conformance.Run(t, conformance.Harness{
		Name: "openai",
		NewProvider: func(baseURL string) llm.Provider {
			return &Provider{httpClient: http.DefaultClient, baseURL: baseURL, model: "gpt-4o-mini"}
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
			"id": "chatcmpl-ct1",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 2}
		}`)
	case conformance.ScenarioToolCall:
		return jsonServer(`{
			"id": "chatcmpl-ct2",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_ct1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}
			]}, "finish_reason": "tool_calls"}],
			"usage": {"prompt_tokens": 8, "completion_tokens": 4}
		}`)
	case conformance.ScenarioSimpleTextStream:
		return sseServer(
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant"}, "finish_reason": nil}}}),
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": "hello"}, "finish_reason": nil}}}),
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}}}),
			sseData(map[string]any{"choices": []any{}, "usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2}}),
			"data: [DONE]\n\n",
		)
	case conformance.ScenarioToolCallStream:
		return sseServer(
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant"}, "finish_reason": nil}}}),
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
				map[string]any{"index": 0, "id": "call_ct1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": ""}},
			}}, "finish_reason": nil}}}),
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
				map[string]any{"index": 0, "function": map[string]any{"arguments": `{"location":"Paris"}`}},
			}}, "finish_reason": nil}}}),
			sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}}}),
			sseData(map[string]any{"choices": []any{}, "usage": map[string]any{"prompt_tokens": 8, "completion_tokens": 4}}),
			"data: [DONE]\n\n",
		)
	case conformance.ScenarioServerError:
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error": {"message": "bad key"}}`))
		}))
	case conformance.ScenarioHang:
		// Bounded rather than truly infinite — see the identical comment
		// in the Anthropic conformance harness for why.
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

func sseData(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return "data: " + string(b) + "\n\n"
}
