package ollama

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/conformance"
)

// TestConformance is 6.4's AC: the shared conformance suite passes for the
// Ollama adapter unchanged.
func TestConformance(t *testing.T) {
	conformance.Run(t, conformance.Harness{
		Name: "ollama",
		NewProvider: func(baseURL string) llm.Provider {
			return &Provider{httpClient: http.DefaultClient, baseURL: baseURL, model: "qwen3:8b", numCtx: 4096}
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
			"model": "qwen3:8b",
			"message": {"role": "assistant", "content": "hello"},
			"done": true, "done_reason": "stop",
			"prompt_eval_count": 5, "eval_count": 2
		}`)
	case conformance.ScenarioToolCall:
		return jsonServer(`{
			"model": "qwen3:8b",
			"message": {"role": "assistant", "content": "", "tool_calls": [
				{"id": "call_ct1", "function": {"name": "get_weather", "arguments": {"location":"Paris"}}}
			]},
			"done": true, "done_reason": "stop",
			"prompt_eval_count": 8, "eval_count": 4
		}`)
	case conformance.ScenarioSimpleTextStream:
		return ndjsonServer(
			`{"model":"qwen3:8b","message":{"role":"assistant","content":"hello"},"done":false}`,
			`{"model":"qwen3:8b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}`,
		)
	case conformance.ScenarioToolCallStream:
		return ndjsonServer(
			`{"model":"qwen3:8b","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_ct1","function":{"name":"get_weather","arguments":{"location":"Paris"}}}]},"done":false}`,
			`{"model":"qwen3:8b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":8,"eval_count":4}`,
		)
	case conformance.ScenarioServerError:
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"bad key"}`))
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

func ndjsonServer(lines ...string) *httptest.Server {
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
}
