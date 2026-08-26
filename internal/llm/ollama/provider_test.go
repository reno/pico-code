package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

func TestChatEndToEndAgainstFakeServer(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("request path = %q, want /api/chat", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "qwen3:8b",
			"message": {"role": "assistant", "content": "hi there"},
			"done": true,
			"done_reason": "stop",
			"prompt_eval_count": 10,
			"eval_count": 5
		}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen3:8b", numCtx: 8192}
	resp, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if got := gotBody["model"]; got != "qwen3:8b" {
		t.Errorf("request model = %v, want qwen3:8b", got)
	}
	options, _ := gotBody["options"].(map[string]any)
	if got := options["num_ctx"]; got != float64(8192) {
		t.Errorf("request options.num_ctx = %v, want 8192", got)
	}
	if resp.StopReason != "stop" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "stop")
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("len(Blocks) = %d, want 1", len(resp.Message.Blocks))
	}
	text, ok := resp.Message.Blocks[0].(llm.Text)
	if !ok || text.Text != "hi there" {
		t.Errorf("Blocks[0] = %#v, want Text{\"hi there\"}", resp.Message.Blocks[0])
	}
}

func TestChatHTTPErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("model not found"))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen3:8b", numCtx: 4096}
	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("Chat() error = nil, want an error for a 500 response")
	}
}
