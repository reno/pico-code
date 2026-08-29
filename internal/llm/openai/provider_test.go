package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
)

func TestChatEndToEndAgainstFakeServer(t *testing.T) {
	var gotBody map[string]any
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("request path = %q, want /chat/completions", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5}
		}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, apiKey: "sk-test", model: "gpt-4o-mini"}
	resp, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if got := gotBody["model"]; got != "gpt-4o-mini" {
		t.Errorf("request model = %v, want gpt-4o-mini", got)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization header = %q, want %q", gotAuth, "Bearer sk-test")
	}
	if resp.StopReason != "stop" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "stop")
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v, want {10 5}", resp.Usage)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("len(Blocks) = %d, want 1", len(resp.Message.Blocks))
	}
	text, ok := resp.Message.Blocks[0].(llm.Text)
	if !ok || text.Text != "hi there" {
		t.Errorf("Blocks[0] = %#v, want Text{\"hi there\"}", resp.Message.Blocks[0])
	}
}

// TestChatNoAPIKeyOmitsAuthHeader covers the adapter's key difference from
// Anthropic's: a compatible local endpoint with no auth at all must still
// work, so an empty APIKey must not send an empty/malformed Authorization
// header.
func TestChatNoAPIKeyOmitsAuthHeader(t *testing.T) {
	var gotAuth string
	var sawAuthHeader bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, sawAuthHeader = r.Header.Get("Authorization"), r.Header.Get("Authorization") != ""
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "local-model"}
	if _, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if sawAuthHeader {
		t.Errorf("Authorization header = %q, want no header sent when APIKey is empty", gotAuth)
	}
}

func TestChatHTTPErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error": {"message": "rate limited"}}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "gpt-4o-mini"}
	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("Chat() error = %v, want wrapping ErrRateLimited", err)
	}
}

func TestStreamNotImplemented(t *testing.T) {
	p := &Provider{httpClient: http.DefaultClient, baseURL: "http://example.invalid", model: "gpt-4o-mini"}
	if _, err := p.Stream(context.Background(), llm.Request{}); err == nil {
		t.Fatal("Stream() error = nil, want an error until phase 12.2 implements it")
	}
}

func TestNewDefaultsBaseURL(t *testing.T) {
	p, err := New(&config.Config{Provider: config.ProviderOpenAI, Model: "gpt-4o-mini"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	prov, ok := p.(*Provider)
	if !ok {
		t.Fatalf("New() = %T, want *Provider", p)
	}
	if prov.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", prov.baseURL, defaultBaseURL)
	}
}

func TestNewTrimsTrailingSlashFromBaseURL(t *testing.T) {
	p, err := New(&config.Config{Provider: config.ProviderOpenAI, Model: "m", OpenAIBaseURL: "https://example.test/v1/"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	prov := p.(*Provider)
	if prov.baseURL != "https://example.test/v1" {
		t.Errorf("baseURL = %q, want trailing slash trimmed", prov.baseURL)
	}
}
