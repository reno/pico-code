package ollama

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

func TestMapErrorNotFound(t *testing.T) {
	got := mapError(http.StatusNotFound, []byte(`{"error":"model 'ghost' not found, try pulling it first"}`))
	if !errors.Is(got, ErrNotFound) {
		t.Errorf("mapError(404) = %v, want wrapping %v", got, ErrNotFound)
	}
}

func TestMapErrorUnmappedStatusPassesThrough(t *testing.T) {
	got := mapError(http.StatusInternalServerError, []byte("boom"))
	if got == nil {
		t.Fatal("mapError() = nil, want an error for a 5xx status")
	}
	if errors.Is(got, ErrNotFound) {
		t.Errorf("mapError(500) wraps ErrNotFound, want an unmapped status to pass through unclassified")
	}
}

// unreachableURL returns an httptest.Server address that's guaranteed to
// refuse connections: closing the server frees the port without anything
// else picking it back up before the request is attempted.
func unreachableURL(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	srv.Close()
	return srv.URL
}

// TestProbeShowUnreachableDaemon covers the reported failure mode: no
// Ollama daemon listening at all (e.g. "ollama serve" was never started),
// hit via ValidateModel's and checkToolSupport's shared probeShow path.
func TestProbeShowUnreachableDaemon(t *testing.T) {
	p := &Provider{httpClient: http.DefaultClient, baseURL: unreachableURL(t), model: "qwen3:8b"}

	_, err := p.probeShow(context.Background(), "qwen3:8b")
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("probeShow() error = %v, want wrapping %v", err, ErrUnreachable)
	}
}

func TestChatUnreachableDaemon(t *testing.T) {
	p := &Provider{httpClient: http.DefaultClient, baseURL: unreachableURL(t), model: "qwen3:8b", numCtx: 4096}

	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("Chat() error = %v, want wrapping %v", err, ErrUnreachable)
	}
}

func TestStreamUnreachableDaemon(t *testing.T) {
	p := &Provider{httpClient: http.DefaultClient, baseURL: unreachableURL(t), model: "qwen3:8b", numCtx: 4096}

	_, err := p.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("Stream() error = %v, want wrapping %v", err, ErrUnreachable)
	}
}
