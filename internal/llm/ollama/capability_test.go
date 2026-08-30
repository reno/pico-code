package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

func fakeShowServer(t *testing.T, fixture string) (*httptest.Server, *int32) {
	t.Helper()
	body, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", fixture, err)
	}

	var requests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			t.Errorf("request path = %q, want /api/show", r.URL.Path)
		}
		atomic.AddInt32(&requests, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	return srv, &requests
}

// TestSupportsToolsDecision is 3.2's AC: a fixture for a tool-capable and a
// non-capable model yield the right decision.
func TestSupportsToolsDecision(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		want    bool
	}{
		{"tool-capable model", "testdata/golden/show_tool_capable.json", true},
		{"non-capable model", "testdata/golden/show_not_capable.json", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := fakeShowServer(t, tt.fixture)
			defer srv.Close()

			p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "test-model"}
			got, err := p.supportsTools(context.Background())
			if err != nil {
				t.Fatalf("supportsTools() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("supportsTools() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSupportsToolsCachedPerProcess is 3.2's other AC: the probe result is
// cached per process — here, per Provider, which is constructed once and
// lives for the process's session.
func TestSupportsToolsCachedPerProcess(t *testing.T) {
	srv, requests := fakeShowServer(t, "testdata/golden/show_tool_capable.json")
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "test-model"}

	for i := 0; i < 3; i++ {
		got, err := p.supportsTools(context.Background())
		if err != nil {
			t.Fatalf("supportsTools() call %d error = %v", i, err)
		}
		if !got {
			t.Fatalf("supportsTools() call %d = false, want true", i)
		}
	}

	if got := atomic.LoadInt32(requests); got != 1 {
		t.Errorf("/api/show was requested %d times, want 1 (result should be cached)", got)
	}
}

func TestChatFailsFastWithoutToolSupport(t *testing.T) {
	srv, requests := fakeShowServer(t, "testdata/golden/show_not_capable.json")
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "test-model", numCtx: 4096}
	req := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		Tools: []llm.ToolDefinition{
			{Name: "read_file", Description: "read a file", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	_, err := p.Chat(context.Background(), req)
	if !errors.Is(err, ErrToolsNotSupported) {
		t.Fatalf("Chat() error = %v, want wrapping ErrToolsNotSupported", err)
	}
	if got := atomic.LoadInt32(requests); got != 1 {
		t.Errorf("/api/show was requested %d times, want 1", got)
	}
}

func TestChatSkipsProbeWithoutTools(t *testing.T) {
	var showRequests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			atomic.AddInt32(&showRequests, 1)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"test-model","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "test-model", numCtx: 4096}
	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got := atomic.LoadInt32(&showRequests); got != 0 {
		t.Errorf("/api/show was requested %d times for a tool-less request, want 0", got)
	}
}
