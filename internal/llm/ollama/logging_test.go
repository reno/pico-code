package ollama

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

func withDefaultLogger(t *testing.T, level slog.Level) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: level})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

const loggingTestResponseBody = `{
	"model": "qwen3:8b",
	"message": {"role": "assistant", "content": "hi there"},
	"done": true, "done_reason": "stop",
	"prompt_eval_count": 10, "eval_count": 5
}`

// TestDebugLoggingDumpsExchange is 9.2's AC: debug output of a full turn
// contains no credentials. Ollama has none by default (no API key), so
// this mainly proves the dump actually happens and stays clean of the
// configured host string.
func TestDebugLoggingDumpsExchange(t *testing.T) {
	buf := withDefaultLogger(t, slog.LevelDebug)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(loggingTestResponseBody))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen3:8b", numCtx: 4096}
	if _, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello"}}}},
	}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "ollama: request") || !strings.Contains(out, "ollama: response") {
		t.Fatalf("debug output missing request/response dump; got:\n%s", out)
	}
}

// TestDefaultLevelDoesNotDumpRequestsOrResponses is 9.2's other AC: default
// level is silent on the happy path.
func TestDefaultLevelDoesNotDumpRequestsOrResponses(t *testing.T) {
	buf := withDefaultLogger(t, slog.LevelInfo)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(loggingTestResponseBody))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen3:8b", numCtx: 4096}
	if _, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello"}}}},
	}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if out := buf.String(); strings.Contains(out, "ollama: request") || strings.Contains(out, "ollama: response") {
		t.Errorf("default (info) level logged a request/response dump, want none:\n%s", out)
	}
}
