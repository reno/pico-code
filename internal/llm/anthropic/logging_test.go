package anthropic

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func withDefaultLogger(t *testing.T, level slog.Level) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: level})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

// TestDebugLoggingDumpsExchangeWithoutCredentials is 9.2's AC: debug output
// of a full turn contains no credentials.
func TestDebugLoggingDumpsExchangeWithoutCredentials(t *testing.T) {
	buf := withDefaultLogger(t, slog.LevelDebug)

	const apiKey = "sk-ant-super-secret-test-key"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okMessageBody))
	}))
	defer srv.Close()

	p := &Provider{
		client: sdk.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(srv.URL)),
		model:  "claude-test-model",
		apiKey: apiKey,
	}
	if _, err := p.Chat(context.Background(), testChatRequest()); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "anthropic: request") || !strings.Contains(out, "anthropic: response") {
		t.Fatalf("debug output missing request/response dump; got:\n%s", out)
	}
	if strings.Contains(out, apiKey) {
		t.Errorf("debug output contains the API key:\n%s", out)
	}
}

// TestDefaultLevelDoesNotDumpRequestsOrResponses is 9.2's other AC: default
// level is silent on the happy path — no request/response body ever
// reaches the log outside --log-level=debug.
func TestDefaultLevelDoesNotDumpRequestsOrResponses(t *testing.T) {
	buf := withDefaultLogger(t, slog.LevelInfo)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okMessageBody))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	if _, err := p.Chat(context.Background(), testChatRequest()); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if out := buf.String(); strings.Contains(out, "anthropic: request") || strings.Contains(out, "anthropic: response") {
		t.Errorf("default (info) level logged a request/response dump, want none:\n%s", out)
	}
}
