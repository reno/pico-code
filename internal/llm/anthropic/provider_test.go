package anthropic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/reno/pico-code/internal/llm"
)

func testProvider(baseURL string) *Provider {
	return &Provider{
		client: sdk.NewClient(option.WithAPIKey("test-key"), option.WithBaseURL(baseURL)),
		model:  "claude-test-model",
	}
}

func testChatRequest() llm.Request {
	return llm.Request{
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		MaxTokens: 16,
	}
}

const rateLimitBody = `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`

const okMessageBody = `{
	"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test-model",
	"content": [{"type": "text", "text": "ok"}],
	"stop_reason": "end_turn",
	"usage": {"input_tokens": 1, "output_tokens": 1}
}`

// TestChatRetriesOn429ThenSucceeds is CLAUDE.md 2.2's AC: a server
// returning 429 twice then 200 succeeds after two retries. The SDK's
// default MaxRetries is 2, so this exercises its retry loop rather than
// one this adapter writes itself.
func TestChatRetriesOn429ThenSucceeds(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		w.Header().Set("Content-Type", "application/json")
		if n <= 2 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(rateLimitBody))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(okMessageBody))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	resp, err := p.Chat(context.Background(), testChatRequest())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Fatalf("attempts = %d, want 3 (initial request + 2 retries)", got)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "end_turn")
	}
}

// TestChatCancelledContextReturnsPromptly is CLAUDE.md 2.2's AC: with a
// cancelled ctx, Chat returns promptly with ctx.Err() rather than waiting
// out the retry schedule. The server always 429s, which would normally
// trigger the SDK's exponential backoff (hundreds of ms); cancelling ctx
// shortly after the first attempt should interrupt that wait.
func TestChatCancelledContextReturnsPromptly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(rateLimitBody))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)

	start := time.Now()
	_, err := p.Chat(ctx, testChatRequest())
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Chat() error = %v, want it to wrap context.Canceled", err)
	}
	if elapsed > 300*time.Millisecond {
		t.Fatalf("Chat() took %s after cancellation, want a prompt return (well under the ~500ms first retry delay)", elapsed)
	}
}

func TestChatMapsUnauthorized(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	_, err := p.Chat(context.Background(), testChatRequest())
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("Chat() error = %v, want wrapping ErrUnauthorized", err)
	}
}
