package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/llm"
)

const (
	streamGoldenPath      = "testdata/golden/stream_response.sse"
	streamEquivGoldenPath = "testdata/golden/stream_response_equivalent.json"
)

// TestStreamMatchesNonStreamingEquivalent is 6.2's AC: replaying a recorded
// SSE fixture that splits a tool argument across 7 chunks — including one
// split between the "\u00" and "e9" halves of a single \uXXXX escape, i.e.
// mid-character — yields the exact same Message as the non-streaming path
// for the same content.
func TestStreamMatchesNonStreamingEquivalent(t *testing.T) {
	sse, err := os.ReadFile(streamGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", streamGoldenPath, err)
	}
	streamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(sse)
	}))
	defer streamSrv.Close()

	nonStreamBody, err := os.ReadFile(streamEquivGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", streamEquivGoldenPath, err)
	}
	nonStreamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(nonStreamBody)
	}))
	defer nonStreamSrv.Close()

	req := llm.Request{
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "what's the weather in the café district?"}}}},
		MaxTokens: 64,
	}

	streamP := testProvider(streamSrv.URL)
	ch, err := streamP.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	streamed, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	nonStreamP := testProvider(nonStreamSrv.URL)
	nonStreamed, err := nonStreamP.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if diff := cmp.Diff(nonStreamed, streamed, rawJSONSemanticEqual); diff != "" {
		t.Errorf("streamed Response differs from non-streaming equivalent (-nonStreaming +streamed):\n%s", diff)
	}

	// Pin the specific split-argument reconstruction directly: this is
	// what actually exercises the mid-\uXXXX-escape chunk boundary, since
	// the cmp.Diff above only proves semantic equality after decoding.
	toolUse, ok := streamed.Message.Blocks[1].(llm.ToolUse)
	if !ok {
		t.Fatalf("Blocks[1] = %T, want llm.ToolUse", streamed.Message.Blocks[1])
	}
	wantInput := `{"location":"caf\u00e9"}`
	if string(toolUse.Input) != wantInput {
		t.Errorf("reassembled tool input = %s, want the exact accumulated bytes %s", toolUse.Input, wantInput)
	}
}

// TestStreamCancelledContextStopsPromptly proves ctx cancellation reaches
// an in-flight stream (CLAUDE.md invariant 6), reusing the same
// slow-429-forever server pattern as the non-streaming cancellation test.
func TestStreamCancelledContextStopsPromptly(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()
	defer close(block)

	p := testProvider(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ch, err := p.Stream(ctx, testChatRequest())
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, err = llm.CollectStream(context.Background(), ch)
	if err == nil {
		t.Fatal("CollectStream() error = nil, want an error for a cancelled stream")
	}
}

// cacheUsageSSE is a minimal message_start/content_block/message_delta/
// message_stop stream carrying cache fields on message_start, standing in
// for the first request of a cached exchange (a cache write).
const cacheUsageSSE = `event: message_start
data: {"message":{"content":[],"id":"msg_cache_1","model":"claude-test-model","role":"assistant","stop_reason":null,"stop_sequence":null,"type":"message","usage":{"input_tokens":20,"output_tokens":1,"cache_creation_input_tokens":500,"cache_read_input_tokens":0}},"type":"message_start"}

event: content_block_start
data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}

event: content_block_delta
data: {"delta":{"text":"ok","type":"text_delta"},"index":0,"type":"content_block_delta"}

event: content_block_stop
data: {"index":0,"type":"content_block_stop"}

event: message_delta
data: {"delta":{"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

// TestStreamAccumulatorParsesCacheWriteUsage is half of 15.2's AC (the
// streaming path's share of "a recorded exchange shows a cache write");
// TestFromResponseParsesCacheWriteThenCacheRead in translate_test.go covers
// the non-streaming path for both write and read.
func TestStreamAccumulatorParsesCacheWriteUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(cacheUsageSSE))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	ch, err := p.Stream(context.Background(), testChatRequest())
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	resp, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}
	if resp.Usage.CacheWriteTokens != 500 || resp.Usage.CacheReadTokens != 0 {
		t.Errorf("Usage = %+v, want CacheWriteTokens 500, CacheReadTokens 0", resp.Usage)
	}
}
