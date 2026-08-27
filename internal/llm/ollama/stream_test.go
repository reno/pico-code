package ollama

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
	streamGoldenPath      = "testdata/golden/stream_response.ndjson"
	streamEquivGoldenPath = "testdata/golden/stream_response_equivalent.json"
)

// TestStreamMatchesNonStreamingEquivalent is 6.3's assembly half: the same
// conversation replayed through NDJSON streaming (including one line whose
// tool call arguments arrive as a JSON-encoded string, the normalization
// quirk 3.1 already handles for the non-streaming path) produces the exact
// same Response as the non-streaming path.
func TestStreamMatchesNonStreamingEquivalent(t *testing.T) {
	ndjson, err := os.ReadFile(streamGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", streamGoldenPath, err)
	}
	streamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(ndjson)
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
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "what's the weather in Paris?"}}}},
		Tools: []llm.ToolDefinition{
			{Name: "get_weather", Description: "get the weather", InputSchema: []byte(`{"type":"object"}`)},
		},
	}

	streamP := &Provider{httpClient: http.DefaultClient, baseURL: streamSrv.URL, model: "qwen3:8b", numCtx: 4096}
	streamP.toolSupport = true
	streamP.toolSupportOnce.Do(func() {})
	ch, err := streamP.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	streamed, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	nonStreamP := &Provider{httpClient: http.DefaultClient, baseURL: nonStreamSrv.URL, model: "qwen3:8b", numCtx: 4096}
	nonStreamP.toolSupport = true
	nonStreamP.toolSupportOnce.Do(func() {})
	nonStreamed, err := nonStreamP.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if diff := cmp.Diff(nonStreamed, streamed); diff != "" {
		t.Errorf("streamed Response differs from non-streaming equivalent (-nonStreaming +streamed):\n%s", diff)
	}
}

func TestStreamCancelledContextStopsPromptly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen3:8b", numCtx: 4096}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}}})
	if err == nil {
		t.Fatal("Stream() error = nil, want an error for a cancelled ctx")
	}
}
