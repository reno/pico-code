package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
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

// narratedStreamServer serves an NDJSON stream whose Content chunks join
// into text, with no structured tool_calls anywhere in the response —
// reproducing what qwen2.5-coder:7b actually sends over the wire for a
// narrated tool call.
func narratedStreamServer(t *testing.T, contentChunks ...string) *httptest.Server {
	t.Helper()
	var b strings.Builder
	for _, c := range contentChunks {
		escaped, err := jsonEscape(c)
		if err != nil {
			t.Fatalf("jsonEscape(%q) error = %v", c, err)
		}
		b.WriteString(`{"model":"qwen2.5-coder:7b","message":{"role":"assistant","content":` + escaped + `},"done":false}` + "\n")
	}
	b.WriteString(`{"model":"qwen2.5-coder:7b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}` + "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(b.String()))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func jsonEscape(s string) (string, error) {
	b, err := json.Marshal(s)
	return string(b), err
}

// TestStreamRecoversNarratedToolCallAcrossChunks is the streaming half of
// the narratedToolCall regression: the same recovery model.go/translate.go
// gives the non-streaming path must also work when the narrated JSON
// arrives split across several NDJSON chunks, exactly as real Ollama
// streaming delivers it token by token.
func TestStreamRecoversNarratedToolCallAcrossChunks(t *testing.T) {
	srv := narratedStreamServer(t,
		`{"name": "read_file", `,
		`"arguments": {"path": "a.txt"}}`,
	)

	req := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "read a.txt"}}}},
		Tools:    []llm.ToolDefinition{{Name: "read_file", Description: "read a file", InputSchema: []byte(`{"type":"object"}`)}},
	}
	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen2.5-coder:7b", numCtx: 4096}
	p.toolSupport = true
	p.toolSupportOnce.Do(func() {})

	ch, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var sawTextDelta bool
	var done *llm.ToolUseDone
	for e := range ch {
		switch v := e.(type) {
		case llm.TextDelta:
			sawTextDelta = true
		case llm.ToolUseDone:
			done = &v
		}
	}
	if sawTextDelta {
		t.Error("a TextDelta was sent for the narrated call — it should have been recovered as a ToolUse before ever reaching the renderer")
	}
	if done == nil {
		t.Fatal("no ToolUseDone event was sent — the narrated call was not recovered")
	}
	if string(done.Input) != `{"path": "a.txt"}` {
		t.Errorf("Input = %s, want the arguments object verbatim", done.Input)
	}
}

// TestStreamFlushesUnrecoverableJSONAsTextOnceStreamEnds proves the
// candidacy check is scoped to tools actually offered: content that starts
// with '{' but never resolves into a recognized call is still delivered as
// ordinary text (just at the end, once the whole response is known,
// instead of live) rather than silently dropped.
func TestStreamFlushesUnrecoverableJSONAsTextOnceStreamEnds(t *testing.T) {
	content := `{"name": "Ada Lovelace", "arguments": {"born": 1815}}`
	srv := narratedStreamServer(t, content)

	req := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "who was Ada Lovelace?"}}}},
		Tools:    []llm.ToolDefinition{{Name: "read_file", Description: "read a file", InputSchema: []byte(`{"type":"object"}`)}},
	}
	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "qwen2.5-coder:7b", numCtx: 4096}
	p.toolSupport = true
	p.toolSupportOnce.Do(func() {})

	ch, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	resp, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}
	want := []llm.Block{llm.Text{Text: content}}
	if diff := cmp.Diff(want, resp.Message.Blocks); diff != "" {
		t.Errorf("Blocks mismatch (-want +got):\n%s", diff)
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
