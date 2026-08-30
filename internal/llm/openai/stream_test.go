package openai

import (
	"context"
	"errors"
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

// TestStreamMatchesNonStreamingEquivalent is 12.2's assembly half: a
// recorded SSE fixture that splits one tool call's arguments across
// several per-index chunks (including a split mid-object, between a key
// and its value) produces the exact same Response as the non-streaming
// path for the same content.
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
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "what's the weather in Paris?"}}}},
		MaxTokens: 64,
	}

	streamP := &Provider{httpClient: http.DefaultClient, baseURL: streamSrv.URL, model: "gpt-4o-mini"}
	ch, err := streamP.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	streamed, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	nonStreamP := &Provider{httpClient: http.DefaultClient, baseURL: nonStreamSrv.URL, model: "gpt-4o-mini"}
	nonStreamed, err := nonStreamP.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if diff := cmp.Diff(nonStreamed, streamed); diff != "" {
		t.Errorf("streamed Response differs from non-streaming equivalent (-nonStreaming +streamed):\n%s", diff)
	}

	// Pin the specific split-argument reconstruction directly: this is
	// what actually exercises the per-index chunk accumulation, since the
	// cmp.Diff above only proves the end result matches.
	toolUse, ok := streamed.Message.Blocks[1].(llm.ToolUse)
	if !ok {
		t.Fatalf("Blocks[1] = %T, want llm.ToolUse", streamed.Message.Blocks[1])
	}
	wantInput := `{"location":"Paris","unit":"celsius"}`
	if string(toolUse.Input) != wantInput {
		t.Errorf("reassembled tool input = %s, want the exact accumulated bytes %s", toolUse.Input, wantInput)
	}
}

// TestStreamInterleavedToolCallsAccumulatePerIndex proves argument
// fragments for two parallel tool calls, whose chunks arrive interleaved
// rather than one call fully at a time, are kept separate by Index and
// reassembled correctly — the scenario "per-index accumulation" exists to
// handle.
func TestStreamInterleavedToolCallsAccumulatePerIndex(t *testing.T) {
	srv := sseServer(
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "id": "call_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": ""}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 1, "id": "call_2", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": ""}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "function": map[string]any{"arguments": `{"location":`}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 1, "function": map[string]any{"arguments": `{"location":`}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "function": map[string]any{"arguments": `"Paris"}`}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 1, "function": map[string]any{"arguments": `"Tokyo"}`}},
		}}, "finish_reason": nil}}}),
		sseData(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}}}),
		"data: [DONE]\n\n",
	)
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "gpt-4o-mini"}
	ch, err := p.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "weather in Paris and Tokyo?"}}}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	resp, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("len(Blocks) = %d, want 2", len(resp.Message.Blocks))
	}
	first, ok := resp.Message.Blocks[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("Blocks[0] = %T, want llm.ToolUse", resp.Message.Blocks[0])
	}
	second, ok := resp.Message.Blocks[1].(llm.ToolUse)
	if !ok {
		t.Fatalf("Blocks[1] = %T, want llm.ToolUse", resp.Message.Blocks[1])
	}
	if first.ID != "call_1" || string(first.Input) != `{"location":"Paris"}` {
		t.Errorf("Blocks[0] = %+v, want call_1 with input {\"location\":\"Paris\"}", first)
	}
	if second.ID != "call_2" || string(second.Input) != `{"location":"Tokyo"}` {
		t.Errorf("Blocks[1] = %+v, want call_2 with input {\"location\":\"Tokyo\"}", second)
	}
}

func TestStreamCancelledContextStopsPromptly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "gpt-4o-mini"}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}}})
	if err == nil {
		t.Fatal("Stream() error = nil, want an error for a cancelled ctx")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Stream() error = %v, want it to wrap context.Canceled", err)
	}
}

func TestStreamMalformedChunkSendsError(t *testing.T) {
	srv := sseServer("data: not json\n\n")
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "gpt-4o-mini"}
	ch, err := p.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if _, err := llm.CollectStream(context.Background(), ch); err == nil {
		t.Fatal("CollectStream() error = nil, want an error for a malformed chunk")
	}
}
