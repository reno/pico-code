// Package conformance is a single test suite run against every llm.Provider
// adapter, so a behavior guaranteed by the Provider interface (a canonical
// round trip, a tool call decodes correctly, an HTTP error surfaces as a Go
// error, ctx cancellation returns promptly, streaming and non-streaming
// agree) is asserted once instead of once per adapter. Adding a third
// provider only requires building a Harness for it — this package never
// imports a provider SDK.
package conformance

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/llm"
)

// Scenario names one canned server behavior a Harness must be able to
// serve, so the suite can drive each adapter through the same situations.
type Scenario int

const (
	// ScenarioSimpleText is a plain-text, non-streaming reply.
	ScenarioSimpleText Scenario = iota
	// ScenarioSimpleTextStream is ScenarioSimpleText's streaming form.
	ScenarioSimpleTextStream
	// ScenarioToolCall is a non-streaming reply containing one tool call.
	ScenarioToolCall
	// ScenarioToolCallStream is ScenarioToolCall's streaming form.
	ScenarioToolCallStream
	// ScenarioServerError is an HTTP error response.
	ScenarioServerError
	// ScenarioHang never responds until the request's ctx is done.
	ScenarioHang
)

// Harness is what a provider package supplies to run this suite against
// its own adapter.
type Harness struct {
	// Name identifies the provider in subtest names.
	Name string
	// NewProvider returns a Provider whose requests go to baseURL.
	NewProvider func(baseURL string) llm.Provider
	// Server returns an httptest.Server serving scenario's canned
	// response; the suite closes it.
	Server func(t *testing.T, scenario Scenario) *httptest.Server
	// WantText is the text ScenarioSimpleText/ScenarioSimpleTextStream
	// decode to.
	WantText string
	// WantToolUse is the call ScenarioToolCall/ScenarioToolCallStream
	// decode to (ID is not compared).
	WantToolUse llm.ToolUse
}

// Run drives every conformance check against h.
func Run(t *testing.T, h Harness) {
	t.Run(h.Name+"_CanonicalRoundTrip", func(t *testing.T) { testCanonicalRoundTrip(t, h) })
	t.Run(h.Name+"_ToolCall", func(t *testing.T) { testToolCall(t, h) })
	t.Run(h.Name+"_ErrorMapping", func(t *testing.T) { testErrorMapping(t, h) })
	t.Run(h.Name+"_Cancellation", func(t *testing.T) { testCancellation(t, h) })
	t.Run(h.Name+"_StreamNonStreamEquivalence", func(t *testing.T) { testStreamNonStreamEquivalence(t, h) })
}

func request() llm.Request {
	return llm.Request{
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		MaxTokens: 32,
	}
}

func testCanonicalRoundTrip(t *testing.T, h Harness) {
	srv := h.Server(t, ScenarioSimpleText)
	defer srv.Close()

	resp, err := h.NewProvider(srv.URL).Chat(context.Background(), request())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("Blocks = %v, want exactly one Text block", resp.Message.Blocks)
	}
	text, ok := resp.Message.Blocks[0].(llm.Text)
	if !ok || text.Text != h.WantText {
		t.Errorf("Blocks[0] = %#v, want Text{%q}", resp.Message.Blocks[0], h.WantText)
	}
}

func testToolCall(t *testing.T, h Harness) {
	srv := h.Server(t, ScenarioToolCall)
	defer srv.Close()

	resp, err := h.NewProvider(srv.URL).Chat(context.Background(), request())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	assertToolUse(t, resp.Message.Blocks, h.WantToolUse)
}

func testErrorMapping(t *testing.T, h Harness) {
	srv := h.Server(t, ScenarioServerError)
	defer srv.Close()

	_, err := h.NewProvider(srv.URL).Chat(context.Background(), request())
	if err == nil {
		t.Fatal("Chat() error = nil, want a non-nil error for a server error response")
	}
}

func testCancellation(t *testing.T, h Harness) {
	srv := h.Server(t, ScenarioHang)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)

	start := time.Now()
	_, err := h.NewProvider(srv.URL).Chat(ctx, request())
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Chat() error = %v, want it to wrap context.Canceled", err)
	}
	if elapsed > time.Second {
		t.Fatalf("Chat() took %s after cancellation, want a prompt return", elapsed)
	}
}

func testStreamNonStreamEquivalence(t *testing.T, h Harness) {
	streamSrv := h.Server(t, ScenarioToolCallStream)
	defer streamSrv.Close()
	nonStreamSrv := h.Server(t, ScenarioToolCall)
	defer nonStreamSrv.Close()

	ch, err := h.NewProvider(streamSrv.URL).Stream(context.Background(), request())
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	streamed, err := llm.CollectStream(context.Background(), ch)
	if err != nil {
		t.Fatalf("CollectStream() error = %v", err)
	}

	nonStreamed, err := h.NewProvider(nonStreamSrv.URL).Chat(context.Background(), request())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	rawJSONSemanticEqual := cmp.Comparer(func(a, b json.RawMessage) bool {
		var va, vb any
		if json.Unmarshal(a, &va) != nil || json.Unmarshal(b, &vb) != nil {
			return false
		}
		return reflect.DeepEqual(va, vb)
	})
	if diff := cmp.Diff(nonStreamed, streamed, rawJSONSemanticEqual); diff != "" {
		t.Errorf("streamed Response differs from non-streaming equivalent (-nonStreaming +streamed):\n%s", diff)
	}
}

func assertToolUse(t *testing.T, blocks []llm.Block, want llm.ToolUse) {
	t.Helper()
	var got *llm.ToolUse
	for _, b := range blocks {
		if tu, ok := b.(llm.ToolUse); ok {
			got = &tu
		}
	}
	if got == nil {
		t.Fatalf("Blocks = %v, want a ToolUse block", blocks)
	}
	if got.Name != want.Name {
		t.Errorf("ToolUse.Name = %q, want %q", got.Name, want.Name)
	}
	var gotArgs, wantArgs any
	if err := json.Unmarshal(got.Input, &gotArgs); err != nil {
		t.Fatalf("ToolUse.Input is not valid JSON: %v", err)
	}
	if err := json.Unmarshal(want.Input, &wantArgs); err != nil {
		t.Fatalf("want.Input is not valid JSON: %v", err)
	}
	if diff := cmp.Diff(wantArgs, gotArgs); diff != "" {
		t.Errorf("ToolUse.Input mismatch (-want +got):\n%s", diff)
	}
}
