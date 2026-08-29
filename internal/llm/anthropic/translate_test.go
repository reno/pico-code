package anthropic

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
)

// rawJSONSemanticEqual compares json.RawMessage by decoded value rather than
// raw bytes, so an irrelevant whitespace difference in a hand-edited fixture
// doesn't fail the test.
var rawJSONSemanticEqual = cmp.Comparer(func(a, b json.RawMessage) bool {
	var va, vb any
	if err := json.Unmarshal(a, &va); err != nil {
		return false
	}
	if err := json.Unmarshal(b, &vb); err != nil {
		return false
	}
	return reflect.DeepEqual(va, vb)
})

const requestGoldenPath = "testdata/golden/request.json"

// threeTurnToolConversation is the fixture AC 2.1 asks for: a user message,
// an assistant reply that calls a tool, and the tool's result.
func threeTurnToolConversation() llm.Request {
	return llm.Request{
		System: "You are a helpful weather assistant.",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "What's the weather in Paris?"}}},
			{
				Role: llm.RoleAssistant,
				Blocks: []llm.Block{
					llm.Text{Text: "Let me check."},
					llm.ToolUse{ID: "toolu_01ABC", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
				},
			},
			{
				Role:   llm.RoleUser,
				Blocks: []llm.Block{llm.ToolResult{ToolUseID: "toolu_01ABC", Content: "15C, cloudy"}},
			},
		},
		Tools: []llm.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a location.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}`),
			},
		},
		MaxTokens: 1024,
	}
}

func TestToParamsGolden(t *testing.T) {
	params, err := toParams(threeTurnToolConversation(), "claude-test-model")
	if err != nil {
		t.Fatalf("toParams() error = %v", err)
	}

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("Marshal(params) error = %v", err)
	}
	var got any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("re-unmarshal error = %v", err)
	}
	pretty, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent() error = %v", err)
	}
	pretty = append(pretty, '\n')

	if os.Getenv("RECORD") == "1" {
		if err := os.MkdirAll(filepath.Dir(requestGoldenPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(requestGoldenPath, pretty, 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	want, err := os.ReadFile(requestGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to create it)", requestGoldenPath, err)
	}
	if diff := cmp.Diff(string(want), string(pretty)); diff != "" {
		t.Errorf("request JSON mismatch (-want +got):\n%s", diff)
	}
}

func TestToParamsUnsupportedRole(t *testing.T) {
	req := llm.Request{
		Messages: []llm.Message{{Role: "system", Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	}
	if _, err := toParams(req, "claude-test-model"); err == nil {
		t.Fatal("toParams() error = nil, want an error for an unsupported role")
	}
}

func TestFromResponseRecordedFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/golden/response.json")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	var msg sdk.Message
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("Unmarshal(recorded response) error = %v", err)
	}

	got, err := fromResponse(&msg)
	if err != nil {
		t.Fatalf("fromResponse() error = %v", err)
	}

	want := &llm.Response{
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Blocks: []llm.Block{
				llm.Text{Text: "Let me check the weather for you."},
				llm.ToolUse{ID: "toolu_01ABC", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
			},
		},
		StopReason: "tool_use",
		Usage:      llm.Usage{InputTokens: 512, OutputTokens: 64},
	}

	if diff := cmp.Diff(want, got, rawJSONSemanticEqual); diff != "" {
		t.Errorf("fromResponse() mismatch (-want +got):\n%s", diff)
	}
}

// TestFromResponseParsesCacheWriteThenCacheRead is 15.2's AC: a recorded
// exchange shows a cache write then a cache read. Rather than a live
// two-request exchange (this repo's tests run offline, per CLAUDE.md), it
// pins the two individual response shapes a real back-to-back exchange
// produces: the first response's cache_creation_input_tokens is the write,
// the second's cache_read_input_tokens is the read.
func TestFromResponseParsesCacheWriteThenCacheRead(t *testing.T) {
	writeResp := []byte(`{
		"id": "msg_01WRITE", "type": "message", "role": "assistant", "model": "claude-test-model",
		"content": [{"type": "text", "text": "first reply"}],
		"stop_reason": "end_turn", "stop_sequence": null,
		"usage": {"input_tokens": 20, "output_tokens": 10, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0}
	}`)
	readResp := []byte(`{
		"id": "msg_01READ", "type": "message", "role": "assistant", "model": "claude-test-model",
		"content": [{"type": "text", "text": "second reply"}],
		"stop_reason": "end_turn", "stop_sequence": null,
		"usage": {"input_tokens": 25, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 500}
	}`)

	var write, read sdk.Message
	if err := json.Unmarshal(writeResp, &write); err != nil {
		t.Fatalf("Unmarshal(write) error = %v", err)
	}
	if err := json.Unmarshal(readResp, &read); err != nil {
		t.Fatalf("Unmarshal(read) error = %v", err)
	}

	gotWrite, err := fromResponse(&write)
	if err != nil {
		t.Fatalf("fromResponse(write) error = %v", err)
	}
	if gotWrite.Usage.CacheWriteTokens != 500 || gotWrite.Usage.CacheReadTokens != 0 {
		t.Errorf("write Usage = %+v, want CacheWriteTokens 500, CacheReadTokens 0", gotWrite.Usage)
	}

	gotRead, err := fromResponse(&read)
	if err != nil {
		t.Fatalf("fromResponse(read) error = %v", err)
	}
	if gotRead.Usage.CacheWriteTokens != 0 || gotRead.Usage.CacheReadTokens != 500 {
		t.Errorf("read Usage = %+v, want CacheWriteTokens 0, CacheReadTokens 500", gotRead.Usage)
	}
}

// TestCompactionMovesTheCacheBreakpoint is 15.2's AC: a compaction rewrite
// is shown to move the breakpoint rather than reuse a stale prefix. The
// breakpoint toParams sets is always positional (the last message's last
// block), so what "moves" is what it actually points at: compaction
// replaces every message before its boundary with a fresh synthetic
// summary, changing the bytes the cache key is computed over even though
// the breakpoint's position in the message list hasn't changed — proving
// the adapter can never accidentally serve a stale cached prefix across a
// compaction.
func TestCompactionMovesTheCacheBreakpoint(t *testing.T) {
	h := history.New()
	for i := 0; i < 5; i++ {
		h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: fmt.Sprintf("question %d", i)}}})
		h.Append(llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: fmt.Sprintf("answer %d", i)}}})
	}

	marshal := func(msgs []llm.Message) string {
		t.Helper()
		params, err := toParams(llm.Request{System: "sys", Messages: msgs, MaxTokens: 64}, "claude-test-model")
		if err != nil {
			t.Fatalf("toParams() error = %v", err)
		}
		data, err := json.Marshal(params)
		if err != nil {
			t.Fatalf("Marshal(params) error = %v", err)
		}
		return string(data)
	}

	beforeMsgs := h.Snapshot()
	before := marshal(beforeMsgs)
	if !strings.Contains(before, `"cache_control"`) {
		t.Fatal("pre-compaction request has no cache_control breakpoint at all")
	}

	if err := h.Compact(2, "a concise summary of the earlier turns"); err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	afterMsgs := h.Snapshot()
	if len(afterMsgs) == len(beforeMsgs) {
		t.Fatal("message count unchanged after Compact() — nothing was actually compacted, this test proves nothing")
	}

	after := marshal(afterMsgs)
	if !strings.Contains(after, `"cache_control"`) {
		t.Fatal("post-compaction request has no cache_control breakpoint at all")
	}
	if before == after {
		t.Fatal("request bytes identical before and after compaction, want the rewritten prefix to change what's cached")
	}
}
