package ollama

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"

	"github.com/reno/pico-code/internal/llm"
)

func decodeFixture(t *testing.T, path string) *llm.Response {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", path, err)
	}
	normalized, err := normalizeToolCallArguments(data)
	if err != nil {
		t.Fatalf("normalizeToolCallArguments(%s) error = %v", path, err)
	}

	var chatResp api.ChatResponse
	if err := json.Unmarshal(normalized, &chatResp); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", path, err)
	}
	got, err := fromResponse(llm.Request{}, &chatResp)
	if err != nil {
		t.Fatalf("fromResponse(%s) error = %v", path, err)
	}
	return got
}

// TestFromResponseStringArgumentsMatchesObjectForm is 3.1's specific AC: a
// fixture where arguments arrive as a JSON-encoded string parses
// identically to the object form.
func TestFromResponseStringArgumentsMatchesObjectForm(t *testing.T) {
	objectForm := decodeFixture(t, "testdata/golden/response_object_args.json")
	stringForm := decodeFixture(t, "testdata/golden/response_string_args.json")

	if diff := cmp.Diff(objectForm, stringForm); diff != "" {
		t.Errorf("string-form arguments produced a different Response than object-form (-object +string):\n%s", diff)
	}

	want := &llm.Response{
		Message: llm.Message{
			Role: llm.RoleAssistant,
			Blocks: []llm.Block{
				llm.Text{Text: "Let me check the weather for you."},
				llm.ToolUse{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
			},
		},
		StopReason: "stop",
		Usage:      llm.Usage{InputTokens: 512, OutputTokens: 64},
	}
	if diff := cmp.Diff(want, objectForm); diff != "" {
		t.Errorf("object-form fixture mismatch (-want +got):\n%s", diff)
	}
}

// TestFromMessageRecoversNarratedToolCall is a regression test: some
// models that report native tool support (e.g. qwen2.5-coder:7b via
// Ollama) still write the call as plain JSON text in Content instead of
// populating ToolCalls — reproduced live against real Ollama. fromMessage
// must recover a real ToolUse from that text instead of showing it as an
// ordinary reply.
func TestFromMessageRecoversNarratedToolCall(t *testing.T) {
	tools := []llm.ToolDefinition{{Name: "read_file"}}
	m := api.Message{Role: "assistant", Content: `{"name": "read_file", "arguments": {"path": "zen_of_python.txt"}}`}

	blocks, err := fromMessage(m, tools)
	if err != nil {
		t.Fatalf("fromMessage() error = %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("len(blocks) = %d, want 1", len(blocks))
	}
	tu, ok := blocks[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("blocks[0] = %T, want llm.ToolUse", blocks[0])
	}
	if tu.Name != "read_file" {
		t.Errorf("Name = %q, want %q", tu.Name, "read_file")
	}
	if string(tu.Input) != `{"path": "zen_of_python.txt"}` {
		t.Errorf("Input = %s, want the arguments object verbatim", tu.Input)
	}
}

// TestFromMessageLeavesUnknownNamedJSONAsText proves the recovery is
// scoped to tools actually offered this turn, so a model's legitimate
// "please output JSON" answer that happens to have a "name" field isn't
// misread as a call it never intended to make.
func TestFromMessageLeavesUnknownNamedJSONAsText(t *testing.T) {
	tools := []llm.ToolDefinition{{Name: "read_file"}}
	content := `{"name": "Ada Lovelace", "arguments": {"born": 1815}}`
	m := api.Message{Role: "assistant", Content: content}

	blocks, err := fromMessage(m, tools)
	if err != nil {
		t.Fatalf("fromMessage() error = %v", err)
	}
	want := []llm.Block{llm.Text{Text: content}}
	if diff := cmp.Diff(want, blocks); diff != "" {
		t.Errorf("blocks mismatch (-want +got):\n%s", diff)
	}
}

// TestFromMessageDoesNotRecoverWhenRealToolCallsPresent proves the
// recovery only kicks in when ToolCalls is genuinely empty, never as a
// second guess alongside a real structured call.
func TestFromMessageDoesNotRecoverWhenRealToolCallsPresent(t *testing.T) {
	tools := []llm.ToolDefinition{{Name: "read_file"}}
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "b.txt")
	m := api.Message{
		Role:    "assistant",
		Content: `{"name": "read_file", "arguments": {"path": "a.txt"}}`,
		ToolCalls: []api.ToolCall{
			{Function: api.ToolCallFunction{Name: "read_file", Arguments: args}},
		},
	}

	blocks, err := fromMessage(m, tools)
	if err != nil {
		t.Fatalf("fromMessage() error = %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("len(blocks) = %d, want 2 (the raw text plus the real tool call, no recovery attempted)", len(blocks))
	}
	if _, ok := blocks[0].(llm.Text); !ok {
		t.Errorf("blocks[0] = %T, want llm.Text (content passed through untouched)", blocks[0])
	}
}

func TestFromResponseSynthesizesMissingID(t *testing.T) {
	got := decodeFixture(t, "testdata/golden/response_missing_id.json")
	if len(got.Message.Blocks) != 1 {
		t.Fatalf("len(Blocks) = %d, want 1", len(got.Message.Blocks))
	}
	tu, ok := got.Message.Blocks[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("Blocks[0] = %T, want llm.ToolUse", got.Message.Blocks[0])
	}
	if tu.ID == "" {
		t.Error("ID was not synthesized: still empty")
	}
}
