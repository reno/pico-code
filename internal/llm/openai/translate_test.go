package openai

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/llm"
)

const requestGoldenPath = "testdata/golden/request.json"

// threeTurnToolConversation mirrors the Anthropic and Ollama adapters' own
// golden fixture (2.1, 3.1) so all three adapters' request shapes for an
// equivalent conversation are easy to compare side by side.
func threeTurnToolConversation() llm.Request {
	return llm.Request{
		System: "You are a helpful weather assistant.",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "What's the weather in Paris?"}}},
			{
				Role: llm.RoleAssistant,
				Blocks: []llm.Block{
					llm.Text{Text: "Let me check."},
					llm.ToolUse{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
				},
			},
			{
				Role:   llm.RoleUser,
				Blocks: []llm.Block{llm.ToolResult{ToolUseID: "call_1", Content: "15C, cloudy"}},
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

func TestToRequestGolden(t *testing.T) {
	cr, err := toRequest(threeTurnToolConversation(), "gpt-4o-mini")
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}

	data, err := json.Marshal(cr)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
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

// TestToRequestFoldsThinkingBlockIntoText is a regression test: a Thinking
// block reaching this adapter from history built against a different
// provider (this one never produces one itself, 16.1) must not error the
// whole turn — it folds into plain text instead.
func TestToRequestFoldsThinkingBlockIntoText(t *testing.T) {
	req := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "what's 7*8?"}}},
			{
				Role: llm.RoleAssistant,
				Blocks: []llm.Block{
					llm.Thinking{Text: "7 times 8 is 56."},
					llm.Text{Text: "56"},
				},
			},
		},
	}
	if _, err := toRequest(req, "gpt-4o-mini"); err != nil {
		t.Fatalf("toRequest() error = %v, want the Thinking block folded into text instead of erroring", err)
	}
}

func TestToRequestUnsupportedRole(t *testing.T) {
	req := llm.Request{
		Messages: []llm.Message{{Role: "system", Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	}
	if _, err := toRequest(req, "gpt-4o-mini"); err == nil {
		t.Fatal("toRequest() error = nil, want an error for an unsupported role")
	}
}

func TestToRequestParallelToolResultsExpandToMultipleMessages(t *testing.T) {
	req := llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Blocks: []llm.Block{
					llm.ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{}`)},
					llm.ToolUse{ID: "call_2", Name: "read_file", Input: json.RawMessage(`{}`)},
				},
			},
			{
				Role: llm.RoleUser,
				Blocks: []llm.Block{
					llm.ToolResult{ToolUseID: "call_1", Content: "a"},
					llm.ToolResult{ToolUseID: "call_2", Content: "b", IsError: true},
				},
			},
		},
	}
	cr, err := toRequest(req, "gpt-4o-mini")
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}
	if len(cr.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3 (1 assistant + 2 tool results)", len(cr.Messages))
	}
	if cr.Messages[1].Role != "tool" || cr.Messages[1].ToolCallID != "call_1" {
		t.Errorf("Messages[1] = %+v, want role=tool tool_call_id=call_1", cr.Messages[1])
	}
	if cr.Messages[2].Role != "tool" || cr.Messages[2].ToolCallID != "call_2" {
		t.Errorf("Messages[2] = %+v, want role=tool tool_call_id=call_2", cr.Messages[2])
	}
}

func TestToRequestToolParametersPassThroughUnchanged(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`)
	req := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		Tools:    []llm.ToolDefinition{{Name: "get_weather", Description: "get weather", InputSchema: schema}},
	}
	cr, err := toRequest(req, "gpt-4o-mini")
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}
	if len(cr.Tools) != 1 {
		t.Fatalf("len(Tools) = %d, want 1", len(cr.Tools))
	}
	if diff := cmp.Diff(string(schema), string(cr.Tools[0].Function.Parameters)); diff != "" {
		t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
	}
}

func decodeFixture(t *testing.T, path string) *llm.Response {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", path, err)
	}
	var cresp chatResponse
	if err := json.Unmarshal(data, &cresp); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", path, err)
	}
	got, err := fromResponse(&cresp)
	if err != nil {
		t.Fatalf("fromResponse(%s) error = %v", path, err)
	}
	return got
}

// TestFromResponseStringArgumentsMatchesObjectForm is 12.1's specific AC: a
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
		StopReason: "tool_calls",
		Usage:      llm.Usage{InputTokens: 512, OutputTokens: 64},
	}
	if diff := cmp.Diff(want, objectForm); diff != "" {
		t.Errorf("object-form fixture mismatch (-want +got):\n%s", diff)
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

func TestFromResponseNoChoicesErrors(t *testing.T) {
	if _, err := fromResponse(&chatResponse{}); err == nil {
		t.Fatal("fromResponse() error = nil, want an error for zero choices")
	}
}

func TestDecodeArgumentsEmptyString(t *testing.T) {
	got, err := decodeArguments(json.RawMessage(`""`))
	if err != nil {
		t.Fatalf("decodeArguments() error = %v", err)
	}
	if string(got) != "{}" {
		t.Errorf("decodeArguments(empty string) = %s, want {}", got)
	}
}

func TestDecodeArgumentsInvalidStringErrors(t *testing.T) {
	if _, err := decodeArguments(json.RawMessage(`"not json"`)); err == nil {
		t.Fatal("decodeArguments() error = nil, want an error for an invalid JSON string")
	}
}
