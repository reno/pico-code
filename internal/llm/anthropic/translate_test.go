package anthropic

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"

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
