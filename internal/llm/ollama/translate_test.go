package ollama

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/llm"
)

const requestGoldenPath = "testdata/golden/request.json"

// threeTurnToolConversation mirrors the Anthropic adapter's golden fixture
// (2.1) so the two adapters' request shapes for an equivalent conversation
// are easy to compare side by side.
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
	ar, err := toRequest(threeTurnToolConversation(), "qwen3:8b", 8192)
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}

	data, err := json.Marshal(ar)
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

func TestToRequestUnsupportedRole(t *testing.T) {
	req := llm.Request{
		Messages: []llm.Message{{Role: "system", Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	}
	if _, err := toRequest(req, "qwen3:8b", 4096); err == nil {
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
	ar, err := toRequest(req, "qwen3:8b", 4096)
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}
	if len(ar.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3 (1 assistant + 2 tool results)", len(ar.Messages))
	}
	if ar.Messages[1].Role != "tool" || ar.Messages[1].ToolCallID != "call_1" {
		t.Errorf("Messages[1] = %+v, want role=tool tool_call_id=call_1", ar.Messages[1])
	}
	if ar.Messages[2].Role != "tool" || ar.Messages[2].ToolCallID != "call_2" {
		t.Errorf("Messages[2] = %+v, want role=tool tool_call_id=call_2", ar.Messages[2])
	}
}
