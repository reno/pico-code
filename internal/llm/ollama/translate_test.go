package ollama

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"

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

// TestToRequestThinkSetsThinkValue and TestToRequestThinkUnsetLeavesThinkNil
// are 16.1's AC: an unset Request.Think (the default) leaves the wire
// request unchanged from before 16.1, and a set one turns Think on.
func TestToRequestThinkSetsThinkValue(t *testing.T) {
	ar, err := toRequest(llm.Request{Think: true}, "qwen3:8b", 4096)
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}
	if ar.Think == nil || ar.Think.Value != true {
		t.Errorf("Think = %+v, want a ThinkValue{Value: true}", ar.Think)
	}
}

func TestToRequestThinkUnsetLeavesThinkNil(t *testing.T) {
	ar, err := toRequest(llm.Request{}, "qwen3:8b", 4096)
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}
	if ar.Think != nil {
		t.Errorf("Think = %+v, want nil when Request.Think is unset", ar.Think)
	}
}

// TestToRequestRoundTripsThinkingBlockFromHistory is a regression test: a
// prior assistant turn's Thinking block (fromResponse prepends one ahead
// of Text when the model returns one) gets replayed back on the next
// request once it's sitting in history, same as any other block. Before
// this fix, toAssistantMessages had no case for llm.Thinking at all, so a
// second turn after a thinking-enabled reply failed outright with
// "unsupported block type llm.Thinking in assistant message".
func TestToRequestRoundTripsThinkingBlockFromHistory(t *testing.T) {
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
			{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "and 6*9?"}}},
		},
	}
	ar, err := toRequest(req, "qwen3:8b", 4096)
	if err != nil {
		t.Fatalf("toRequest() error = %v", err)
	}

	var assistant *api.Message
	for i := range ar.Messages {
		if ar.Messages[i].Role == "assistant" {
			assistant = &ar.Messages[i]
		}
	}
	if assistant == nil {
		t.Fatal("no assistant message in the translated request")
	}
	if assistant.Thinking != "7 times 8 is 56." {
		t.Errorf("assistant.Thinking = %q, want the replayed thinking text", assistant.Thinking)
	}
	if assistant.Content != "56" {
		t.Errorf("assistant.Content = %q, want %q", assistant.Content, "56")
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
