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
