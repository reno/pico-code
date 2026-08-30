package llm

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestMessageRoundTrip(t *testing.T) {
	tests := []struct {
		name         string
		conversation []Message
	}{
		{
			name: "single text turn",
			conversation: []Message{
				{Role: RoleUser, Blocks: []Block{Text{Text: "list the files in /tmp"}}},
			},
		},
		{
			name: "assistant text plus parallel tool calls",
			conversation: []Message{
				{Role: RoleUser, Blocks: []Block{Text{Text: "read a.txt and b.txt"}}},
				{
					Role: RoleAssistant,
					Blocks: []Block{
						Text{Text: "On it."},
						ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)},
						ToolUse{ID: "call_2", Name: "read_file", Input: json.RawMessage(`{"path":"b.txt"}`)},
					},
				},
			},
		},
		{
			name: "tool results, one error",
			conversation: []Message{
				{
					Role: RoleUser,
					Blocks: []Block{
						ToolResult{ToolUseID: "call_1", Content: "contents of a.txt"},
						ToolResult{ToolUseID: "call_2", Content: "no such file", IsError: true},
					},
				},
			},
		},
		{
			name: "full conversation with all three block kinds",
			conversation: []Message{
				{Role: RoleUser, Blocks: []Block{Text{Text: "read config.json"}}},
				{
					Role: RoleAssistant,
					Blocks: []Block{
						Text{Text: "Reading it now."},
						ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"config.json"}`)},
					},
				},
				{
					Role:   RoleUser,
					Blocks: []Block{ToolResult{ToolUseID: "call_1", Content: `{"debug":true}`}},
				},
				{Role: RoleAssistant, Blocks: []Block{Text{Text: "Debug mode is enabled."}}},
			},
		},
		{
			name: "thinking ahead of text",
			conversation: []Message{
				{Role: RoleUser, Blocks: []Block{Text{Text: "what's 7*8?"}}},
				{
					Role: RoleAssistant,
					Blocks: []Block{
						Thinking{Text: "7*8 is 56."},
						Text{Text: "56."},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.conversation)
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}

			var got []Message
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}

			if diff := cmp.Diff(tt.conversation, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("round trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMessageUnmarshalUnknownBlockType(t *testing.T) {
	var m Message
	err := json.Unmarshal([]byte(`{"role":"user","blocks":[{"type":"bogus"}]}`), &m)
	if err == nil {
		t.Fatal("expected an error for an unknown block type")
	}
}

func TestMessageMarshalUnknownBlockType(t *testing.T) {
	type rogueBlock struct{ Block }
	m := Message{Role: RoleUser, Blocks: []Block{rogueBlock{}}}
	if _, err := json.Marshal(m); err == nil {
		t.Fatal("expected an error marshalling a block outside the sealed set")
	}
}
