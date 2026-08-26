package history_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
)

func textMsg(role llm.Role, text string) llm.Message {
	return llm.Message{Role: role, Blocks: []llm.Block{llm.Text{Text: text}}}
}

func toolUseMsg(calls ...llm.ToolUse) llm.Message {
	blocks := make([]llm.Block, len(calls))
	for i, c := range calls {
		blocks[i] = c
	}
	return llm.Message{Role: llm.RoleAssistant, Blocks: blocks}
}

func toolResultMsg(results ...llm.ToolResult) llm.Message {
	blocks := make([]llm.Block, len(results))
	for i, r := range results {
		blocks[i] = r
	}
	return llm.Message{Role: llm.RoleUser, Blocks: blocks}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name     string
		messages []llm.Message
		wantErr  string
	}{
		{
			name: "valid single tool call",
			messages: []llm.Message{
				textMsg(llm.RoleUser, "read a.txt"),
				toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{}`)}),
				toolResultMsg(llm.ToolResult{ToolUseID: "call_1", Content: "hi"}),
				textMsg(llm.RoleAssistant, "done"),
			},
		},
		{
			name: "valid parallel tool calls",
			messages: []llm.Message{
				toolUseMsg(
					llm.ToolUse{ID: "call_1", Name: "read_file"},
					llm.ToolUse{ID: "call_2", Name: "read_file"},
				),
				toolResultMsg(
					llm.ToolResult{ToolUseID: "call_1", Content: "a"},
					llm.ToolResult{ToolUseID: "call_2", Content: "b", IsError: true},
				),
			},
		},
		{
			name: "missing result",
			messages: []llm.Message{
				toolUseMsg(
					llm.ToolUse{ID: "call_1", Name: "read_file"},
					llm.ToolUse{ID: "call_2", Name: "read_file"},
				),
				toolResultMsg(llm.ToolResult{ToolUseID: "call_1", Content: "hi"}),
			},
			wantErr: "missing ToolResult",
		},
		{
			name: "extra result",
			messages: []llm.Message{
				toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file"}),
				toolResultMsg(
					llm.ToolResult{ToolUseID: "call_1", Content: "hi"},
					llm.ToolResult{ToolUseID: "call_2", Content: "extra"},
				),
			},
			wantErr: "extra ToolResult",
		},
		{
			name: "mismatched id",
			messages: []llm.Message{
				toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file"}),
				toolResultMsg(llm.ToolResult{ToolUseID: "call_XXX", Content: "hi"}),
			},
			wantErr: `want ToolResult for "call_1"`,
		},
		{
			name: "wrong order",
			messages: []llm.Message{
				toolUseMsg(
					llm.ToolUse{ID: "call_1", Name: "read_file"},
					llm.ToolUse{ID: "call_2", Name: "read_file"},
				),
				toolResultMsg(
					llm.ToolResult{ToolUseID: "call_2", Content: "b"},
					llm.ToolResult{ToolUseID: "call_1", Content: "a"},
				),
			},
			wantErr: `position 0: want ToolResult for "call_1"`,
		},
		{
			name: "orphan tool result",
			messages: []llm.Message{
				toolResultMsg(llm.ToolResult{ToolUseID: "call_1", Content: "huh"}),
			},
			wantErr: "no preceding ToolUse",
		},
		{
			name: "tool call with no follow-up message",
			messages: []llm.Message{
				toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file"}),
			},
			wantErr: "no following message",
		},
		{
			name: "tool use on a user message",
			messages: []llm.Message{
				{Role: llm.RoleUser, Blocks: []llm.Block{llm.ToolUse{ID: "call_1", Name: "read_file"}}},
			},
			wantErr: `role is "user"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := history.New()
			for _, m := range tt.messages {
				h.Append(m)
			}
			err := h.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Validate() = nil, want an error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Validate() error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	h := history.New()
	h.Append(textMsg(llm.RoleUser, "read a.txt"))
	h.Append(toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)}))
	h.Append(toolResultMsg(llm.ToolResult{ToolUseID: "call_1", Content: "contents"}))
	h.Append(textMsg(llm.RoleAssistant, "done"))

	path := filepath.Join(t.TempDir(), "history.json")
	if err := h.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	loaded, err := history.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if err := loaded.Validate(); err != nil {
		t.Fatalf("Validate() after load error = %v", err)
	}

	if diff := cmp.Diff(h.Snapshot(), loaded.Snapshot()); diff != "" {
		t.Errorf("round trip mismatch (-want +got):\n%s", diff)
	}
}

func TestSaveRefusesInvalidHistory(t *testing.T) {
	h := history.New()
	h.Append(toolUseMsg(llm.ToolUse{ID: "call_1", Name: "read_file"}))

	path := filepath.Join(t.TempDir(), "history.json")
	if err := h.Save(path); err == nil {
		t.Fatal("Save() = nil, want an error for an invalid (unanswered ToolUse) history")
	}
}

func TestLoadRejectsCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.json")
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if _, err := history.Load(path); err == nil {
		t.Fatal("Load() = nil, want an error for a corrupt file")
	}
}
