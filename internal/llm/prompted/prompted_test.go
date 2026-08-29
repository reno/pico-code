package prompted

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/reno/pico-code/internal/llm"
)

type stubProvider struct {
	respText   string
	respErr    error
	lastReq    llm.Request
	callsCount int
}

func (s *stubProvider) Name() string { return "stub" }

func (s *stubProvider) Chat(_ context.Context, req llm.Request) (*llm.Response, error) {
	s.lastReq = req
	s.callsCount++
	if s.respErr != nil {
		return nil, s.respErr
	}
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: s.respText}}},
		StopReason: "end_turn",
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil
}

func (s *stubProvider) Stream(_ context.Context, _ llm.Request) (<-chan llm.Event, error) {
	return nil, errors.New("stub: not implemented")
}

func testTools() []llm.ToolDefinition {
	return []llm.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get the current weather for a location.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		},
	}
}

// ignoreToolUseID treats two ToolUse blocks as equal on Name and Input
// alone, ignoring the timestamp-derived ID. For the parse-error sentinel,
// Input also isn't compared — it carries a real error message and the raw
// model text, whose exact wording isn't the contract; the tests separately
// assert its shape (non-empty "error" and "raw" fields).
var ignoreToolUseID = cmp.Comparer(func(a, b llm.ToolUse) bool {
	if a.Name != b.Name {
		return false
	}
	if a.Name == parseErrorToolName {
		return true
	}
	return string(a.Input) == string(b.Input)
})

// TestChatParsesFencedToolCall is 3.3's AC: a table over 6 realistic
// outputs (missing fence, trailing prose, single-quoted JSON, array
// instead of object, unknown tool, valid), each producing its documented
// outcome, never a panic.
func TestChatParsesFencedToolCall(t *testing.T) {
	tests := []struct {
		name     string
		respText string
		want     []llm.Block
		wantStop string
	}{
		{
			name:     "missing fence: no tool call attempted",
			respText: "The weather looks nice today, no need to check.",
			want:     []llm.Block{llm.Text{Text: "The weather looks nice today, no need to check."}},
			wantStop: "end_turn",
		},
		{
			// Regression: a real answer fenced with an unrelated language
			// label used to be silently absorbed as if it were a ```json
			// block, producing a bogus parse-error ToolUse instead of
			// passing the answer through untouched.
			name:     "non-json fence: genuine answer, not a tool call attempt",
			respText: "```plaintext\nthe quick brown zebra juggles 42 lime-green kazoos.\n```",
			want: []llm.Block{
				llm.Text{Text: "```plaintext\nthe quick brown zebra juggles 42 lime-green kazoos.\n```"},
			},
			wantStop: "end_turn",
		},
		{
			name:     "trailing prose: fence extracted, surrounding text kept",
			respText: "Let me check that.\n```json\n{\"tool\":\"get_weather\",\"input\":{\"location\":\"Paris\"}}\n```\nOne moment please.",
			want: []llm.Block{
				llm.Text{Text: "Let me check that."},
				llm.ToolUse{Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
				llm.Text{Text: "One moment please."},
			},
			wantStop: "tool_use",
		},
		{
			name:     "single-quoted JSON: malformed, sentinel ToolUse carries the error",
			respText: "```json\n{'tool': 'get_weather', 'input': {'location': 'Paris'}}\n```",
			want: []llm.Block{
				llm.ToolUse{Name: parseErrorToolName, Input: json.RawMessage(`{}`)},
			},
			wantStop: "tool_use",
		},
		{
			name:     "array instead of object: malformed, sentinel ToolUse carries the error",
			respText: "```json\n[\"get_weather\", {\"location\":\"Paris\"}]\n```",
			want: []llm.Block{
				llm.ToolUse{Name: parseErrorToolName, Input: json.RawMessage(`{}`)},
			},
			wantStop: "tool_use",
		},
		{
			name:     "unknown tool: valid shape, unfamiliar name — not this layer's concern",
			respText: "```json\n{\"tool\":\"time_travel\",\"input\":{}}\n```",
			want: []llm.Block{
				llm.ToolUse{Name: "time_travel", Input: json.RawMessage(`{}`)},
			},
			wantStop: "tool_use",
		},
		{
			name:     "valid: clean single fenced call",
			respText: "```json\n{\"tool\":\"get_weather\",\"input\":{\"location\":\"Paris\"}}\n```",
			want: []llm.Block{
				llm.ToolUse{Name: "get_weather", Input: json.RawMessage(`{"location":"Paris"}`)},
			},
			wantStop: "tool_use",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &stubProvider{respText: tt.respText}
			p := Wrap(inner)

			resp, err := p.Chat(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "what's the weather?"}}}},
				Tools:    testTools(),
			})
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}

			if diff := cmp.Diff(tt.want, resp.Message.Blocks, ignoreToolUseID, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Blocks mismatch (-want +got):\n%s", diff)
			}
			if resp.StopReason != tt.wantStop {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, tt.wantStop)
			}

			// For the malformed cases, the sentinel's input should still
			// carry enough for a human (or a future loop) to see why it
			// failed and what the model actually said.
			for _, b := range resp.Message.Blocks {
				tu, ok := b.(llm.ToolUse)
				if !ok || tu.Name != parseErrorToolName {
					continue
				}
				var payload map[string]string
				if err := json.Unmarshal(tu.Input, &payload); err != nil {
					t.Fatalf("sentinel ToolUse.Input is not valid JSON: %v", err)
				}
				if payload["error"] == "" || payload["raw"] == "" {
					t.Errorf("sentinel ToolUse.Input = %s, want non-empty error and raw", tu.Input)
				}
			}
		})
	}
}

func TestChatInjectsSchemasAndClearsNativeTools(t *testing.T) {
	inner := &stubProvider{respText: "no tool needed"}
	p := Wrap(inner)

	_, err := p.Chat(context.Background(), llm.Request{
		System:   "You are helpful.",
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		Tools:    testTools(),
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if inner.lastReq.Tools != nil {
		t.Errorf("inner request Tools = %v, want nil (native tools must not be used in prompted mode)", inner.lastReq.Tools)
	}
	if inner.lastReq.System == "You are helpful." {
		t.Error("inner request System was not augmented with tool schemas")
	}
	for _, want := range []string{"You are helpful.", "get_weather", "location"} {
		if !strings.Contains(inner.lastReq.System, want) {
			t.Errorf("inner request System = %q, want it to contain %q", inner.lastReq.System, want)
		}
	}
}

func TestChatWithoutToolsBypassesPromptedMode(t *testing.T) {
	inner := &stubProvider{respText: "```json\n{\"tool\":\"x\",\"input\":{}}\n```"}
	p := Wrap(inner)

	resp, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if inner.lastReq.System != "" {
		t.Errorf("inner request System = %q, want untouched (no tools offered)", inner.lastReq.System)
	}
	// Passed straight through: no parsing attempted since Tools was empty.
	text, ok := resp.Message.Blocks[0].(llm.Text)
	if !ok || text.Text != inner.respText {
		t.Errorf("Blocks[0] = %#v, want the raw stub text unchanged", resp.Message.Blocks[0])
	}
}

func TestChatPropagatesInnerError(t *testing.T) {
	inner := &stubProvider{respErr: errors.New("boom")}
	p := Wrap(inner)

	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
		Tools:    testTools(),
	})
	if err == nil {
		t.Fatal("Chat() error = nil, want the inner provider's error propagated")
	}
}
