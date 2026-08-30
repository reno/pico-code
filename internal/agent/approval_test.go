package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestAutoApproveAlwaysApproves(t *testing.T) {
	ok, err := AutoApprove.Approve(context.Background(), "any_tool", json.RawMessage(`{}`), "")
	if err != nil {
		t.Fatalf("Approve() error = %v", err)
	}
	if !ok {
		t.Error("AutoApprove.Approve() = false, want true")
	}
}

func TestConsoleApprover(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"yes", "y\n", true},
		{"full yes", "yes\n", true},
		{"no", "n\n", false},
		{"empty defaults to no", "\n", false},
		{"garbage defaults to no", "sure\n", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			c := ConsoleApprover{In: strings.NewReader(tt.input), Out: out}

			got, err := c.Approve(context.Background(), "run_command", json.RawMessage(`{"command":"echo"}`), "preview text")
			if err != nil {
				t.Fatalf("Approve() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("Approve() = %v, want %v", got, tt.want)
			}
			if !strings.Contains(out.String(), "run_command") || !strings.Contains(out.String(), "preview text") {
				t.Errorf("prompt output = %q, want it to mention the tool name and preview", out.String())
			}
		})
	}
}
