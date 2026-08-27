package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestRunCommandRejectsBlockedBinary(t *testing.T) {
	tool, err := NewRunCommandTool([]string{"echo"}, time.Second)
	if err != nil {
		t.Fatalf("NewRunCommandTool() error = %v", err)
	}
	input, _ := json.Marshal(RunCommandInput{Command: "rm", Args: []string{"-rf", "/"}})

	_, err = tool.Run(context.Background(), input)
	if !errors.Is(err, ErrCommandNotAllowed) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrCommandNotAllowed)
	}
}

func TestRunCommandRejectsShellMetacharacters(t *testing.T) {
	tool, err := NewRunCommandTool([]string{"echo"}, time.Second)
	if err != nil {
		t.Fatalf("NewRunCommandTool() error = %v", err)
	}
	input, _ := json.Marshal(RunCommandInput{Command: "echo", Args: []string{"$(whoami)"}})

	_, err = tool.Run(context.Background(), input)
	if !errors.Is(err, ErrShellMetacharacter) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrShellMetacharacter)
	}
}

func TestRunCommandReportsExitCode(t *testing.T) {
	tool, err := NewRunCommandTool([]string{"sh"}, time.Second)
	if err != nil {
		t.Fatalf("NewRunCommandTool() error = %v", err)
	}
	input, _ := json.Marshal(RunCommandInput{Command: "sh", Args: []string{"-c", "exit 3"}})

	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v, want a nonzero exit reported as data, not a tool error", err)
	}
	if !strings.Contains(got, "exit code: 3") {
		t.Errorf("Run() = %q, want it to report exit code 3", got)
	}
}

func TestRunCommandTruncatesOutput(t *testing.T) {
	// No shell means no pipes either, so this generates 100000 bytes via a
	// single allowlisted binary and its flags rather than "yes | head".
	tool, err := NewRunCommandTool([]string{"head"}, 5*time.Second)
	if err != nil {
		t.Fatalf("NewRunCommandTool() error = %v", err)
	}
	input, _ := json.Marshal(RunCommandInput{Command: "head", Args: []string{"-c", "100000", "/dev/zero"}})

	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(got) > defaultByteBudget+64 {
		t.Errorf("Run() output length = %d, want roughly within the %d byte budget", len(got), defaultByteBudget)
	}
	if !strings.Contains(got, "elided") {
		t.Errorf("Run() = %q, want an elision marker", got[:200])
	}
}

// TestRunCommandTimeoutKillsTheProcess proves the timeout actually
// terminates a hung child rather than Run() blocking until it exits
// naturally. Setpgid+Cancel (see run_command.go) target the negative PID
// specifically so descendants die too, but exercising an actual
// grandchild would need a non-shell helper binary to spawn one — this
// verifies the direct child dies promptly instead.
func TestRunCommandTimeoutKillsTheProcess(t *testing.T) {
	tool, err := NewRunCommandTool([]string{"sleep"}, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("NewRunCommandTool() error = %v", err)
	}
	input, _ := json.Marshal(RunCommandInput{Command: "sleep", Args: []string{"30"}})

	start := time.Now()
	_, err = tool.Run(context.Background(), input)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("Run() error = nil, want a timeout error")
	}
	if elapsed > 2*time.Second {
		t.Errorf("Run() took %s, want it to return promptly once the 50ms timeout fires, not wait out the 30s sleep", elapsed)
	}
}
