package tools

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestReadFileTool(t *testing.T, root string) *ReadFileTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewReadFileTool(sandbox)
	if err != nil {
		t.Fatalf("NewReadFileTool() error = %v", err)
	}
	return tool
}

func TestReadFileReturnsFullContents(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("line1\nline2\nline3"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestReadFileTool(t, root)

	input, _ := json.Marshal(ReadFileInput{Path: "a.txt"})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "line1\nline2\nline3" {
		t.Errorf("Run() = %q, want full file contents", got)
	}
}

func TestReadFileLineRange(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("line1\nline2\nline3\nline4"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestReadFileTool(t, root)

	input, _ := json.Marshal(ReadFileInput{Path: "a.txt", StartLine: 2, EndLine: 3})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got != "line2\nline3" {
		t.Errorf("Run() = %q, want %q", got, "line2\nline3")
	}
}

func TestReadFileTruncatesLargeFile(t *testing.T) {
	root := t.TempDir()
	big := strings.Repeat("X", 10*1024*1024)
	if err := os.WriteFile(filepath.Join(root, "big.txt"), []byte(big), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestReadFileTool(t, root)

	input, _ := json.Marshal(ReadFileInput{Path: "big.txt"})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(got) > defaultByteBudget {
		t.Errorf("Run() returned %d bytes, want <= %d", len(got), defaultByteBudget)
	}
	if !strings.Contains(got, "elided") {
		t.Error("Run() output missing elision marker")
	}
}

func TestReadFileRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestReadFileTool(t, root)

	input, _ := json.Marshal(ReadFileInput{Path: "../outside.txt"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}
