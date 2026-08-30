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

func newTestWriteFileTool(t *testing.T, root string) *WriteFileTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewWriteFileTool(sandbox)
	if err != nil {
		t.Fatalf("NewWriteFileTool() error = %v", err)
	}
	return tool
}

func TestWriteFileCreatesNewFile(t *testing.T) {
	root := t.TempDir()
	tool := newTestWriteFileTool(t, root)

	input, _ := json.Marshal(WriteFileInput{Path: "new.txt", Content: "hello"})
	if _, err := tool.Run(context.Background(), input); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got, err := os.ReadFile(filepath.Join(root, "new.txt"))
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != "hello" {
		t.Errorf("file contents = %q, want %q", got, "hello")
	}
}

func TestWriteFileOverwritesExistingFile(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "existing.txt")
	if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestWriteFileTool(t, root)

	input, _ := json.Marshal(WriteFileInput{Path: "existing.txt", Content: "new"})
	if _, err := tool.Run(context.Background(), input); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != "new" {
		t.Errorf("file contents = %q, want %q", got, "new")
	}
}

func TestWriteFileRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestWriteFileTool(t, root)

	input, _ := json.Marshal(WriteFileInput{Path: "../outside.txt", Content: "x"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}

// TestWriteFileInterruptedRunLeavesOriginalUntouched is 5.4's atomicity AC:
// if ctx is cancelled after the temp file is written but before the commit
// (rename), the write is aborted and the target file is byte-identical to
// what it was before the call.
func TestWriteFileInterruptedRunLeavesOriginalUntouched(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "existing.txt")
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestWriteFileTool(t, root)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	input, _ := json.Marshal(WriteFileInput{Path: "existing.txt", Content: "interrupted write"})
	_, err := tool.Run(ctx, input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for a cancelled ctx")
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != "original" {
		t.Errorf("file contents = %q, want the original %q untouched", got, "original")
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("ReadDir() error = %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".pico-write-") {
			t.Errorf("leftover temp file %q was not cleaned up", e.Name())
		}
	}
}

func TestWriteFilePreviewShowsDiff(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "existing.txt")
	if err := os.WriteFile(path, []byte("line1\nline2\nline3"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	tool := newTestWriteFileTool(t, root)

	input, _ := json.Marshal(WriteFileInput{Path: "existing.txt", Content: "line1\nCHANGED\nline3"})
	got, err := tool.Preview(context.Background(), input)
	if err != nil {
		t.Fatalf("Preview() error = %v", err)
	}
	if !strings.Contains(got, "- line2") {
		t.Errorf("Preview() = %q, want it to show the removed line", got)
	}
	if !strings.Contains(got, "+ CHANGED") {
		t.Errorf("Preview() = %q, want it to show the added line", got)
	}
	if !strings.Contains(got, "  line1") || !strings.Contains(got, "  line3") {
		t.Errorf("Preview() = %q, want unchanged lines shown as context", got)
	}
}

func TestWriteFilePreviewForNewFile(t *testing.T) {
	root := t.TempDir()
	tool := newTestWriteFileTool(t, root)

	input, _ := json.Marshal(WriteFileInput{Path: "new.txt", Content: "hello"})
	got, err := tool.Preview(context.Background(), input)
	if err != nil {
		t.Fatalf("Preview() error = %v", err)
	}
	if !strings.Contains(got, "+ hello") {
		t.Errorf("Preview() = %q, want the new content shown as added", got)
	}
}

func TestWriteFileNeedsApproval(t *testing.T) {
	root := t.TempDir()
	tool := newTestWriteFileTool(t, root)
	if !tool.NeedsApproval() {
		t.Error("NeedsApproval() = false, want true")
	}
}
