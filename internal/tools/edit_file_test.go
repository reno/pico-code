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

func newTestEditFileTool(t *testing.T, root string) *EditFileTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewEditFileTool(sandbox)
	if err != nil {
		t.Fatalf("NewEditFileTool() error = %v", err)
	}
	return tool
}

func TestEditFileReplacesUniqueMatch(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "a.go")
	mustWriteFile(t, path, "func Foo() {}\nfunc Bar() {}\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "func Foo", NewString: "func Baz"})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(got, "replaced 1 occurrence") {
		t.Errorf("Run() = %q, want it to report 1 replacement", got)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(data) != "func Baz() {}\nfunc Bar() {}\n" {
		t.Errorf("file contents = %q, want the edit applied", data)
	}
}

func TestEditFileZeroMatchesErrors(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "package a\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "nowhere", NewString: "x"})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error when old_string isn't found")
	}
}

func TestEditFileAmbiguousMatchErrorsWithoutReplaceAll(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "x\nx\nx\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "x", NewString: "y"})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for a non-unique old_string")
	}
	if !strings.Contains(err.Error(), "matches 3 times") {
		t.Errorf("Run() error = %v, want it to report the match count", err)
	}
}

func TestEditFileReplaceAllReplacesEveryOccurrence(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "a.go")
	mustWriteFile(t, path, "x\nx\nx\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "x", NewString: "y", ReplaceAll: true})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(got, "replaced 3 occurrence") {
		t.Errorf("Run() = %q, want it to report 3 replacements", got)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(data) != "y\ny\ny\n" {
		t.Errorf("file contents = %q, want every occurrence replaced", data)
	}
}

func TestEditFileRejectsEmptyOldString(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "package a\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "", NewString: "x"})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for an empty old_string")
	}
}

func TestEditFileRejectsIdenticalStrings(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "package a\n")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "package a", NewString: "package a"})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error when old_string equals new_string")
	}
}

func TestEditFileRejectsNonexistentFile(t *testing.T) {
	root := t.TempDir()
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "missing.go", OldString: "x", NewString: "y"})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for a file that doesn't exist")
	}
}

func TestEditFileRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "../outside.go", OldString: "x", NewString: "y"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}

func TestEditFilePreviewShowsDiff(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "line1\nline2\nline3")
	tool := newTestEditFileTool(t, root)

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "line2", NewString: "CHANGED"})
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
}

func TestEditFileInterruptedRunLeavesOriginalUntouched(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "a.go")
	mustWriteFile(t, path, "original")
	tool := newTestEditFileTool(t, root)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	input, _ := json.Marshal(EditFileInput{Path: "a.go", OldString: "original", NewString: "edited"})
	_, err := tool.Run(ctx, input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for a cancelled ctx")
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != "original" {
		t.Errorf("file contents = %q, want the original untouched", got)
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

func TestEditFileNeedsApproval(t *testing.T) {
	root := t.TempDir()
	tool := newTestEditFileTool(t, root)
	if !tool.NeedsApproval() {
		t.Error("NeedsApproval() = false, want true")
	}
}
