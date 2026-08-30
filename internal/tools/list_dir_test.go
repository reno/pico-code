package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestListDirTool(t *testing.T, root string) *ListDirTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewListDirTool(sandbox)
	if err != nil {
		t.Fatalf("NewListDirTool() error = %v", err)
	}
	return tool
}

func TestListDirRespectsMaxDepth(t *testing.T) {
	root := t.TempDir()
	mustMkdirAll(t, filepath.Join(root, "a", "b"))
	mustWriteFile(t, filepath.Join(root, "top.txt"), "x")
	mustWriteFile(t, filepath.Join(root, "a", "mid.txt"), "x")
	mustWriteFile(t, filepath.Join(root, "a", "b", "deep.txt"), "x")

	tool := newTestListDirTool(t, root)

	input, _ := json.Marshal(ListDirInput{Path: ".", MaxDepth: 0})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if strings.Contains(got, "mid.txt") || strings.Contains(got, "deep.txt") {
		t.Errorf("Run() with MaxDepth=0 = %q, want no recursion", got)
	}
	if !strings.Contains(got, "top.txt") || !strings.Contains(got, "a/") {
		t.Errorf("Run() with MaxDepth=0 = %q, want top-level entries", got)
	}

	input, _ = json.Marshal(ListDirInput{Path: ".", MaxDepth: 2})
	got, err = tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	for _, want := range []string{"top.txt", "a/", "a/mid.txt", "a/b/", "a/b/deep.txt"} {
		if !strings.Contains(got, want) {
			t.Errorf("Run() with MaxDepth=2 = %q, want it to contain %q", got, want)
		}
	}
}

func TestListDirEntryCap(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < 10; i++ {
		mustWriteFile(t, filepath.Join(root, fmt.Sprintf("file%02d.txt", i)), "x")
	}

	tool := newTestListDirTool(t, root)
	tool.entryCap = 3

	input, _ := json.Marshal(ListDirInput{Path: "."})
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(got, "entry cap of 3 reached") {
		t.Errorf("Run() = %q, want an entry-cap marker", got)
	}
	entryLines := strings.Split(strings.TrimSuffix(got, "\n... [entry cap of 3 reached, output truncated] ..."), "\n")
	if len(entryLines) != 3 {
		t.Errorf("Run() listed %d entries before the marker, want 3", len(entryLines))
	}
}

func TestListDirRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestListDirTool(t, root)

	input, _ := json.Marshal(ListDirInput{Path: "../"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}

func mustMkdirAll(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", path, err)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", path, err)
	}
}
