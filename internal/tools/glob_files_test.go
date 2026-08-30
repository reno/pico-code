package tools

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func newTestGlobFilesTool(t *testing.T, root string) *GlobFilesTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewGlobFilesTool(sandbox)
	if err != nil {
		t.Fatalf("NewGlobFilesTool() error = %v", err)
	}
	return tool
}

func runGlob(t *testing.T, tool *GlobFilesTool, in GlobFilesInput) string {
	t.Helper()
	input, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	got, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	return got
}

func TestGlobFilesSingleStarMatchesOnlyTopLevel(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "x")
	mustMkdirAll(t, filepath.Join(root, "sub"))
	mustWriteFile(t, filepath.Join(root, "sub", "b.go"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "*.go"})

	if !strings.Contains(got, "a.go") {
		t.Errorf("Run() = %q, want a.go matched", got)
	}
	if strings.Contains(got, "sub/b.go") {
		t.Errorf("Run() = %q, want sub/b.go not matched by a non-recursive pattern", got)
	}
}

func TestGlobFilesDoubleStarMatchesAcrossDirectories(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "x")
	mustMkdirAll(t, filepath.Join(root, "sub", "deeper"))
	mustWriteFile(t, filepath.Join(root, "sub", "b.go"), "x")
	mustWriteFile(t, filepath.Join(root, "sub", "deeper", "c.go"), "x")
	mustWriteFile(t, filepath.Join(root, "a.txt"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "**/*.go"})

	for _, want := range []string{"a.go", "sub/b.go", "sub/deeper/c.go"} {
		if !strings.Contains(got, want) {
			t.Errorf("Run() = %q, want it to contain %q", got, want)
		}
	}
	if strings.Contains(got, "a.txt") {
		t.Errorf("Run() = %q, want a.txt not matched", got)
	}
}

func TestGlobFilesPrefixedDirectoryPattern(t *testing.T) {
	root := t.TempDir()
	mustMkdirAll(t, filepath.Join(root, "internal", "tools"))
	mustMkdirAll(t, filepath.Join(root, "cmd"))
	mustWriteFile(t, filepath.Join(root, "internal", "tools", "a_test.go"), "x")
	mustWriteFile(t, filepath.Join(root, "internal", "a_test.go"), "x")
	mustWriteFile(t, filepath.Join(root, "cmd", "a_test.go"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "internal/**/*_test.go"})

	if !strings.Contains(got, "internal/tools/a_test.go") {
		t.Errorf("Run() = %q, want internal/tools/a_test.go matched", got)
	}
	if !strings.Contains(got, "internal/a_test.go") {
		t.Errorf("Run() = %q, want internal/a_test.go matched (** matches zero segments)", got)
	}
	if strings.Contains(got, "cmd/a_test.go") {
		t.Errorf("Run() = %q, want cmd/a_test.go not matched", got)
	}
}

func TestGlobFilesNoMatches(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "*.rb"})

	if got != "(no matches)" {
		t.Errorf("Run() = %q, want %q", got, "(no matches)")
	}
}

func TestGlobFilesSkipsDeniedFiles(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, ".env"), "x")
	mustWriteFile(t, filepath.Join(root, "a.go"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "**/*"})

	if strings.Contains(got, ".env") {
		t.Errorf("Run() = %q, want .env skipped", got)
	}
	if !strings.Contains(got, "a.go") {
		t.Errorf("Run() = %q, want a.go matched", got)
	}
}

func TestGlobFilesSkipsGitDir(t *testing.T) {
	root := t.TempDir()
	mustMkdirAll(t, filepath.Join(root, ".git"))
	mustWriteFile(t, filepath.Join(root, ".git", "config"), "x")

	tool := newTestGlobFilesTool(t, root)
	got := runGlob(t, tool, GlobFilesInput{Pattern: "**/*"})

	if strings.Contains(got, ".git") {
		t.Errorf("Run() = %q, want .git skipped entirely", got)
	}
}

func TestGlobFilesMatchCap(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < 10; i++ {
		mustWriteFile(t, filepath.Join(root, "file"+string(rune('a'+i))+".go"), "x")
	}

	tool := newTestGlobFilesTool(t, root)
	tool.matchCap = 3

	got := runGlob(t, tool, GlobFilesInput{Pattern: "*.go"})
	if !strings.Contains(got, "match cap of 3 reached") {
		t.Errorf("Run() = %q, want a match-cap marker", got)
	}
}

func TestGlobFilesRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestGlobFilesTool(t, root)

	input, _ := json.Marshal(GlobFilesInput{Pattern: "*.go", Path: "../"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}

func TestGlobFilesRejectsEmptyPattern(t *testing.T) {
	root := t.TempDir()
	tool := newTestGlobFilesTool(t, root)

	input, _ := json.Marshal(GlobFilesInput{Pattern: ""})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for an empty pattern")
	}
}

func TestGlobToRegexp(t *testing.T) {
	tests := []struct {
		pattern string
		path    string
		want    bool
	}{
		{"*.go", "a.go", true},
		{"*.go", "sub/a.go", false},
		{"**/*.go", "a.go", true},
		{"**/*.go", "sub/a.go", true},
		{"**/*.go", "sub/deeper/a.go", true},
		{"**", "anything/at/all.txt", true},
		{"a?.go", "ab.go", true},
		{"a?.go", "abc.go", false},
		{"internal/**/*_test.go", "internal/a_test.go", true},
		{"internal/**/*_test.go", "internal/tools/a_test.go", true},
		{"internal/**/*_test.go", "cmd/a_test.go", false},
	}
	for _, tt := range tests {
		re, err := globToRegexp(tt.pattern)
		if err != nil {
			t.Fatalf("globToRegexp(%q) error = %v", tt.pattern, err)
		}
		if got := re.MatchString(tt.path); got != tt.want {
			t.Errorf("globToRegexp(%q).MatchString(%q) = %v, want %v", tt.pattern, tt.path, got, tt.want)
		}
	}
}
