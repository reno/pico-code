package tools

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func newTestSearchFilesTool(t *testing.T, root string) *SearchFilesTool {
	t.Helper()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	tool, err := NewSearchFilesTool(sandbox)
	if err != nil {
		t.Fatalf("NewSearchFilesTool() error = %v", err)
	}
	return tool
}

func runSearch(t *testing.T, tool *SearchFilesTool, in SearchFilesInput) string {
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

func TestSearchFilesLiteralMatch(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "package a\n\nfunc Foo() {}\n")
	mustWriteFile(t, filepath.Join(root, "b.go"), "package b\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "func Foo"})

	if !strings.Contains(got, "a.go:3: func Foo() {}") {
		t.Errorf("Run() = %q, want a match on a.go:3", got)
	}
	if strings.Contains(got, "b.go") {
		t.Errorf("Run() = %q, want no match in b.go", got)
	}
}

func TestSearchFilesRegexMatch(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "func Foo() {}\nfunc Bar() {}\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: `func (Foo|Bar)\(`, Regex: true})

	for _, want := range []string{"a.go:1: func Foo() {}", "a.go:2: func Bar() {}"} {
		if !strings.Contains(got, want) {
			t.Errorf("Run() = %q, want it to contain %q", got, want)
		}
	}
}

func TestSearchFilesInvalidRegex(t *testing.T) {
	root := t.TempDir()
	tool := newTestSearchFilesTool(t, root)

	input, _ := json.Marshal(SearchFilesInput{Pattern: "(", Regex: true})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for invalid regex")
	}
}

func TestSearchFilesGlobFilter(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "needle\n")
	mustWriteFile(t, filepath.Join(root, "a.txt"), "needle\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle", Glob: "*.go"})

	if !strings.Contains(got, "a.go:1") {
		t.Errorf("Run() = %q, want a match in a.go", got)
	}
	if strings.Contains(got, "a.txt") {
		t.Errorf("Run() = %q, want no match in a.txt", got)
	}
}

func TestSearchFilesIgnoreCase(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "NEEDLE\n")

	tool := newTestSearchFilesTool(t, root)

	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle"})
	if strings.Contains(got, "a.go") {
		t.Errorf("Run() case-sensitive = %q, want no match", got)
	}

	got = runSearch(t, tool, SearchFilesInput{Pattern: "needle", IgnoreCase: true})
	if !strings.Contains(got, "a.go:1: NEEDLE") {
		t.Errorf("Run() ignore_case = %q, want a match", got)
	}
}

func TestSearchFilesNoMatches(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, "a.go"), "package a\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "nowhere to be found"})

	if got != "(no matches)" {
		t.Errorf("Run() = %q, want %q", got, "(no matches)")
	}
}

func TestSearchFilesSkipsDeniedFiles(t *testing.T) {
	root := t.TempDir()
	mustWriteFile(t, filepath.Join(root, ".env"), "SECRET=needle\n")
	mustWriteFile(t, filepath.Join(root, "a.go"), "needle\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle"})

	if strings.Contains(got, ".env") {
		t.Errorf("Run() = %q, want .env skipped", got)
	}
	if !strings.Contains(got, "a.go:1") {
		t.Errorf("Run() = %q, want a.go matched", got)
	}
}

func TestSearchFilesSkipsGitDir(t *testing.T) {
	root := t.TempDir()
	mustMkdirAll(t, filepath.Join(root, ".git"))
	mustWriteFile(t, filepath.Join(root, ".git", "config"), "needle\n")

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle"})

	if got != "(no matches)" {
		t.Errorf("Run() = %q, want .git skipped entirely", got)
	}
}

func TestSearchFilesSkipsBinaryFiles(t *testing.T) {
	root := t.TempDir()
	binary := append([]byte("needle"), 0x00, 0x01, 0x02)
	mustWriteFile(t, filepath.Join(root, "a.bin"), string(binary))

	tool := newTestSearchFilesTool(t, root)
	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle"})

	if got != "(no matches)" {
		t.Errorf("Run() = %q, want binary file skipped", got)
	}
}

func TestSearchFilesMatchCap(t *testing.T) {
	root := t.TempDir()
	var b strings.Builder
	for i := 0; i < 10; i++ {
		b.WriteString("needle\n")
	}
	mustWriteFile(t, filepath.Join(root, "a.go"), b.String())

	tool := newTestSearchFilesTool(t, root)
	tool.matchCap = 3

	got := runSearch(t, tool, SearchFilesInput{Pattern: "needle"})
	if !strings.Contains(got, "match cap of 3 reached") {
		t.Errorf("Run() = %q, want a match-cap marker", got)
	}
	matchLines := strings.Split(strings.TrimSuffix(got, "\n... [match cap of 3 reached, output truncated] ..."), "\n")
	if len(matchLines) != 3 {
		t.Errorf("Run() returned %d matches before the marker, want 3", len(matchLines))
	}
}

func TestSearchFilesRejectsSandboxEscape(t *testing.T) {
	root := t.TempDir()
	tool := newTestSearchFilesTool(t, root)

	input, _ := json.Marshal(SearchFilesInput{Pattern: "needle", Path: "../"})
	_, err := tool.Run(context.Background(), input)
	if !errors.Is(err, ErrPathEscapesSandbox) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrPathEscapesSandbox)
	}
}

func TestSearchFilesRejectsEmptyPattern(t *testing.T) {
	root := t.TempDir()
	tool := newTestSearchFilesTool(t, root)

	input, _ := json.Marshal(SearchFilesInput{Pattern: ""})
	_, err := tool.Run(context.Background(), input)
	if err == nil {
		t.Fatal("Run() error = nil, want an error for an empty pattern")
	}
}
