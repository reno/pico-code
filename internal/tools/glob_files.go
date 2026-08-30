package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// GlobFilesInput is glob_files's argument shape.
type GlobFilesInput struct {
	Pattern string `json:"pattern" jsonschema:"description=Glob pattern to match file paths against; ** matches across directories, * matches within one path segment, e.g. **/*.go or internal/**/*_test.go"`
	Path    string `json:"path,omitempty" jsonschema:"description=Directory to search from, relative to the workspace root; defaults to the root"`
}

// GlobFilesTool finds files whose path matches a glob pattern, recursively,
// from within a Sandbox.
type GlobFilesTool struct {
	sandbox  *Sandbox
	schema   json.RawMessage
	matchCap int
}

// NewGlobFilesTool returns a GlobFilesTool confined to sandbox.
func NewGlobFilesTool(sandbox *Sandbox) (*GlobFilesTool, error) {
	schema, err := GenerateSchema(GlobFilesInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: glob_files schema: %w", err)
	}
	return &GlobFilesTool{sandbox: sandbox, schema: schema, matchCap: defaultEntryCap}, nil
}

// Name implements tools.Tool.
func (t *GlobFilesTool) Name() string { return "glob_files" }

// Description implements tools.Tool.
func (t *GlobFilesTool) Description() string {
	return "Find files whose path matches a glob pattern, recursively, within the workspace."
}

// Schema implements tools.Tool.
func (t *GlobFilesTool) Schema() json.RawMessage { return t.schema }

// Run walks Path (the workspace root by default), returning every file
// whose path relative to Path matches Pattern, up to matchCap results,
// output truncated to defaultByteBudget. Directories named .git and files
// denied by the sandbox are skipped silently, same as search_files.
func (t *GlobFilesTool) Run(_ context.Context, input json.RawMessage) (string, error) {
	var in GlobFilesInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: glob_files: %w", err)
	}
	if in.Pattern == "" {
		return "", fmt.Errorf("tools: glob_files: pattern must not be empty")
	}

	path := in.Path
	if path == "" {
		path = "."
	}
	resolved, err := t.sandbox.Resolve(path)
	if err != nil {
		return "", err
	}

	re, err := globToRegexp(in.Pattern)
	if err != nil {
		return "", fmt.Errorf("tools: glob_files: invalid pattern: %w", err)
	}

	var results []string
	count := 0
	capped := false

	var walk func(dir, prefix string) error
	walk = func(dir, prefix string) error {
		if capped {
			return nil
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return err
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })

		for _, e := range entries {
			if capped {
				return nil
			}
			if e.IsDir() {
				if e.Name() == ".git" {
					continue
				}
				if err := walk(filepath.Join(dir, e.Name()), prefix+e.Name()+"/"); err != nil {
					return err
				}
				continue
			}

			full := filepath.Join(dir, e.Name())
			if t.sandbox.Denied(full) {
				continue
			}

			rel := prefix + e.Name()
			if !re.MatchString(rel) {
				continue
			}

			results = append(results, rel)
			count++
			if count >= t.matchCap {
				capped = true
			}
		}
		return nil
	}

	if err := walk(resolved, ""); err != nil {
		return "", fmt.Errorf("tools: glob_files %q: %w", path, err)
	}

	out := strings.Join(results, "\n")
	if capped {
		out += fmt.Sprintf("\n... [match cap of %d reached, output truncated] ...", t.matchCap)
	}
	if out == "" {
		out = "(no matches)"
	}
	return truncateBytes(out, defaultByteBudget), nil
}

// globToRegexp translates a doublestar-style glob into an anchored regexp:
// "**/" matches zero or more whole path segments, a lone "**" matches
// anything including "/", "*" matches within a single segment, "?" matches
// one character within a segment. Everything else is matched literally.
func globToRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")

	runes := []rune(pattern)
	for i := 0; i < len(runes); {
		switch runes[i] {
		case '*':
			if i+1 < len(runes) && runes[i+1] == '*' {
				if i+2 < len(runes) && runes[i+2] == '/' {
					b.WriteString("(?:.*/)?")
					i += 3
					continue
				}
				b.WriteString(".*")
				i += 2
				continue
			}
			b.WriteString("[^/]*")
			i++
		case '?':
			b.WriteString("[^/]")
			i++
		default:
			b.WriteString(regexp.QuoteMeta(string(runes[i])))
			i++
		}
	}

	b.WriteString("$")
	return regexp.Compile(b.String())
}

var _ Tool = (*GlobFilesTool)(nil)
