package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	defaultMatchCap  = 500
	binarySniffBytes = 8000
)

// SearchFilesInput is search_files's argument shape.
type SearchFilesInput struct {
	Pattern    string `json:"pattern" jsonschema:"description=Text to search for within file contents"`
	Path       string `json:"path,omitempty" jsonschema:"description=Directory to search, relative to the workspace root; defaults to the root"`
	Glob       string `json:"glob,omitempty" jsonschema:"description=Only search files whose name matches this glob, e.g. *.go"`
	Regex      bool   `json:"regex,omitempty" jsonschema:"description=Treat pattern as an RE2 regular expression instead of a literal substring"`
	IgnoreCase bool   `json:"ignore_case,omitempty" jsonschema:"description=Case-insensitive match"`
}

// SearchFilesTool searches file contents recursively from within a Sandbox.
type SearchFilesTool struct {
	sandbox  *Sandbox
	schema   json.RawMessage
	matchCap int
}

// NewSearchFilesTool returns a SearchFilesTool confined to sandbox.
func NewSearchFilesTool(sandbox *Sandbox) (*SearchFilesTool, error) {
	schema, err := GenerateSchema(SearchFilesInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: search_files schema: %w", err)
	}
	return &SearchFilesTool{sandbox: sandbox, schema: schema, matchCap: defaultMatchCap}, nil
}

// Name implements tools.Tool.
func (t *SearchFilesTool) Name() string { return "search_files" }

// Description implements tools.Tool.
func (t *SearchFilesTool) Description() string {
	return "Search file contents for a literal string or RE2 regular expression within the workspace, returning matching file:line results."
}

// Schema implements tools.Tool.
func (t *SearchFilesTool) Schema() json.RawMessage { return t.schema }

// Run walks Path (the workspace root by default), matching Pattern against
// every non-binary file's lines, up to matchCap results, output truncated
// to defaultByteBudget. Directories named .git and files denied by the
// sandbox are skipped silently, same as a direct read_file of one would be
// rejected; an unreadable file (permissions, a race with deletion) is
// skipped rather than failing the whole search.
func (t *SearchFilesTool) Run(_ context.Context, input json.RawMessage) (string, error) {
	var in SearchFilesInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: search_files: %w", err)
	}
	if in.Pattern == "" {
		return "", fmt.Errorf("tools: search_files: pattern must not be empty")
	}

	path := in.Path
	if path == "" {
		path = "."
	}
	resolved, err := t.sandbox.Resolve(path)
	if err != nil {
		return "", err
	}

	match, err := newContentMatcher(in.Pattern, in.Regex, in.IgnoreCase)
	if err != nil {
		return "", fmt.Errorf("tools: search_files: %w", err)
	}

	var lines []string
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

			if in.Glob != "" {
				if ok, _ := filepath.Match(in.Glob, e.Name()); !ok {
					continue
				}
			}

			full := filepath.Join(dir, e.Name())
			if t.sandbox.Denied(full) {
				continue
			}

			matches, err := searchFile(full, prefix+e.Name(), match, t.matchCap-count)
			if err != nil {
				continue
			}
			lines = append(lines, matches...)
			count += len(matches)
			if count >= t.matchCap {
				capped = true
			}
		}
		return nil
	}

	if err := walk(resolved, ""); err != nil {
		return "", fmt.Errorf("tools: search_files %q: %w", path, err)
	}

	out := strings.Join(lines, "\n")
	if capped {
		out += fmt.Sprintf("\n... [match cap of %d reached, output truncated] ...", t.matchCap)
	}
	if out == "" {
		out = "(no matches)"
	}
	return truncateBytes(out, defaultByteBudget), nil
}

// contentMatcher reports whether a single line matches a search pattern.
type contentMatcher func(line string) bool

// newContentMatcher builds a literal substring matcher, or a regexp-backed
// one when useRegex is set. RE2 (Go's regexp) rather than a shelled-out
// grep: no new binary dependency, and RE2's linear-time guarantee means an
// untrusted model-supplied pattern can't cause catastrophic backtracking.
func newContentMatcher(pattern string, useRegex, ignoreCase bool) (contentMatcher, error) {
	if useRegex {
		expr := pattern
		if ignoreCase {
			expr = "(?i)" + expr
		}
		re, err := regexp.Compile(expr)
		if err != nil {
			return nil, fmt.Errorf("invalid regex: %w", err)
		}
		return re.MatchString, nil
	}

	needle := pattern
	if ignoreCase {
		needle = strings.ToLower(needle)
	}
	return func(line string) bool {
		if ignoreCase {
			line = strings.ToLower(line)
		}
		return strings.Contains(line, needle)
	}, nil
}

// searchFile reads full, skips it if binary (a NUL byte within the first
// binarySniffBytes), and returns up to limit "rel:line: text" matches.
func searchFile(full, rel string, match contentMatcher, limit int) ([]string, error) {
	if limit <= 0 {
		return nil, nil
	}
	data, err := os.ReadFile(full)
	if err != nil {
		return nil, err
	}

	sniff := data
	if len(sniff) > binarySniffBytes {
		sniff = sniff[:binarySniffBytes]
	}
	if bytes.IndexByte(sniff, 0) != -1 {
		return nil, nil
	}

	var out []string
	for i, line := range strings.Split(string(data), "\n") {
		if len(out) >= limit {
			break
		}
		if match(line) {
			out = append(out, fmt.Sprintf("%s:%d: %s", rel, i+1, line))
		}
	}
	return out, nil
}

var _ Tool = (*SearchFilesTool)(nil)
