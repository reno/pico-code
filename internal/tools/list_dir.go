package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const defaultEntryCap = 500

// ListDirInput is list_dir's argument shape.
type ListDirInput struct {
	Path     string `json:"path" jsonschema:"description=Directory path, relative to the workspace root"`
	MaxDepth int    `json:"max_depth,omitempty" jsonschema:"description=Maximum recursion depth; 0 lists only the given directory"`
}

// ListDirTool lists directory entries from within a Sandbox.
type ListDirTool struct {
	sandbox  *Sandbox
	schema   json.RawMessage
	entryCap int
}

// NewListDirTool returns a ListDirTool confined to sandbox.
func NewListDirTool(sandbox *Sandbox) (*ListDirTool, error) {
	schema, err := GenerateSchema(ListDirInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: list_dir schema: %w", err)
	}
	return &ListDirTool{sandbox: sandbox, schema: schema, entryCap: defaultEntryCap}, nil
}

// Name implements tools.Tool.
func (t *ListDirTool) Name() string { return "list_dir" }

// Description implements tools.Tool.
func (t *ListDirTool) Description() string {
	return "List directory entries, optionally recursively, from within the workspace."
}

// Schema implements tools.Tool.
func (t *ListDirTool) Schema() json.RawMessage { return t.schema }

// Run lists the requested directory up to MaxDepth and entryCap entries,
// output truncated to defaultByteBudget.
func (t *ListDirTool) Run(_ context.Context, input json.RawMessage) (string, error) {
	var in ListDirInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: list_dir: %w", err)
	}

	resolved, err := t.sandbox.Resolve(in.Path)
	if err != nil {
		return "", err
	}

	var lines []string
	count := 0
	capped := false

	var walk func(dir string, depth int, prefix string) error
	walk = func(dir string, depth int, prefix string) error {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return err
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })

		for _, e := range entries {
			if count >= t.entryCap {
				capped = true
				return nil
			}
			rel := prefix + e.Name()
			if e.IsDir() {
				rel += "/"
			}
			lines = append(lines, rel)
			count++

			if e.IsDir() && depth < in.MaxDepth {
				if err := walk(filepath.Join(dir, e.Name()), depth+1, rel); err != nil {
					return err
				}
			}
		}
		return nil
	}

	if err := walk(resolved, 0, ""); err != nil {
		return "", fmt.Errorf("tools: list_dir %q: %w", in.Path, err)
	}

	out := strings.Join(lines, "\n")
	if capped {
		out += fmt.Sprintf("\n... [entry cap of %d reached, output truncated] ...", t.entryCap)
	}
	return truncateBytes(out, defaultByteBudget), nil
}

var _ Tool = (*ListDirTool)(nil)
