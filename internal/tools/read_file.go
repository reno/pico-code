package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ReadFileInput is read_file's argument shape.
type ReadFileInput struct {
	Path      string `json:"path" jsonschema:"description=Path to the file, relative to the workspace root"`
	StartLine int    `json:"start_line,omitempty" jsonschema:"description=1-based first line to include; 0 means the start of the file"`
	EndLine   int    `json:"end_line,omitempty" jsonschema:"description=1-based last line to include, inclusive; 0 means the end of the file"`
}

// ReadFileTool reads a file's contents, or a line range of it, from within
// a Sandbox.
type ReadFileTool struct {
	sandbox *Sandbox
	schema  json.RawMessage
}

// NewReadFileTool returns a ReadFileTool confined to sandbox.
func NewReadFileTool(sandbox *Sandbox) (*ReadFileTool, error) {
	schema, err := GenerateSchema(ReadFileInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: read_file schema: %w", err)
	}
	return &ReadFileTool{sandbox: sandbox, schema: schema}, nil
}

// Name implements tools.Tool.
func (t *ReadFileTool) Name() string { return "read_file" }

// Description implements tools.Tool.
func (t *ReadFileTool) Description() string {
	return "Read a file's contents, optionally a line range, from within the workspace."
}

// Schema implements tools.Tool.
func (t *ReadFileTool) Schema() json.RawMessage { return t.schema }

// Sandbox returns the Sandbox this tool is confined to. read_file is always
// registered (unlike write_file or run_command), so a caller that needs to
// re-root the workspace (the /cd command) can reach the shared Sandbox
// instance through it without the registry exposing sandboxes as a concept
// of its own.
func (t *ReadFileTool) Sandbox() *Sandbox { return t.sandbox }

// Run reads the requested file and returns its contents (or line range),
// truncated to defaultByteBudget.
func (t *ReadFileTool) Run(_ context.Context, input json.RawMessage) (string, error) {
	var in ReadFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: read_file: %w", err)
	}

	resolved, err := t.sandbox.Resolve(in.Path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(resolved)
	if err != nil {
		return "", fmt.Errorf("tools: read_file %q: %w", in.Path, err)
	}

	text := string(data)
	if in.StartLine > 0 || in.EndLine > 0 {
		text = sliceLines(text, in.StartLine, in.EndLine)
	}
	return truncateBytes(text, defaultByteBudget), nil
}

func sliceLines(text string, start, end int) string {
	lines := strings.Split(text, "\n")
	if start <= 0 {
		start = 1
	}
	if end <= 0 || end > len(lines) {
		end = len(lines)
	}
	if start > end || start > len(lines) {
		return ""
	}
	return strings.Join(lines[start-1:end], "\n")
}

var _ Tool = (*ReadFileTool)(nil)
