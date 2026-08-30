package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// WriteFileInput is write_file's argument shape.
type WriteFileInput struct {
	Path    string `json:"path" jsonschema:"description=Path to the file, relative to the workspace root"`
	Content string `json:"content" jsonschema:"description=The file's new full contents"`
}

// WriteFileTool atomically writes a file's full contents within a Sandbox.
type WriteFileTool struct {
	sandbox *Sandbox
	schema  json.RawMessage
}

// NewWriteFileTool returns a WriteFileTool confined to sandbox.
func NewWriteFileTool(sandbox *Sandbox) (*WriteFileTool, error) {
	schema, err := GenerateSchema(WriteFileInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: write_file schema: %w", err)
	}
	return &WriteFileTool{sandbox: sandbox, schema: schema}, nil
}

// Name implements tools.Tool.
func (t *WriteFileTool) Name() string { return "write_file" }

// Description implements tools.Tool.
func (t *WriteFileTool) Description() string {
	return "Write a file's full contents atomically, creating or overwriting it, within the workspace."
}

// Schema implements tools.Tool.
func (t *WriteFileTool) Schema() json.RawMessage { return t.schema }

// NeedsApproval implements tools.ApprovalRequired: every write needs
// sign-off.
func (t *WriteFileTool) NeedsApproval() bool { return true }

// Preview implements tools.Previewable: a line diff of the current file
// contents (empty if it doesn't exist yet) against the proposed content.
func (t *WriteFileTool) Preview(_ context.Context, input json.RawMessage) (string, error) {
	var in WriteFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: write_file preview: %w", err)
	}
	resolved, err := t.sandbox.Resolve(in.Path)
	if err != nil {
		return "", err
	}
	old, err := os.ReadFile(resolved)
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("tools: write_file preview %q: %w", in.Path, err)
	}
	return lineDiff(string(old), in.Content), nil
}

// Run writes in.Content to in.Path atomically (see atomicWriteFile).
func (t *WriteFileTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in WriteFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: write_file: %w", err)
	}

	resolved, err := t.sandbox.Resolve(in.Path)
	if err != nil {
		return "", err
	}

	if err := atomicWriteFile(ctx, resolved, in.Content); err != nil {
		return "", fmt.Errorf("tools: write_file %q: %w", in.Path, err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(in.Content), in.Path), nil
}

// atomicWriteFile writes content to resolved (an already sandbox-resolved
// absolute path) by writing to a temp file in the same directory, then
// renaming it over the target — so a reader never observes a partially
// written file and an interrupted run leaves the original untouched.
// Shared by write_file and edit_file so both commit through the identical
// mechanism.
func atomicWriteFile(ctx context.Context, resolved, content string) error {
	dir := filepath.Dir(resolved)
	tmp, err := os.CreateTemp(dir, ".pico-write-*")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath) //nolint:errcheck // no-op once the rename below succeeds

	if _, err := tmp.WriteString(content); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := os.Rename(tmpPath, resolved); err != nil {
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

var (
	_ ApprovalRequired = (*WriteFileTool)(nil)
	_ Previewable      = (*WriteFileTool)(nil)
)
