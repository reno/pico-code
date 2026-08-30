package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// EditFileInput is edit_file's argument shape.
type EditFileInput struct {
	Path       string `json:"path" jsonschema:"description=Path to the file, relative to the workspace root"`
	OldString  string `json:"old_string" jsonschema:"description=Exact text to find; must match exactly once unless replace_all is set"`
	NewString  string `json:"new_string" jsonschema:"description=Text to replace old_string with"`
	ReplaceAll bool   `json:"replace_all,omitempty" jsonschema:"description=Replace every occurrence of old_string instead of requiring exactly one match"`
}

// EditFileTool replaces a substring within an existing file atomically,
// within a Sandbox. Unlike WriteFileTool, it never creates a file: a
// targeted edit only makes sense against content the model has already
// read, so a missing file is an error rather than a fresh write.
type EditFileTool struct {
	sandbox *Sandbox
	schema  json.RawMessage
}

// NewEditFileTool returns an EditFileTool confined to sandbox.
func NewEditFileTool(sandbox *Sandbox) (*EditFileTool, error) {
	schema, err := GenerateSchema(EditFileInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: edit_file schema: %w", err)
	}
	return &EditFileTool{sandbox: sandbox, schema: schema}, nil
}

// Name implements tools.Tool.
func (t *EditFileTool) Name() string { return "edit_file" }

// Description implements tools.Tool.
func (t *EditFileTool) Description() string {
	return "Replace an exact substring within an existing file, atomically, within the workspace."
}

// Schema implements tools.Tool.
func (t *EditFileTool) Schema() json.RawMessage { return t.schema }

// NeedsApproval implements tools.ApprovalRequired: every edit needs
// sign-off, same tier as write_file.
func (t *EditFileTool) NeedsApproval() bool { return true }

// Preview implements tools.Previewable: a line diff of the file's current
// contents against the result of applying the edit.
func (t *EditFileTool) Preview(_ context.Context, input json.RawMessage) (string, error) {
	var in EditFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: edit_file preview: %w", err)
	}
	old, _, err := t.readAndApply(in)
	if err != nil {
		return "", err
	}
	newContent, _, err := applyEdit(old, in)
	if err != nil {
		return "", err
	}
	return lineDiff(old, newContent), nil
}

// Run applies the edit and writes the result atomically: a temp file in
// the same directory, then a rename over the target, so a reader never
// observes a partially written file and an interrupted run leaves the
// original untouched (same mechanism as write_file.Run).
func (t *EditFileTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in EditFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: edit_file: %w", err)
	}

	old, resolved, err := t.readAndApply(in)
	if err != nil {
		return "", err
	}
	newContent, count, err := applyEdit(old, in)
	if err != nil {
		return "", err
	}

	if err := atomicWriteFile(ctx, resolved, newContent); err != nil {
		return "", fmt.Errorf("tools: edit_file %q: %w", in.Path, err)
	}
	return fmt.Sprintf("replaced %d occurrence(s) in %s", count, in.Path), nil
}

// readAndApply resolves in.Path and reads its current contents, erroring
// on a file that doesn't exist rather than treating edit_file as a way to
// create one.
func (t *EditFileTool) readAndApply(in EditFileInput) (content, resolved string, err error) {
	resolved, err = t.sandbox.Resolve(in.Path)
	if err != nil {
		return "", "", err
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		if os.IsNotExist(err) {
			return "", "", fmt.Errorf("tools: edit_file %q: file does not exist; edit_file only modifies existing files", in.Path)
		}
		return "", "", fmt.Errorf("tools: edit_file %q: %w", in.Path, err)
	}
	return string(data), resolved, nil
}

// applyEdit validates in and returns the patched content plus the number
// of occurrences replaced.
func applyEdit(content string, in EditFileInput) (string, int, error) {
	if in.OldString == "" {
		return "", 0, fmt.Errorf("tools: edit_file %q: old_string must not be empty", in.Path)
	}
	if in.OldString == in.NewString {
		return "", 0, fmt.Errorf("tools: edit_file %q: old_string and new_string are identical", in.Path)
	}

	count := strings.Count(content, in.OldString)
	switch {
	case count == 0:
		return "", 0, fmt.Errorf("tools: edit_file %q: old_string not found; re-read the file, its contents may have changed", in.Path)
	case count > 1 && !in.ReplaceAll:
		return "", 0, fmt.Errorf("tools: edit_file %q: old_string matches %d times, want exactly 1; add more surrounding context to make it unique, or set replace_all", in.Path, count)
	}

	if in.ReplaceAll {
		return strings.ReplaceAll(content, in.OldString, in.NewString), count, nil
	}
	return strings.Replace(content, in.OldString, in.NewString, 1), 1, nil
}

var (
	_ ApprovalRequired = (*EditFileTool)(nil)
	_ Previewable      = (*EditFileTool)(nil)
)
