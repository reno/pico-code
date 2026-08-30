// Package tools defines the Tool interface every built-in tool (phase 5)
// implements, plus a Registry that looks tools up by name and validates
// their input against a JSON Schema before running them.
package tools

import (
	"context"
	"encoding/json"
)

// Tool is a single capability the model can invoke. Schema returns a JSON
// Schema describing Run's expected input, generated once at construction
// from a Go struct via GenerateSchema so the schema and the decoding logic
// in Run can never drift apart.
type Tool interface {
	Name() string
	Description() string
	Schema() json.RawMessage
	Run(ctx context.Context, input json.RawMessage) (string, error)
}

// ApprovalRequired is implemented by a Tool whose side effects need
// explicit user sign-off before Run executes. CLAUDE.md invariant 5 makes
// the agent loop the sole owner of approval policy, so a Tool only
// declares whether it needs approval — it never prompts.
type ApprovalRequired interface {
	Tool
	NeedsApproval() bool
}

// Previewable is implemented by a Tool that can describe its effect ahead
// of Run — e.g. write_file's diff against the file's current contents —
// so the loop can show it alongside an approval prompt.
type Previewable interface {
	Tool
	Preview(ctx context.Context, input json.RawMessage) (string, error)
}
