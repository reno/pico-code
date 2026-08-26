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
