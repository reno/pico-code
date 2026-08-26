package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// sampleInput mirrors the shape a real tool (phase 5's read_file) will use:
// one required field, two optional ones, each with a description that ends
// up in the generated schema.
type sampleInput struct {
	Path      string `json:"path" jsonschema_description:"path to the file, relative to the workspace root"`
	Recursive bool   `json:"recursive,omitempty" jsonschema_description:"recurse into subdirectories"`
	MaxDepth  int    `json:"max_depth,omitempty" jsonschema_description:"maximum recursion depth"`
}

const goldenPath = "testdata/golden/sample_input_schema.json"

func TestGenerateSchemaMatchesGolden(t *testing.T) {
	data, err := GenerateSchema(sampleInput{})
	if err != nil {
		t.Fatalf("GenerateSchema() error = %v", err)
	}

	var got any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("generated schema is not valid JSON: %v", err)
	}
	pretty, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent() error = %v", err)
	}
	pretty = append(pretty, '\n')

	if os.Getenv("RECORD") == "1" {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(goldenPath, pretty, 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to create it)", goldenPath, err)
	}

	if diff := cmp.Diff(string(want), string(pretty)); diff != "" {
		t.Errorf("generated schema mismatch (-want +got):\n%s", diff)
	}
}
