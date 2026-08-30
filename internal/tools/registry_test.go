package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type fakeToolInput struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type fakeTool struct {
	name   string
	schema json.RawMessage
	run    func(ctx context.Context, input json.RawMessage) (string, error)
}

func newFakeTool(t *testing.T, name string) *fakeTool {
	t.Helper()
	schema, err := GenerateSchema(fakeToolInput{})
	if err != nil {
		t.Fatalf("GenerateSchema() error = %v", err)
	}
	return &fakeTool{name: name, schema: schema}
}

func (f *fakeTool) Name() string            { return f.name }
func (f *fakeTool) Description() string     { return "a fake tool for tests" }
func (f *fakeTool) Schema() json.RawMessage { return f.schema }
func (f *fakeTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	if f.run != nil {
		return f.run(ctx, input)
	}
	return "ok", nil
}

var _ Tool = (*fakeTool)(nil)

func TestRegistryRegisterAndGet(t *testing.T) {
	r := NewRegistry()
	tool := newFakeTool(t, "read_file")
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	got, err := r.Get("read_file")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got != tool {
		t.Errorf("Get() returned a different tool instance")
	}
}

func TestRegistryDuplicateNameErrors(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(newFakeTool(t, "read_file")); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}
	err := r.Register(newFakeTool(t, "read_file"))
	if !errors.Is(err, ErrDuplicateTool) {
		t.Fatalf("second Register() error = %v, want wrapping %v", err, ErrDuplicateTool)
	}
}

func TestRegistryGetUnknownTool(t *testing.T) {
	r := NewRegistry()
	_, err := r.Get("does_not_exist")
	if !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("Get() error = %v, want wrapping %v", err, ErrToolNotFound)
	}
}

func TestRegistryRunUnknownToolName(t *testing.T) {
	r := NewRegistry()
	_, err := r.Run(context.Background(), "does_not_exist", json.RawMessage(`{}`))
	if !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("Run() error = %v, want wrapping %v", err, ErrToolNotFound)
	}
}

func TestRegistryRunValidatesInputBeforeCallingRun(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantErrPart string
	}{
		{
			name:        "missing required field",
			input:       `{"recursive":true}`,
			wantErrPart: `missing required field "path"`,
		},
		{
			name:        "wrong type",
			input:       `{"path":123}`,
			wantErrPart: `input.path: want string`,
		},
		{
			name:        "not json",
			input:       `not json`,
			wantErrPart: "not valid JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewRegistry()
			called := false
			tool := newFakeTool(t, "read_file")
			tool.run = func(_ context.Context, _ json.RawMessage) (string, error) {
				called = true
				return "should not run", nil
			}
			if err := r.Register(tool); err != nil {
				t.Fatalf("Register() error = %v", err)
			}

			_, err := r.Run(context.Background(), "read_file", json.RawMessage(tt.input))
			if err == nil {
				t.Fatal("Run() error = nil, want a validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErrPart) {
				t.Errorf("Run() error = %q, want substring %q", err.Error(), tt.wantErrPart)
			}
			if called {
				t.Error("Run() invoked the tool despite invalid input")
			}
		})
	}
}

func TestRegistryRunCallsToolOnValidInput(t *testing.T) {
	r := NewRegistry()
	tool := newFakeTool(t, "read_file")
	tool.run = func(_ context.Context, input json.RawMessage) (string, error) {
		var in fakeToolInput
		if err := json.Unmarshal(input, &in); err != nil {
			return "", err
		}
		return "read " + in.Path, nil
	}
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	out, err := r.Run(context.Background(), "read_file", json.RawMessage(`{"path":"a.txt"}`))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if out != "read a.txt" {
		t.Errorf("Run() = %q, want %q", out, "read a.txt")
	}
}
