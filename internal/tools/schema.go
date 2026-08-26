package tools

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/invopop/jsonschema"
)

// schemaReflector configures how Go structs become JSON Schema for a tool's
// Schema(): flattened (no top-level $ref/$defs indirection for the input
// struct itself) and without the $schema/$id noise a provider's tool
// definition has no use for.
var schemaReflector = &jsonschema.Reflector{
	ExpandedStruct: true,
	DoNotReference: true,
}

// GenerateSchema builds a Tool's Schema() return value from a Go struct,
// the input type Run will decode into. Passing a struct rather than hand-
// writing JSON keeps the schema and the decode logic from drifting apart.
// Tools call this once, at construction, and propagate a non-nil error
// through their own constructor rather than from Schema() (whose signature,
// fixed by the Tool interface, has no error to return).
func GenerateSchema(v any) (json.RawMessage, error) {
	s := schemaReflector.Reflect(v)
	s.Version = ""
	s.ID = ""

	data, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("tools: marshal generated schema: %w", err)
	}
	return data, nil
}

// validateInput checks input against schema, returning a descriptive error
// naming the offending field on the first mismatch. It covers the subset of
// JSON Schema GenerateSchema actually produces (object/array/string/number/
// integer/boolean, required, enum) — not arbitrary externally authored
// schemas.
func validateInput(schema json.RawMessage, input json.RawMessage) error {
	var s jsonschema.Schema
	if err := json.Unmarshal(schema, &s); err != nil {
		return fmt.Errorf("tools: schema is not valid JSON Schema: %w", err)
	}

	var value any
	if err := json.Unmarshal(input, &value); err != nil {
		return fmt.Errorf("input: not valid JSON: %w", err)
	}

	return validateValue(&s, value, "input")
}

func validateValue(schema *jsonschema.Schema, value any, path string) error {
	if schema == nil {
		return nil
	}

	if len(schema.Enum) > 0 && !enumContains(schema.Enum, value) {
		return fmt.Errorf("%s: value %v is not one of %v", path, value, schema.Enum)
	}

	switch schema.Type {
	case "", "object":
		return validateObject(schema, value, path)
	case "array":
		return validateArray(schema, value, path)
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("%s: want string, got %T", path, value)
		}
	case "integer":
		n, ok := value.(float64)
		if !ok {
			return fmt.Errorf("%s: want integer, got %T", path, value)
		}
		if n != float64(int64(n)) {
			return fmt.Errorf("%s: want integer, got %v", path, value)
		}
	case "number":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("%s: want number, got %T", path, value)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s: want boolean, got %T", path, value)
		}
	}
	return nil
}

func validateObject(schema *jsonschema.Schema, value any, path string) error {
	obj, ok := value.(map[string]any)
	if !ok {
		return fmt.Errorf("%s: want object, got %T", path, value)
	}

	for _, name := range schema.Required {
		if _, present := obj[name]; !present {
			return fmt.Errorf("%s: missing required field %q", path, name)
		}
	}

	if schema.Properties == nil {
		return nil
	}
	for name, propSchema := range schema.Properties.FromOldest() {
		v, present := obj[name]
		if !present {
			continue
		}
		if err := validateValue(propSchema, v, path+"."+name); err != nil {
			return err
		}
	}
	return nil
}

func validateArray(schema *jsonschema.Schema, value any, path string) error {
	arr, ok := value.([]any)
	if !ok {
		return fmt.Errorf("%s: want array, got %T", path, value)
	}
	if schema.Items == nil {
		return nil
	}
	for i, v := range arr {
		if err := validateValue(schema.Items, v, fmt.Sprintf("%s[%d]", path, i)); err != nil {
			return err
		}
	}
	return nil
}

func enumContains(enum []any, value any) bool {
	for _, e := range enum {
		if reflect.DeepEqual(e, value) {
			return true
		}
	}
	return false
}
