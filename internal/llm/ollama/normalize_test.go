package ollama

import "testing"

func TestNormalizeToolCallArgumentsNoOpWithoutToolCalls(t *testing.T) {
	data := []byte(`{"message":{"role":"assistant","content":"hi"}}`)
	got, err := normalizeToolCallArguments(data)
	if err != nil {
		t.Fatalf("normalizeToolCallArguments() error = %v", err)
	}
	if string(got) != string(data) {
		t.Errorf("normalizeToolCallArguments() = %s, want unchanged input", got)
	}
}

func TestNormalizeToolCallArgumentsRejectsUndecodableString(t *testing.T) {
	data := []byte(`{"message":{"tool_calls":[{"function":{"name":"f","arguments":"not json"}}]}}`)
	if _, err := normalizeToolCallArguments(data); err == nil {
		t.Fatal("normalizeToolCallArguments() error = nil, want an error for a string that isn't JSON once decoded")
	}
}

func TestNormalizeToolCallArgumentsMissingMessageIsNoOp(t *testing.T) {
	data := []byte(`{"done":true}`)
	got, err := normalizeToolCallArguments(data)
	if err != nil {
		t.Fatalf("normalizeToolCallArguments() error = %v", err)
	}
	if string(got) != string(data) {
		t.Errorf("normalizeToolCallArguments() = %s, want unchanged input", got)
	}
}
