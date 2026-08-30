package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// normalizeToolCallArguments rewrites any message.tool_calls[].function.
// arguments value that arrived as a JSON-encoded string — a quirk some
// models produce instead of the documented object shape — into the object
// it encodes. api.ChatResponse's own decoding only accepts an object there
// and errors on a string, so this runs on the raw bytes before that decode.
// CLAUDE.md: handle the argument-shape quirk in the adapter, not the loop.
func normalizeToolCallArguments(data []byte) ([]byte, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, fmt.Errorf("normalize: %w", err)
	}
	msgRaw, ok := top["message"]
	if !ok {
		return data, nil
	}

	var msg map[string]json.RawMessage
	if err := json.Unmarshal(msgRaw, &msg); err != nil {
		return nil, fmt.Errorf("normalize: message: %w", err)
	}
	callsRaw, ok := msg["tool_calls"]
	if !ok {
		return data, nil
	}

	var calls []map[string]json.RawMessage
	if err := json.Unmarshal(callsRaw, &calls); err != nil {
		return nil, fmt.Errorf("normalize: tool_calls: %w", err)
	}

	changed := false
	for i, call := range calls {
		fnRaw, ok := call["function"]
		if !ok {
			continue
		}
		var fn map[string]json.RawMessage
		if err := json.Unmarshal(fnRaw, &fn); err != nil {
			return nil, fmt.Errorf("normalize: tool_calls[%d].function: %w", i, err)
		}
		argsRaw, ok := fn["arguments"]
		if !ok {
			continue
		}
		if !isJSONString(argsRaw) {
			continue
		}

		var encoded string
		if err := json.Unmarshal(argsRaw, &encoded); err != nil {
			return nil, fmt.Errorf("normalize: tool_calls[%d].function.arguments: %w", i, err)
		}
		if !json.Valid([]byte(encoded)) {
			return nil, fmt.Errorf("normalize: tool_calls[%d].function.arguments: not valid JSON once decoded: %q", i, encoded)
		}
		fn["arguments"] = json.RawMessage(encoded)

		fnData, err := json.Marshal(fn)
		if err != nil {
			return nil, fmt.Errorf("normalize: tool_calls[%d].function: %w", i, err)
		}
		call["function"] = fnData
		calls[i] = call
		changed = true
	}
	if !changed {
		return data, nil
	}

	callsData, err := json.Marshal(calls)
	if err != nil {
		return nil, fmt.Errorf("normalize: %w", err)
	}
	msg["tool_calls"] = callsData

	msgData, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("normalize: %w", err)
	}
	top["message"] = msgData

	return json.Marshal(top)
}

func isJSONString(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '"'
}
