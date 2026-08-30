package ollama

import (
	"bytes"
	"errors"
	"fmt"
	"net/http"
)

// ErrNotFound is returned when Ollama's local daemon doesn't recognize a
// model name — the same "have you pulled it?" case Anthropic and OpenAI
// surface as an HTTP 404, mapped here to a stable sentinel so a caller can
// tell it apart from a transport error or a server-side failure without
// parsing the response body. Ollama has no SDK error type of its own (Chat,
// Stream, and probeShow all speak raw net/http), so mapError takes the
// status and body directly, mirroring internal/llm/openai/errors.go.
var ErrNotFound = errors.New("ollama: not found")

// mapError builds an error from a non-2xx HTTP response, wrapping it in
// ErrNotFound when status is 404 so errors.Is can classify it; any other
// status passes through as a plain error carrying the body Ollama sent.
func mapError(status int, body []byte) error {
	base := fmt.Errorf("ollama: request failed with status %d: %s", status, bytes.TrimSpace(body))
	if status == http.StatusNotFound {
		return fmt.Errorf("%w: %w", ErrNotFound, base)
	}
	return base
}
