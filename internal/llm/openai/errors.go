package openai

import (
	"bytes"
	"errors"
	"fmt"
	"net/http"
)

// Sentinel errors Chat maps a non-2xx status to, so callers can branch on a
// stable error without depending on this adapter's own HTTP details. Same
// status-code table as the Anthropic adapter's mapError (CLAUDE.md §2.2),
// applied here to a raw net/http response rather than an SDK error type:
// there is no OpenAI SDK on the dependency allowlist, and a compatible
// endpoint (vLLM, LM Studio, Ollama's /v1) has no SDK of its own either.
var (
	ErrUnauthorized    = errors.New("openai: unauthorized")
	ErrForbidden       = errors.New("openai: forbidden")
	ErrNotFound        = errors.New("openai: not found")
	ErrPayloadTooLarge = errors.New("openai: payload too large")
	ErrRateLimited     = errors.New("openai: rate limited")
	ErrServerError     = errors.New("openai: server error")
)

// mapError translates a non-2xx HTTP response into one of the sentinels
// above when it recognizes the status code, wrapping a base error built
// from status and body so errors.Is still works and the message still
// carries the backend's own explanation. Any other status passes the base
// error through unchanged.
func mapError(status int, body []byte) error {
	base := fmt.Errorf("openai: request failed with status %d: %s", status, bytes.TrimSpace(body))

	switch status {
	case http.StatusUnauthorized:
		return fmt.Errorf("%w: %w", ErrUnauthorized, base)
	case http.StatusForbidden:
		return fmt.Errorf("%w: %w", ErrForbidden, base)
	case http.StatusNotFound:
		return fmt.Errorf("%w: %w", ErrNotFound, base)
	case http.StatusRequestEntityTooLarge:
		return fmt.Errorf("%w: %w", ErrPayloadTooLarge, base)
	case http.StatusTooManyRequests:
		return fmt.Errorf("%w: %w", ErrRateLimited, base)
	default:
		if status >= http.StatusInternalServerError {
			return fmt.Errorf("%w: %w", ErrServerError, base)
		}
		return base
	}
}
