package anthropic

import (
	"errors"
	"fmt"
	"net/http"

	sdk "github.com/anthropics/anthropic-sdk-go"
)

// Sentinel errors Chat maps API failures to, so callers can branch on a
// stable error without importing anthropic-sdk-go themselves (CLAUDE.md
// invariant 1: internal/llm never depends on a provider SDK).
//
// Retries already happened by the time one of these surfaces: the SDK
// retries 429 and 5xx internally (exponential backoff with jitter, honoring
// Retry-After, ctx-aware) before giving up, so these represent a retry
// budget exhausted or a non-retryable failure.
var (
	ErrUnauthorized    = errors.New("anthropic: unauthorized")
	ErrForbidden       = errors.New("anthropic: forbidden")
	ErrNotFound        = errors.New("anthropic: not found")
	ErrPayloadTooLarge = errors.New("anthropic: payload too large")
	ErrRateLimited     = errors.New("anthropic: rate limited")
	ErrServerError     = errors.New("anthropic: server error")
)

// mapError translates an error from the SDK into one of the sentinels above
// when it recognizes the HTTP status, wrapping the original error so
// errors.Is still works for both. Anything else — a cancelled ctx, a
// transport error, a status code with no sentinel — passes through
// unchanged.
func mapError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return err
	}

	switch apiErr.StatusCode {
	case http.StatusUnauthorized:
		return fmt.Errorf("%w: %w", ErrUnauthorized, err)
	case http.StatusForbidden:
		return fmt.Errorf("%w: %w", ErrForbidden, err)
	case http.StatusNotFound:
		return fmt.Errorf("%w: %w", ErrNotFound, err)
	case http.StatusRequestEntityTooLarge:
		return fmt.Errorf("%w: %w", ErrPayloadTooLarge, err)
	case http.StatusTooManyRequests:
		return fmt.Errorf("%w: %w", ErrRateLimited, err)
	default:
		if apiErr.StatusCode >= http.StatusInternalServerError {
			return fmt.Errorf("%w: %w", ErrServerError, err)
		}
		return err
	}
}
