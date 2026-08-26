package anthropic

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
)

// newAPIError builds an *sdk.Error with a real (non-nil) Request/Response,
// since Error() dereferences both — a zero-value Error, as a caller might
// naively construct in a test, panics on .Error().
func newAPIError(t *testing.T, status int) error {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	res := &http.Response{StatusCode: status, Request: req}
	return &sdk.Error{StatusCode: status, Request: req, Response: res}
}

func TestMapErrorStatusCodes(t *testing.T) {
	tests := []struct {
		status int
		want   error
	}{
		{http.StatusUnauthorized, ErrUnauthorized},
		{http.StatusForbidden, ErrForbidden},
		{http.StatusNotFound, ErrNotFound},
		{http.StatusRequestEntityTooLarge, ErrPayloadTooLarge},
		{http.StatusTooManyRequests, ErrRateLimited},
		{http.StatusInternalServerError, ErrServerError},
		{http.StatusBadGateway, ErrServerError},
		{http.StatusServiceUnavailable, ErrServerError},
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			apiErr := newAPIError(t, tt.status)
			got := mapError(apiErr)
			if !errors.Is(got, tt.want) {
				t.Errorf("mapError(status %d) = %v, want wrapping %v", tt.status, got, tt.want)
			}
			if !errors.Is(got, apiErr) {
				t.Errorf("mapError(status %d) = %v, want it to still wrap the original error", tt.status, got)
			}
		})
	}
}

func TestMapErrorUnmappedStatusPassesThrough(t *testing.T) {
	apiErr := newAPIError(t, http.StatusTeapot)
	got := mapError(apiErr)
	if got != apiErr {
		t.Errorf("mapError() = %v, want the original error unchanged for an unmapped status", got)
	}
}

func TestMapErrorNonAPIErrorPassesThrough(t *testing.T) {
	sentinel := errors.New("boom")
	if got := mapError(sentinel); got != sentinel {
		t.Errorf("mapError() = %v, want the original error unchanged for a non-API error", got)
	}
}

func TestMapErrorNil(t *testing.T) {
	if got := mapError(nil); got != nil {
		t.Errorf("mapError(nil) = %v, want nil", got)
	}
}
