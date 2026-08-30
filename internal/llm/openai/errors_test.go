package openai

import (
	"errors"
	"net/http"
	"testing"
)

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
			got := mapError(tt.status, []byte("boom"))
			if !errors.Is(got, tt.want) {
				t.Errorf("mapError(%d) = %v, want wrapping %v", tt.status, got, tt.want)
			}
		})
	}
}

func TestMapErrorUnmappedStatusReportsRawBody(t *testing.T) {
	got := mapError(http.StatusTeapot, []byte("I'm a teapot"))
	if got == nil {
		t.Fatal("mapError() = nil, want an error for an unmapped 4xx status")
	}
	for _, sentinel := range []error{ErrUnauthorized, ErrForbidden, ErrNotFound, ErrPayloadTooLarge, ErrRateLimited, ErrServerError} {
		if errors.Is(got, sentinel) {
			t.Errorf("mapError(teapot) wraps %v, want none of the mapped sentinels", sentinel)
		}
	}
}
