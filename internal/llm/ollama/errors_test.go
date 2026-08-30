package ollama

import (
	"errors"
	"net/http"
	"testing"
)

func TestMapErrorNotFound(t *testing.T) {
	got := mapError(http.StatusNotFound, []byte(`{"error":"model 'ghost' not found, try pulling it first"}`))
	if !errors.Is(got, ErrNotFound) {
		t.Errorf("mapError(404) = %v, want wrapping %v", got, ErrNotFound)
	}
}

func TestMapErrorUnmappedStatusPassesThrough(t *testing.T) {
	got := mapError(http.StatusInternalServerError, []byte("boom"))
	if got == nil {
		t.Fatal("mapError() = nil, want an error for a 5xx status")
	}
	if errors.Is(got, ErrNotFound) {
		t.Errorf("mapError(500) wraps ErrNotFound, want an unmapped status to pass through unclassified")
	}
}
