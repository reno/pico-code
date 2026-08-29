package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

const notFoundBody = `{"type":"error","error":{"type":"not_found_error","message":"model not found"}}`

const modelInfoBody = `{
	"id": "claude-known-model", "type": "model",
	"display_name": "Known Model", "created_at": "2025-01-01T00:00:00Z",
	"max_input_tokens": 200000, "max_tokens": 8192,
	"capabilities": {}
}`

// TestValidateModelKnownAndUnknown is 11.3's AC: /model with a name the
// provider does not know errors. mapError already maps a 404 to
// ErrNotFound (errors_test.go), so this proves ValidateModel's request
// actually reaches the Models API and that mapError still applies to it.
func TestValidateModelKnownAndUnknown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models/claude-known-model" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(modelInfoBody))
			return
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(notFoundBody))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)

	if err := p.ValidateModel(context.Background(), "claude-known-model"); err != nil {
		t.Errorf("ValidateModel(known) error = %v, want nil", err)
	}
	if err := p.ValidateModel(context.Background(), "claude-unknown-model"); !errors.Is(err, ErrNotFound) {
		t.Errorf("ValidateModel(unknown) error = %v, want wrapping %v", err, ErrNotFound)
	}
}

// TestSetModelChangesTheModelChatSends is 11.3's AC that switching models
// actually takes effect: after SetModel, Chat's request targets the new
// model, not the one Provider was constructed with.
func TestSetModelChangesTheModelChatSends(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(okMessageBody))
	}))
	defer srv.Close()

	p := testProvider(srv.URL)
	p.SetModel("claude-new-model")

	if _, err := p.Chat(context.Background(), testChatRequest()); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if gotModel != "claude-new-model" {
		t.Errorf("request model = %q, want %q", gotModel, "claude-new-model")
	}
}

var _ llm.ModelSwitcher = (*Provider)(nil)
