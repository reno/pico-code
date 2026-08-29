package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/reno/pico-code/internal/llm"
)

// TestValidateModelKnownAndUnknown is 11.3's AC: /model with a name the
// provider does not know errors. A model /api/show has never pulled comes
// back as a non-2xx status (a real Ollama server returns 404), which
// ValidateModel surfaces as a plain error.
func TestValidateModelKnownAndUnknown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		if body.Model == "known-model" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"capabilities": ["completion"]}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "known-model"}

	if err := p.ValidateModel(context.Background(), "known-model"); err != nil {
		t.Errorf("ValidateModel(known) error = %v, want nil", err)
	}
	if err := p.ValidateModel(context.Background(), "unknown-model"); err == nil {
		t.Error("ValidateModel(unknown) error = nil, want an error for a model /api/show doesn't recognize")
	}
}

// TestSetModelChangesTheModelChatSends mirrors the Anthropic adapter's
// equivalent test: after SetModel, Chat's request targets the new model.
func TestSetModelChangesTheModelChatSends(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "new-model",
			"message": {"role": "assistant", "content": "hi"},
			"done": true,
			"done_reason": "stop"
		}`))
	}))
	defer srv.Close()

	p := &Provider{httpClient: http.DefaultClient, baseURL: srv.URL, model: "old-model", numCtx: 8192}
	p.SetModel("new-model")

	if _, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hi"}}}},
	}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if gotModel != "new-model" {
		t.Errorf("request model = %q, want %q", gotModel, "new-model")
	}
}

var _ llm.ModelSwitcher = (*Provider)(nil)
