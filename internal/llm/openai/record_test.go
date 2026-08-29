package openai

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/recordutil"
)

// TestRecordLiveExchange is 9.1's record half for this adapter: with
// RECORD=1, a real OPENAI_API_KEY (or another compatible endpoint's key),
// and PICO_RECORD_OPENAI_MODEL set, it drives one real Chat call and writes
// the scrubbed request/response to testdata/golden/live_exchange. It never
// runs in `make test` (no RECORD set), so the suite stays offline.
func TestRecordLiveExchange(t *testing.T) {
	if !recordutil.Enabled() {
		t.Skip("set RECORD=1 (with an OpenAI-compatible key) to re-record this fixture against the live API")
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	model := os.Getenv("PICO_RECORD_OPENAI_MODEL")
	if model == "" {
		t.Skip("set PICO_RECORD_OPENAI_MODEL to a model to record against")
	}
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	var exchange recordutil.Exchange
	p := &Provider{
		httpClient: &http.Client{Transport: &recordutil.Recorder{OnExchange: func(e recordutil.Exchange) { exchange = e }}},
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		apiKey:     apiKey,
		model:      model,
	}

	_, err := p.Chat(context.Background(), llm.Request{
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "Reply with exactly: ok"}}}},
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	dir := "testdata/golden/live_exchange"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	reqOut := recordutil.Scrub(string(exchange.RequestBody), apiKey)
	respOut := recordutil.Scrub(string(exchange.ResponseBody), apiKey)
	if err := os.WriteFile(filepath.Join(dir, "request.json"), []byte(reqOut), 0o644); err != nil {
		t.Fatalf("WriteFile(request) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "response.json"), []byte(respOut), 0o644); err != nil {
		t.Fatalf("WriteFile(response) error = %v", err)
	}
}

// TestGoldenFixturesContainNoSecrets is 9.1's AC: no key ever appears in a
// fixture.
func TestGoldenFixturesContainNoSecrets(t *testing.T) {
	err := filepath.WalkDir("testdata/golden", func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		if strings.Contains(string(data), "sk-") {
			t.Errorf("%s contains an sk-shaped key literal", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking testdata/golden: %v", err)
	}
}
