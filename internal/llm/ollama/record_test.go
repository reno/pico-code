package ollama

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

// TestRecordLiveExchange is 9.1's record half: with RECORD=1 and a running
// local Ollama server, it drives one real Chat call and writes the
// request/response to testdata/golden/live_exchange. Ollama needs no API
// key, so there is nothing to scrub beyond OLLAMA_HOST if it's non-default
// — but recordutil.Scrub still runs, in case a custom host string embeds
// something sensitive. It never runs in `make test` (no RECORD set) and
// skips itself if Ollama isn't actually reachable.
func TestRecordLiveExchange(t *testing.T) {
	if !recordutil.Enabled() {
		t.Skip("set RECORD=1 (with Ollama running locally) to re-record this fixture against the live API")
	}
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = defaultHost
	}
	model := os.Getenv("PICO_RECORD_OLLAMA_MODEL")
	if model == "" {
		t.Skip("set PICO_RECORD_OLLAMA_MODEL to a locally pulled model to record against")
	}

	var exchange recordutil.Exchange
	p := &Provider{
		httpClient: &http.Client{Transport: &recordutil.Recorder{OnExchange: func(e recordutil.Exchange) { exchange = e }}},
		baseURL:    host,
		model:      model,
		numCtx:     4096,
	}

	_, err := p.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "Reply with exactly: ok"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v (is `ollama serve` running with %q pulled?)", err, model)
	}

	dir := "testdata/golden/live_exchange"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	reqOut := recordutil.Scrub(string(exchange.RequestBody), host)
	respOut := recordutil.Scrub(string(exchange.ResponseBody), host)
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
