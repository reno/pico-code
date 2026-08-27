package anthropic

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/recordutil"
)

// TestRecordLiveExchange is 9.1's record half: with RECORD=1 and a real
// ANTHROPIC_API_KEY, it drives one real Chat call and writes the scrubbed
// request/response to testdata/golden/live_exchange, the fixture a
// developer re-recording this adapter's behavior would inspect. It never
// runs in `make test` (no RECORD, no key, no network) — only this
// repository's maintainer running it manually can actually record.
func TestRecordLiveExchange(t *testing.T) {
	if !recordutil.Enabled() {
		t.Skip("set RECORD=1 (with ANTHROPIC_API_KEY) to re-record this fixture against the live API")
	}
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("RECORD=1 requires ANTHROPIC_API_KEY")
	}

	var exchange recordutil.Exchange
	recordingClient := &http.Client{Transport: &recordutil.Recorder{
		OnExchange: func(e recordutil.Exchange) { exchange = e },
	}}
	client := sdk.NewClient(
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(recordingClient),
	)
	p := &Provider{client: client, model: string(sdk.ModelClaudeHaiku4_5)}

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
// fixture. It walks every checked-in fixture (recorded or hand-written)
// and greps for an sk-shaped literal and the actual ANTHROPIC_API_KEY value
// if one happens to be set in this environment.
func TestGoldenFixturesContainNoSecrets(t *testing.T) {
	assertNoSecretsUnder(t, "testdata/golden")
}

func assertNoSecretsUnder(t *testing.T, root string) {
	t.Helper()
	liveKey := os.Getenv("ANTHROPIC_API_KEY")

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		content := string(data)
		if strings.Contains(content, "sk-") {
			t.Errorf("%s contains an sk-shaped key literal", path)
		}
		if liveKey != "" && strings.Contains(content, liveKey) {
			t.Errorf("%s contains the live ANTHROPIC_API_KEY value", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking %s: %v", root, err)
	}
}
