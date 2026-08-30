package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"

	"github.com/ollama/ollama/api"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/recordutil"
)

func init() {
	llm.Register(string(config.ProviderOllama), New)
}

const defaultHost = "http://localhost:11434"

// Provider adapts Ollama's /api/chat to llm.Provider. It speaks HTTP
// directly with net/http rather than through api.Client: the client's own
// JSON decoding rejects a tool call whose arguments arrived as a
// JSON-encoded string (see normalize.go), so this package needs the raw
// response bytes to fix that up before decoding into api.ChatResponse.
type Provider struct {
	httpClient *http.Client
	baseURL    string
	model      string
	numCtx     int

	toolSupportOnce sync.Once
	toolSupport     bool
	toolSupportErr  error
}

// New constructs a Provider from resolved config. It is registered under
// "ollama" in init, so llm.New can find it once this package has been
// imported (a blank import is enough to trigger registration).
func New(cfg *config.Config) (llm.Provider, error) {
	host := cfg.OllamaHost
	if host == "" {
		host = defaultHost
	}
	numCtx := cfg.NumCtx
	if numCtx <= 0 {
		numCtx = 4096
	}
	slog.Info("ollama: context window configured", "num_ctx", numCtx, "model", cfg.Model)

	return &Provider{
		httpClient: http.DefaultClient,
		baseURL:    host,
		model:      cfg.Model,
		numCtx:     numCtx,
	}, nil
}

// Name identifies this provider in logs and config.
func (p *Provider) Name() string { return "ollama" }

// Chat sends req to /api/chat with streaming disabled and translates the
// response back to canonical form.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if err := p.checkToolSupport(ctx, req); err != nil {
		return nil, err
	}

	ar, err := toRequest(req, p.model, p.numCtx)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	body, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	recordutil.LogBytes(ctx, "ollama: request", body)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	defer func() { _ = res.Body.Close() }()

	respBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama: read response: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		return nil, mapError(res.StatusCode, respBody)
	}
	recordutil.LogBytes(ctx, "ollama: response", respBody)

	normalized, err := normalizeToolCallArguments(respBody)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}

	var chatResp api.ChatResponse
	if err := json.Unmarshal(normalized, &chatResp); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}

	return fromResponse(req, &chatResp)
}
