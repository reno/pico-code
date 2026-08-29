// Package openai adapts an OpenAI-compatible /chat/completions endpoint to
// the canonical llm.Provider interface. "OpenAI-compatible" covers more
// than OpenAI itself: vLLM, LM Studio, and Ollama's own /v1 endpoint all
// speak the same flat tool_calls shape, so BaseURL (and an optional APIKey)
// are all resolved config needs to point this adapter at any of them.
// There is no OpenAI SDK on CLAUDE.md's dependency allowlist, so this
// package speaks HTTP directly with net/http, the same approach the Ollama
// adapter takes for its own native endpoint.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/recordutil"
)

func init() {
	llm.Register(string(config.ProviderOpenAI), New)
}

const defaultBaseURL = "https://api.openai.com/v1"

// Provider adapts an OpenAI-compatible /chat/completions endpoint to
// llm.Provider.
type Provider struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
	model      string
}

// New constructs a Provider from resolved config. It is registered under
// "openai" in init, so llm.New can find it once this package has been
// imported (a blank import is enough to trigger registration). Unlike the
// Anthropic adapter, an empty API key is not an error: many
// OpenAI-compatible endpoints (vLLM, LM Studio, a local Ollama /v1) run
// with no auth at all.
func New(cfg *config.Config) (llm.Provider, error) {
	baseURL := cfg.OpenAIBaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		httpClient: http.DefaultClient,
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		apiKey:     cfg.OpenAIAPIKey,
		model:      cfg.Model,
	}, nil
}

// Name identifies this provider in logs and config.
func (p *Provider) Name() string { return "openai" }

// Chat sends req to POST {baseURL}/chat/completions with streaming
// disabled and translates the response back to canonical form.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	cr, err := toRequest(req, p.model)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	body, err := json.Marshal(cr)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	recordutil.LogBytes(ctx, "openai: request", body, p.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer func() { _ = res.Body.Close() }()

	respBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		return nil, mapError(res.StatusCode, respBody)
	}
	recordutil.LogBytes(ctx, "openai: response", respBody, p.apiKey)

	var cresp chatResponse
	if err := json.Unmarshal(respBody, &cresp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	return fromResponse(&cresp)
}
