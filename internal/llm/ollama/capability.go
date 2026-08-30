package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/reno/pico-code/internal/llm"
)

// ErrToolsNotSupported is returned when the configured model has no native
// tool-calling capability. Sending tool definitions to such a model anyway
// tends to produce exactly the misbehavior CLAUDE.md warns small local
// models exhibit (invented names, prose instead of a call, ...), so Chat
// fails fast here instead, pointing at the documented escape hatch.
var ErrToolsNotSupported = errors.New("ollama: model has no native tool support; retry with --tools=prompted")

// supportsTools reports whether p's model advertises the "tools"
// capability, probing /api/show at most once per Provider: the result is
// cached in p for the process's lifetime (a Provider is constructed once at
// startup and lives for the whole session), so a request that includes
// tools never pays for a second /api/show round trip.
func (p *Provider) supportsTools(ctx context.Context) (bool, error) {
	p.toolSupportOnce.Do(func() {
		p.toolSupport, p.toolSupportErr = p.probeToolSupport(ctx)
	})
	return p.toolSupport, p.toolSupportErr
}

func (p *Provider) probeToolSupport(ctx context.Context) (bool, error) {
	show, err := p.probeShow(ctx, p.model)
	if err != nil {
		return false, err
	}
	for _, c := range show.Capabilities {
		if c == model.CapabilityTools {
			return true, nil
		}
	}
	return false, nil
}

// probeShow calls /api/show for modelName, the low-level request both
// probeToolSupport (for p.model, cached via supportsTools) and
// ValidateModel (for an arbitrary candidate, uncached) build on. A model
// /api/show doesn't recognize comes back as a 404, mapped to ErrNotFound.
func (p *Provider) probeShow(ctx context.Context, modelName string) (api.ShowResponse, error) {
	body, err := json.Marshal(api.ShowRequest{Model: modelName})
	if err != nil {
		return api.ShowResponse{}, fmt.Errorf("ollama: marshal /api/show request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return api.ShowResponse{}, fmt.Errorf("ollama: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return api.ShowResponse{}, fmt.Errorf("%w: %w", ErrUnreachable, err)
	}
	defer func() { _ = res.Body.Close() }()

	respBody, err := io.ReadAll(res.Body)
	if err != nil {
		return api.ShowResponse{}, fmt.Errorf("ollama: read /api/show response: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		return api.ShowResponse{}, mapError(res.StatusCode, respBody)
	}

	var show api.ShowResponse
	if err := json.Unmarshal(respBody, &show); err != nil {
		return api.ShowResponse{}, fmt.Errorf("ollama: decode /api/show response: %w", err)
	}
	return show, nil
}

// ValidateModel implements llm.ModelSwitcher by probing /api/show for
// modelName. Ollama's /api/show has no context-window field that's stable
// across model families (it's buried in a per-family key inside
// ModelInfo), so unlike Anthropic's ValidateModel this never derives one;
// num_ctx stays the explicit, user-set knob CLAUDE.md already requires it
// to be.
func (p *Provider) ValidateModel(ctx context.Context, modelName string) error {
	_, err := p.probeShow(ctx, modelName)
	return err
}

// SetModel implements llm.ModelSwitcher.
func (p *Provider) SetModel(modelName string) {
	p.model = modelName
}

var _ llm.ModelSwitcher = (*Provider)(nil)

// checkToolSupport gates a request that carries tool definitions: it is a
// no-op for a tool-less request (the common case, and the only case in
// --tools=prompted mode, where the loop never populates Request.Tools).
func (p *Provider) checkToolSupport(ctx context.Context, req llm.Request) error {
	if len(req.Tools) == 0 {
		return nil
	}
	supported, err := p.supportsTools(ctx)
	if err != nil {
		return fmt.Errorf("ollama: checking tool support: %w", err)
	}
	if !supported {
		return fmt.Errorf("%w (model %q)", ErrToolsNotSupported, p.model)
	}
	return nil
}
