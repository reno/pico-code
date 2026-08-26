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
	body, err := json.Marshal(api.ShowRequest{Model: p.model})
	if err != nil {
		return false, fmt.Errorf("ollama: marshal /api/show request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return false, fmt.Errorf("ollama: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := p.httpClient.Do(httpReq)
	if err != nil {
		return false, fmt.Errorf("ollama: %w", err)
	}
	defer func() { _ = res.Body.Close() }()

	respBody, err := io.ReadAll(res.Body)
	if err != nil {
		return false, fmt.Errorf("ollama: read /api/show response: %w", err)
	}
	if res.StatusCode >= http.StatusBadRequest {
		return false, fmt.Errorf("ollama: /api/show failed with status %d: %s", res.StatusCode, bytes.TrimSpace(respBody))
	}

	var show api.ShowResponse
	if err := json.Unmarshal(respBody, &show); err != nil {
		return false, fmt.Errorf("ollama: decode /api/show response: %w", err)
	}

	for _, c := range show.Capabilities {
		if c == model.CapabilityTools {
			return true, nil
		}
	}
	return false, nil
}

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
