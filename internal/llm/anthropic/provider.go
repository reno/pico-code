package anthropic

import (
	"context"
	"errors"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
)

func init() {
	llm.Register(string(config.ProviderAnthropic), New)
}

// Provider adapts the Anthropic Messages API to llm.Provider.
type Provider struct {
	client sdk.Client
	model  string
}

// New constructs a Provider from resolved config. It is registered under
// "anthropic" in init, so llm.New can find it once this package has been
// imported (a blank import is enough to trigger registration).
func New(cfg *config.Config) (llm.Provider, error) {
	if cfg.AnthropicAPIKey == "" {
		return nil, errors.New("anthropic: ANTHROPIC_API_KEY is not set")
	}
	return &Provider{
		client: sdk.NewClient(option.WithAPIKey(cfg.AnthropicAPIKey)),
		model:  cfg.Model,
	}, nil
}

// Name identifies this provider in logs and config.
func (p *Provider) Name() string { return "anthropic" }

// Chat sends req and translates the response back to canonical form. Error
// mapping and retries land in phase 2.2; this only wraps translation
// failures and the SDK's own error with an "anthropic:" prefix so the loop
// can tell which layer failed.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	params, err := toParams(req, p.model)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	return fromResponse(msg)
}

// Stream is implemented in phase 6.2.
func (p *Provider) Stream(_ context.Context, _ llm.Request) (<-chan llm.Event, error) {
	return nil, errors.New("anthropic: streaming not implemented until phase 6.2")
}
