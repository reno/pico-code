package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/anthropic"
	"github.com/reno/pico-code/internal/llm/ollama"
	"github.com/reno/pico-code/internal/llm/openai"
)

func TestFriendlyAgentErrorNil(t *testing.T) {
	if err := friendlyAgentError(&config.Config{}, nil); err != nil {
		t.Errorf("friendlyAgentError(nil) = %v, want nil", err)
	}
}

func TestFriendlyAgentErrorProviderNotRegistered(t *testing.T) {
	cfg := &config.Config{Provider: "bogus"}
	raw := fmt.Errorf("%w: %q", llm.ErrProviderNotRegistered, cfg.Provider)

	got := friendlyAgentError(cfg, raw)

	if !strings.Contains(got.Error(), "bogus") {
		t.Errorf("friendlyAgentError() = %q, want it to name the provider", got.Error())
	}
	if strings.Contains(got.Error(), "llm: provider not registered") {
		t.Errorf("friendlyAgentError() = %q, want the internal sentinel text stripped", got.Error())
	}
}

// TestFriendlyAgentErrorModelNotFound covers the case config.Load can't
// catch (--model has no startup validation): a typo'd model name that only
// fails on the first real request, deep inside a provider adapter, wrapped
// through internal/agent's "agent: chat: %w" / "agent: stream: %w" prefix
// before it ever reaches here.
func TestFriendlyAgentErrorModelNotFound(t *testing.T) {
	tests := []struct {
		name string
		raw  error
	}{
		{"anthropic", fmt.Errorf("agent: chat: %w: %w", anthropic.ErrNotFound, errors.New("<html>404 raw body dump</html>"))},
		{"openai", fmt.Errorf("agent: chat: %w: %w", openai.ErrNotFound, errors.New(`{"error":"raw json body"}`))},
		{"ollama", fmt.Errorf("agent: stream: %w: %w", ollama.ErrNotFound, errors.New(`{"error":"model not found"}`))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Provider: config.Provider(tt.name), Model: "ghost-model"}

			got := friendlyAgentError(cfg, tt.raw)

			if !strings.Contains(got.Error(), "ghost-model") {
				t.Errorf("friendlyAgentError() = %q, want it to name the model", got.Error())
			}
			if strings.Contains(got.Error(), "raw") {
				t.Errorf("friendlyAgentError() = %q, want the raw transport dump stripped", got.Error())
			}
			if !errors.Is(got, anthropic.ErrNotFound) && !errors.Is(got, openai.ErrNotFound) && !errors.Is(got, ollama.ErrNotFound) {
				t.Errorf("friendlyAgentError() = %v, want it to still wrap the matched not-found sentinel", got)
			}
		})
	}
}

// TestFriendlyAgentErrorOllamaUnreachable covers a local Ollama daemon that
// was never started: p.httpClient.Do fails outright (dial tcp ... connect:
// connection refused), wrapped through internal/agent's "agent: stream: %w"
// and ollama's own "checking tool support: %w" before reaching here.
func TestFriendlyAgentErrorOllamaUnreachable(t *testing.T) {
	dialErr := errors.New(`Post "http://localhost:11434/api/show": dial tcp [::1]:11434: connect: connection refused`)
	raw := fmt.Errorf("agent: stream: ollama: checking tool support: %w: %w", ollama.ErrUnreachable, dialErr)

	t.Run("default host", func(t *testing.T) {
		cfg := &config.Config{Provider: "ollama", Model: "qwen3:8b"}
		got := friendlyAgentError(cfg, raw)

		if !strings.Contains(got.Error(), ollama.DefaultHost) {
			t.Errorf("friendlyAgentError() = %q, want it to name %q", got.Error(), ollama.DefaultHost)
		}
		if strings.Contains(got.Error(), "dial tcp") {
			t.Errorf("friendlyAgentError() = %q, want the raw dial error stripped", got.Error())
		}
		if !errors.Is(got, ollama.ErrUnreachable) {
			t.Errorf("friendlyAgentError() = %v, want it to still wrap %v", got, ollama.ErrUnreachable)
		}
	})

	t.Run("configured host", func(t *testing.T) {
		cfg := &config.Config{Provider: "ollama", Model: "qwen3:8b", OllamaHost: "http://gpu-box:11434"}
		got := friendlyAgentError(cfg, raw)

		if !strings.Contains(got.Error(), "http://gpu-box:11434") {
			t.Errorf("friendlyAgentError() = %q, want it to name the configured OLLAMA_HOST", got.Error())
		}
	})
}

func TestFriendlyAgentErrorPassesThroughUnrecognized(t *testing.T) {
	cfg := &config.Config{Provider: "anthropic", Model: "claude"}
	raw := errors.New("network is down")

	got := friendlyAgentError(cfg, raw)

	if got != raw {
		t.Errorf("friendlyAgentError() = %v, want the original error passed through unchanged", got)
	}
}
