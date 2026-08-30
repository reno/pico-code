package main

import (
	"errors"
	"fmt"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/anthropic"
	"github.com/reno/pico-code/internal/llm/ollama"
	"github.com/reno/pico-code/internal/llm/openai"
)

// friendlyAgentError turns a provider- or model-resolution failure into a
// one-line message naming the actual problem instead of the raw HTTP status
// dump or SDK error string a provider adapter produces. --model isn't
// validated at startup (unlike --provider, checked by config.Load's closed
// enum), so a typo'd model name reaches the provider on the first real
// request and, without this, would print a wall of transport detail and
// (in plain mode) kill the whole session. Anything else — a transport
// error, a rate limit, a tool failure — passes through unchanged: those
// already carry their own useful detail.
func friendlyAgentError(cfg *config.Config, err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, llm.ErrProviderNotRegistered) {
		return fmt.Errorf("provider %q is not available in this build (its adapter isn't registered)", cfg.Provider)
	}

	var sentinel error
	switch {
	case errors.Is(err, anthropic.ErrNotFound):
		sentinel = anthropic.ErrNotFound
	case errors.Is(err, openai.ErrNotFound):
		sentinel = openai.ErrNotFound
	case errors.Is(err, ollama.ErrNotFound):
		sentinel = ollama.ErrNotFound
	default:
		return err
	}
	return fmt.Errorf("model %q was not found for provider %q — check the model name and try again (%w)", cfg.Model, cfg.Provider, sentinel)
}
