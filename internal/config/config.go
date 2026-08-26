// Package config resolves CLI flags and environment variables into a
// validated Config that the rest of pico code depends on.
package config

import (
	"errors"
	"fmt"
)

// Provider identifies an LLM backend.
type Provider string

// Supported providers.
const (
	ProviderAnthropic Provider = "anthropic"
	ProviderOllama    Provider = "ollama"
)

// ToolsMode selects how tool calls are surfaced to the model.
type ToolsMode string

// Supported tool-calling modes.
const (
	ToolsNative   ToolsMode = "native"
	ToolsPrompted ToolsMode = "prompted"
)

// ErrUnknownProvider is returned when --provider names a backend pico code
// does not implement.
var ErrUnknownProvider = errors.New("unknown provider")

// ErrUnknownToolsMode is returned when --tools names a mode other than
// native or prompted.
var ErrUnknownToolsMode = errors.New("unknown tools mode")

// Flags carries the values collected from the chat subcommand's flags,
// already resolved against any environment fallback (e.g. PICO_CODE_PROVIDER)
// by the caller. Load only validates and fills in credentials.
type Flags struct {
	Provider    string
	Model       string
	MaxTurns    int
	TokenBudget int
	Workspace   string
	Yes         bool
	Tools       string
	Stream      bool
	TUI         bool
	LogLevel    string
	NumCtx      int
}

// Config is the fully resolved, validated configuration for a chat session.
type Config struct {
	Provider    Provider
	Model       string
	MaxTurns    int
	TokenBudget int
	Workspace   string
	Yes         bool
	Tools       ToolsMode
	Stream      bool
	TUI         bool
	LogLevel    string

	// NumCtx is the Ollama adapter's context window size (num_ctx). Ollama
	// silently truncates context if this is left unset, so it always has an
	// explicit value rather than an optional one the adapter might omit.
	NumCtx int

	// AnthropicAPIKey and OllamaHost come from the environment only; there
	// is no flag for either, since committing a secret to a shell history
	// is worse than typing it once into the environment.
	AnthropicAPIKey string
	OllamaHost      string
}

// Load validates f and merges in provider credentials read via getenv.
// getenv is injected so tests do not depend on process environment.
func Load(f Flags, getenv func(string) string) (*Config, error) {
	provider := Provider(f.Provider)
	switch provider {
	case ProviderAnthropic, ProviderOllama:
	default:
		return nil, fmt.Errorf("%w: %q (want %q or %q)", ErrUnknownProvider, f.Provider, ProviderAnthropic, ProviderOllama)
	}

	tools := ToolsMode(f.Tools)
	switch tools {
	case ToolsNative, ToolsPrompted:
	default:
		return nil, fmt.Errorf("%w: %q (want %q or %q)", ErrUnknownToolsMode, f.Tools, ToolsNative, ToolsPrompted)
	}

	return &Config{
		Provider:        provider,
		Model:           f.Model,
		MaxTurns:        f.MaxTurns,
		TokenBudget:     f.TokenBudget,
		Workspace:       f.Workspace,
		Yes:             f.Yes,
		Tools:           tools,
		Stream:          f.Stream,
		TUI:             f.TUI,
		LogLevel:        f.LogLevel,
		NumCtx:          f.NumCtx,
		AnthropicAPIKey: getenv("ANTHROPIC_API_KEY"),
		OllamaHost:      getenv("OLLAMA_HOST"),
	}, nil
}
