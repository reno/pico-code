package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/reno/pico-code/internal/config"
)

// Provider is the minimal contract every LLM backend implements. Anything
// provider-specific (thinking blocks, num_ctx, cache control) is configured
// through Request or an adapter-local option, never by widening this
// interface for one backend — CLAUDE.md invariant 2.
type Provider interface {
	Name() string
	Chat(ctx context.Context, req Request) (*Response, error)
	Stream(ctx context.Context, req Request) (<-chan Event, error)
}

// Factory constructs a Provider from resolved config. Adapters register one
// under their name via Register, typically from an init() func.
type Factory func(cfg *config.Config) (Provider, error)

// ErrProviderNotRegistered is returned by New when cfg.Provider names a
// backend whose adapter package was never imported (and so never called
// Register).
var ErrProviderNotRegistered = errors.New("llm: provider not registered")

// registry is the one sanctioned exception to "no package-level mutable
// state outside cmd/" (see CLAUDE.md). Adapter subpackages
// (internal/llm/anthropic, internal/llm/ollama) must import internal/llm for
// the canonical types, so internal/llm cannot import them back to build a
// switch statement without a cycle. Self-registration — the same pattern
// database/sql uses — is the standard way around that.
var (
	registryMu sync.RWMutex
	registry   = map[string]Factory{}
)

// Register makes an adapter's factory available under name, for New to find.
// It panics on a duplicate name: that can only happen from a programming
// error (two adapters claiming the same provider name), never from user
// input, so failing loudly at init time beats a silent overwrite.
func Register(name string, factory Factory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("llm: provider %q already registered", name))
	}
	registry[name] = factory
}

// New returns the provider configured by cfg.Provider. It fails if no
// adapter registered under that name — most likely because the caller
// forgot to blank-import the adapter package.
func New(cfg *config.Config) (Provider, error) {
	registryMu.RLock()
	factory, ok := registry[string(cfg.Provider)]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrProviderNotRegistered, cfg.Provider)
	}
	return factory(cfg)
}
