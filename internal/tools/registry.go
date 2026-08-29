package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/reno/pico-code/internal/llm"
)

// ErrToolNotFound is returned when a name has no registered Tool.
var ErrToolNotFound = errors.New("tools: tool not found")

// ErrDuplicateTool is returned by Register when name is already taken.
var ErrDuplicateTool = errors.New("tools: tool already registered")

// Registry holds the tools available in a session. Unlike internal/llm's
// provider registry, this is an instance, not package-level state: built-in
// tools (phase 5) are constructed with runtime config (a sandbox root, an
// allowlist) that an init()-time self-registration can't supply, so callers
// build a Registry explicitly instead.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{tools: map[string]Tool{}}
}

// Register adds t under t.Name(). It errors on a duplicate name rather than
// silently overwriting, since two tools claiming the same name is always a
// wiring mistake.
func (r *Registry) Register(t Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	name := t.Name()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("%w: %q", ErrDuplicateTool, name)
	}
	r.tools[name] = t
	return nil
}

// Unregister removes the tool registered under name, if any — a no-op
// otherwise, since a caller reconnecting a server that never successfully
// registered any tools shouldn't have to check first. Exists for a
// server's own lifecycle owner (reconnect, teardown) to clear its stale
// entries before registering fresh ones; nothing in the built-in tools
// needs it.
func (r *Registry) Unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.tools, name)
}

// Get returns the tool registered under name.
func (r *Registry) Get(name string) (Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrToolNotFound, name)
	}
	return t, nil
}

// Definitions returns the canonical llm.ToolDefinition for every registered
// tool, sorted by name so the request payload a caller builds from it is
// deterministic across runs.
func (r *Registry) Definitions() []llm.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := make([]llm.ToolDefinition, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, llm.ToolDefinition{
			Name:        t.Name(),
			Description: t.Description(),
			InputSchema: t.Schema(),
		})
	}
	sort.Slice(defs, func(i, j int) bool { return defs[i].Name < defs[j].Name })
	return defs
}

// Run looks up name, validates input against its Schema(), and only then
// calls Run — so a tool implementation never has to defend against a
// malformed or incomplete argument object.
func (r *Registry) Run(ctx context.Context, name string, input json.RawMessage) (string, error) {
	t, err := r.Get(name)
	if err != nil {
		return "", err
	}
	if err := validateInput(t.Schema(), input); err != nil {
		return "", fmt.Errorf("tools: invalid input for %q: %w", name, err)
	}
	return t.Run(ctx, input)
}
