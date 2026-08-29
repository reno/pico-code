package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/reno/pico-code/internal/mcp"
	"github.com/reno/pico-code/internal/tools"
)

// mcpDiscoverTimeout bounds both the initial connection to every
// configured server and a single /mcp reconnect attempt — the same value
// 13.1's Discover uses to guarantee a server that never answers can't hold
// up chat startup (or a reconnect) past this long.
const mcpDiscoverTimeout = 5 * time.Second

// mcpConfigFile is the on-disk shape --mcp-config points at:
// {"mcpServers": {name: {command, args, env}}}, the same convention most
// existing MCP clients (Claude Desktop among them) already use, so an
// existing server list is reusable here as-is.
type mcpConfigFile struct {
	MCPServers map[string]struct {
		Command string            `json:"command"`
		Args    []string          `json:"args"`
		Env     map[string]string `json:"env"`
	} `json:"mcpServers"`
}

// loadMCPServers parses path (empty means "no servers configured") into
// mcp.ServerConfigs, sorted by name so discovery and /mcp's listing are
// deterministic across runs regardless of the file's own key order (Go's
// map iteration order is randomized).
func loadMCPServers(path string) ([]mcp.ServerConfig, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading --mcp-config %s: %w", path, err)
	}
	var file mcpConfigFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parsing --mcp-config %s: %w", path, err)
	}

	names := make([]string, 0, len(file.MCPServers))
	for name := range file.MCPServers {
		names = append(names, name)
	}
	sort.Strings(names)

	servers := make([]mcp.ServerConfig, 0, len(names))
	for _, name := range names {
		s := file.MCPServers[name]
		env := make([]string, 0, len(s.Env))
		for k, v := range s.Env {
			env = append(env, k+"="+v)
		}
		sort.Strings(env)
		servers = append(servers, mcp.ServerConfig{Name: name, Command: s.Command, Args: s.Args, Env: env})
	}
	return servers, nil
}

// mcpServerState is one configured server's current standing.
type mcpServerState struct {
	cfg    mcp.ServerConfig
	client *mcp.Client
	tools  []mcp.ToolInfo
	err    error
}

// mcpStatus is a point-in-time, lock-free snapshot of one server, for /mcp
// to render without holding mcpManager's mutex while writing to out.
type mcpStatus struct {
	name      string
	connected bool
	toolCount int
	err       error
}

// mcpManager owns the lifecycle of every MCP server configured for one
// chat session: initial discovery, namespaced registration of each
// server's tools (13.2) into a shared Registry, on-demand reconnect, and
// clean shutdown of every child process on exit.
type mcpManager struct {
	registry *tools.Registry
	timeout  time.Duration

	mu      sync.Mutex
	servers []*mcpServerState // stable, cfgs order — deterministic /mcp output
}

// newMCPManager discovers every server in cfgs (mcp.Discover, 13.1) and
// registers each connected server's tools into registry (13.2). A server
// that fails to connect, or whose tools fail to register, is recorded with
// its error rather than treated as fatal — startup always reaches the
// prompt regardless of how many configured servers are broken.
func newMCPManager(ctx context.Context, registry *tools.Registry, cfgs []mcp.ServerConfig, timeout time.Duration) *mcpManager {
	m := &mcpManager{registry: registry, timeout: timeout}
	if len(cfgs) == 0 {
		return m
	}
	for _, r := range mcp.Discover(ctx, cfgs, timeout) {
		state := &mcpServerState{cfg: r.Config, client: r.Client, tools: r.Tools, err: r.Err}
		m.servers = append(m.servers, state)
		if err := m.registerTools(state); err != nil {
			slog.WarnContext(ctx, "mcp: tool registration failed", "server", state.cfg.Name, "error", err)
		}
	}
	return m
}

// registerTools registers every tool state advertises, namespaced under
// state.cfg.Name, returning the first registration error (a genuine name
// collision — reconnect always unregisters this same server's own entries
// first, so a collision here means something else already claimed the
// name) while still attempting the rest, so one bad tool name doesn't hide
// problems with the others.
func (m *mcpManager) registerTools(state *mcpServerState) error {
	if state.client == nil {
		return nil
	}
	var firstErr error
	for _, ti := range state.tools {
		if err := m.registry.Register(tools.NewMCPTool(state.client, state.cfg.Name, ti)); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// find returns the server state configured under name, or nil.
func (m *mcpManager) find(name string) *mcpServerState {
	for _, s := range m.servers {
		if s.cfg.Name == name {
			return s
		}
	}
	return nil
}

// disconnect closes state's live client (if any) and unregisters every
// tool it contributed, leaving state ready for a fresh registerTools call.
func (m *mcpManager) disconnect(state *mcpServerState) {
	if state.client == nil {
		return
	}
	for _, ti := range state.tools {
		m.registry.Unregister(tools.MCPToolName(state.cfg.Name, ti.Name))
	}
	_ = state.client.Close()
	state.client = nil
	state.tools = nil
}

// reconnect re-runs discovery for the single configured server named name,
// replacing whatever it previously contributed to the Registry with
// whatever the fresh connection advertises. It errors if name isn't a
// configured server at all; a server that connects but fails to hand
// shake again is still reflected as failed in statuses(), the same way
// newMCPManager reports an initial discovery failure, rather than
// surfaced as reconnect's own error.
func (m *mcpManager) reconnect(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	state := m.find(name)
	if state == nil {
		return fmt.Errorf("mcp: no configured server named %q", name)
	}

	m.disconnect(state)

	results := mcp.Discover(ctx, []mcp.ServerConfig{state.cfg}, m.timeout)
	r := results[0]
	state.client, state.tools, state.err = r.Client, r.Tools, r.Err
	return m.registerTools(state)
}

// statuses snapshots every configured server's current standing, in
// configuration order.
func (m *mcpManager) statuses() []mcpStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]mcpStatus, len(m.servers))
	for i, s := range m.servers {
		out[i] = mcpStatus{name: s.cfg.Name, connected: s.client != nil, toolCount: len(s.tools), err: s.err}
	}
	return out
}

// shutdown closes every connected server's Client, terminating its child
// process. Meant to run once, via defer, right after the manager is
// built — the one chokepoint every chat exit path (REPL EOF, /exit, TUI
// Ctrl+D quit, ctx cancellation unwinding back up) passes through, so a
// clean exit never leaves a child process behind. A double-Ctrl+C
// force-exit skips this (CLAUDE.md 0.3: it calls os.Exit directly, no
// defers run) but doesn't orphan anything either — Client.Start ties each
// process to the same shutdown ctx via exec.CommandContext, which already
// kills every child the instant that ctx is cancelled, on the first
// Ctrl+C.
func (m *mcpManager) shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, s := range m.servers {
		if s.client != nil {
			_ = s.client.Close()
		}
	}
}
