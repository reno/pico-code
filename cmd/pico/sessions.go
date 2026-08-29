package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/reno/pico-code/internal/history"
)

// session tracks which on-disk file (if any) the current conversation
// should be saved to after each turn — set at startup by --session, and
// changeable at runtime by /save and /load. It also carries mcp, the MCP
// server lifecycle manager (13.3): both are per-chat-session mutable state
// threaded through every slash command the same way, so mcp lives here
// rather than widening every commandSpec.run signature for the one
// command that needs it. mcp is nil when no MCP servers are configured, or
// in a test session built without one — /mcp handles that explicitly.
type session struct {
	dir  string
	name string
	mcp  *mcpManager
}

// newSession returns a session rooted at dir (created if missing), named
// initially by name ("" for no active session).
func newSession(dir, name string) (*session, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("creating sessions directory %s: %w", dir, err)
	}
	return &session{dir: dir, name: name}, nil
}

// defaultSessionsDir is where sessions live absent any override; a real
// user's home directory in production, a test's t.TempDir() in tests.
func defaultSessionsDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolving home directory for sessions: %w", err)
	}
	return filepath.Join(home, ".pico-code", "sessions"), nil
}

func (s *session) path(name string) string {
	return filepath.Join(s.dir, name+".json")
}

// active reports whether a session is currently named (and so gets
// auto-saved after each turn).
func (s *session) active() bool {
	return s.name != ""
}

// loadOrCreateHistory resolves cfg's --session into a History: a fresh one
// if unnamed or the file doesn't exist yet, or the resumed one if it does.
// A corrupt session file surfaces as a wrapped error here rather than a
// panic (8.3's AC) — history.Load already guarantees that, this just
// preserves it up the call chain.
func (s *session) loadOrCreateHistory() (*history.History, error) {
	if !s.active() {
		return history.New(), nil
	}
	path := s.path(s.name)
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return history.New(), nil
	}
	h, err := history.Load(path)
	if err != nil {
		return nil, fmt.Errorf("resuming session %q: %w", s.name, err)
	}
	return h, nil
}

// saveIfActive persists h to the current session's file, a no-op if no
// session is active.
func (s *session) saveIfActive(h *history.History) error {
	if !s.active() {
		return nil
	}
	if err := h.Save(s.path(s.name)); err != nil {
		return fmt.Errorf("saving session %q: %w", s.name, err)
	}
	return nil
}

// recent returns up to n saved session names, most recently modified
// first. Errors are swallowed into an empty list: the home screen that
// calls this must render even when the sessions directory is unreadable.
func (s *session) recent(n int) []string {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil
	}
	type entry struct {
		name string
		mod  time.Time
	}
	var found []entry
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		found = append(found, entry{strings.TrimSuffix(e.Name(), ".json"), info.ModTime()})
	}
	sort.Slice(found, func(i, j int) bool { return found[i].mod.After(found[j].mod) })
	if len(found) > n {
		found = found[:n]
	}
	names := make([]string, 0, len(found))
	for _, e := range found {
		names = append(names, e.name)
	}
	return names
}
