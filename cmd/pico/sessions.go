package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/reno/pico-code/internal/history"
)

// session tracks which on-disk file (if any) the current conversation
// should be saved to after each turn — set at startup by --session, and
// changeable at runtime by /save and /load.
type session struct {
	dir  string
	name string
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
