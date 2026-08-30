package tools

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ErrPathEscapesSandbox is returned by Sandbox.Resolve for a path that
// falls outside the workspace root, directly or via a symlink.
var ErrPathEscapesSandbox = errors.New("tools: path escapes workspace sandbox")

// ErrPathDenied is returned by Sandbox.Resolve for a path matching a deny
// glob.
var ErrPathDenied = errors.New("tools: path is denied")

// defaultDenyGlobs are always denied, regardless of caller-supplied globs —
// CLAUDE.md: "Deny reads of .env, .git/config, *.pem, id_*".
var defaultDenyGlobs = []string{".env", ".git/config", "*.pem", "id_*"}

// Sandbox confines filesystem tools to a workspace root, shared by every
// tool that touches disk.
type Sandbox struct {
	root      string
	denyGlobs []string
}

// NewSandbox resolves root to an absolute, symlink-free path and returns a
// Sandbox scoped to it. denyGlobs are checked in addition to
// defaultDenyGlobs.
func NewSandbox(root string, denyGlobs []string) (*Sandbox, error) {
	resolved, err := resolveRoot(root)
	if err != nil {
		return nil, err
	}
	all := make([]string, 0, len(defaultDenyGlobs)+len(denyGlobs))
	all = append(all, defaultDenyGlobs...)
	all = append(all, denyGlobs...)
	return &Sandbox{root: resolved, denyGlobs: all}, nil
}

// Reroot re-resolves root the same way NewSandbox does and, on success,
// replaces s's root in place — every tool sharing this *Sandbox (they're
// all constructed with the same pointer) sees the new root immediately,
// without the caller having to rebuild the tool set. It refuses (leaving s
// unchanged) if root doesn't exist or can't be resolved, for the /cd
// command (CLAUDE.md phase 11.3).
func (s *Sandbox) Reroot(root string) error {
	resolved, err := resolveRoot(root)
	if err != nil {
		return err
	}
	s.root = resolved
	return nil
}

// resolveRoot turns root into an absolute, symlink-free path, refusing one
// that doesn't exist or can't be resolved (EvalSymlinks has to stat every
// path component, so a missing or permission-denied target both surface
// here as an error).
func resolveRoot(root string) (string, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("tools: workspace root %q: %w", root, err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("tools: workspace root %q: %w", root, err)
	}
	return resolved, nil
}

// Resolve validates path (relative to the sandbox root, or absolute) and
// returns its resolved absolute form. It rejects anything that escapes the
// root — including via a symlink whose target lies outside it — or matches
// a deny glob. path need not exist: only its existing ancestors are
// symlink-resolved, so a not-yet-created write target still resolves
// correctly.
func (s *Sandbox) Resolve(path string) (string, error) {
	if path == "" {
		return "", errors.New("tools: empty path")
	}

	joined := path
	if !filepath.IsAbs(path) {
		joined = filepath.Join(s.root, path)
	}
	joined = filepath.Clean(joined)

	resolved, err := resolveExistingSymlinks(joined)
	if err != nil {
		return "", fmt.Errorf("tools: resolve %q: %w", path, err)
	}
	if !s.contains(resolved) {
		return "", fmt.Errorf("%w: %q", ErrPathEscapesSandbox, path)
	}

	if s.denied(resolved) {
		return "", fmt.Errorf("%w: %q", ErrPathDenied, path)
	}

	return resolved, nil
}

func (s *Sandbox) contains(p string) bool {
	return p == s.root || strings.HasPrefix(p, s.root+string(filepath.Separator))
}

// Denied reports whether resolved — an absolute path already known to lie
// beneath the sandbox root — matches a deny glob. Exposed for tools like
// search_files that walk the tree themselves rather than resolving each
// visited file through Resolve.
func (s *Sandbox) Denied(resolved string) bool {
	return s.denied(resolved)
}

func (s *Sandbox) denied(resolved string) bool {
	rel, err := filepath.Rel(s.root, resolved)
	if err != nil {
		return false
	}
	base := filepath.Base(resolved)
	for _, pattern := range s.denyGlobs {
		if ok, _ := filepath.Match(pattern, base); ok {
			return true
		}
		if ok, _ := filepath.Match(pattern, rel); ok {
			return true
		}
	}
	return false
}

// resolveExistingSymlinks walks up from path to the longest existing
// ancestor, symlink-resolves that ancestor, and rejoins the remaining
// (possibly not-yet-existing) suffix.
func resolveExistingSymlinks(path string) (string, error) {
	if _, err := os.Lstat(path); err == nil {
		return filepath.EvalSymlinks(path)
	}
	parent := filepath.Dir(path)
	if parent == path {
		return path, nil
	}
	resolvedParent, err := resolveExistingSymlinks(parent)
	if err != nil {
		return "", err
	}
	return filepath.Join(resolvedParent, filepath.Base(path)), nil
}
