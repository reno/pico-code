package tools

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestSandboxResolveRejections(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".env"), []byte("SECRET=1"), 0o600); err != nil {
		t.Fatalf("WriteFile(.env) error = %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatalf("MkdirAll(sub) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "sub", "ok.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatalf("WriteFile(sub/ok.txt) error = %v", err)
	}

	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("nope"), 0o600); err != nil {
		t.Fatalf("WriteFile(outside) error = %v", err)
	}
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "escape_link")); err != nil {
		t.Fatalf("Symlink() error = %v", err)
	}

	sandbox, err := NewSandbox(root, []string{"custom_deny*"})
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "custom_deny.txt"), []byte("x"), 0o600); err != nil {
		t.Fatalf("WriteFile(custom_deny.txt) error = %v", err)
	}

	tests := []struct {
		name    string
		path    string
		wantErr error
	}{
		{"dot-dot escape", "../outside.txt", ErrPathEscapesSandbox},
		{"absolute path outside root", filepath.Join(outside, "secret.txt"), ErrPathEscapesSandbox},
		{"symlink escapes root", "escape_link", ErrPathEscapesSandbox},
		{"default deny glob", ".env", ErrPathDenied},
		{"caller deny glob", "custom_deny.txt", ErrPathDenied},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := sandbox.Resolve(tt.path)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Resolve(%q) error = %v, want wrapping %v", tt.path, err, tt.wantErr)
			}
			if err.Error() == "" {
				t.Fatal("expected a descriptive error message")
			}
		})
	}
}

func TestSandboxResolveAcceptsPathsInsideRoot(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "sub", "ok.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}

	tests := []string{"sub/ok.txt", filepath.Join(root, "sub", "ok.txt"), "sub/../sub/ok.txt"}
	for _, path := range tests {
		t.Run(path, func(t *testing.T) {
			resolved, err := sandbox.Resolve(path)
			if err != nil {
				t.Fatalf("Resolve(%q) error = %v", path, err)
			}
			want, err := filepath.EvalSymlinks(filepath.Join(root, "sub", "ok.txt"))
			if err != nil {
				t.Fatalf("EvalSymlinks() error = %v", err)
			}
			if resolved != want {
				t.Errorf("Resolve(%q) = %q, want %q", path, resolved, want)
			}
		})
	}
}

func TestSandboxResolveAllowsNotYetExistingPathInsideRoot(t *testing.T) {
	root := t.TempDir()
	sandbox, err := NewSandbox(root, nil)
	if err != nil {
		t.Fatalf("NewSandbox() error = %v", err)
	}

	resolved, err := sandbox.Resolve("new/nested/file.txt")
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("EvalSymlinks() error = %v", err)
	}
	want := filepath.Join(resolvedRoot, "new", "nested", "file.txt")
	if resolved != want {
		t.Errorf("Resolve() = %q, want %q", resolved, want)
	}
}
