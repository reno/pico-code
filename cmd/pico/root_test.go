package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestChatHelpListsEveryFlag(t *testing.T) {
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs([]string{"chat", "--help"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat --help returned error: %v", err)
	}

	out := buf.String()
	wantFlags := []string{
		"--provider", "--model", "--max-turns", "--token-budget",
		"--workspace", "--yes", "--tools", "--stream", "--tui", "--log-level",
		"--num-ctx", "--allow-write",
	}
	for _, flag := range wantFlags {
		if !strings.Contains(out, flag) {
			t.Errorf("chat --help output missing flag %q\noutput:\n%s", flag, out)
		}
	}
}

func TestChatProviderEnvFallback(t *testing.T) {
	env := map[string]string{"PICO_CODE_PROVIDER": "ollama"}
	getenv := func(k string) string { return env[k] }

	root := newRootCmd(getenv)
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"chat"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if !strings.Contains(buf.String(), "provider=ollama") {
		t.Errorf("expected env fallback to select ollama provider, got: %s", buf.String())
	}
}

func TestChatFlagOverridesEnv(t *testing.T) {
	env := map[string]string{"PICO_CODE_PROVIDER": "ollama"}
	getenv := func(k string) string { return env[k] }

	root := newRootCmd(getenv)
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"chat", "--provider=anthropic"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if !strings.Contains(buf.String(), "provider=anthropic") {
		t.Errorf("expected explicit flag to override env, got: %s", buf.String())
	}
}

func TestChatUnknownProviderIsClearError(t *testing.T) {
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs([]string{"chat", "--provider=bogus"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected an error for an unknown provider")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("expected error to name the offending value, got: %v", err)
	}
}
