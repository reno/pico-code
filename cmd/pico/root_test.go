package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/config"
)

// stubRunChat swaps the package-level runChat var for a stub that records
// the resolved cfg instead of driving a real provider/agent turn, returning
// a getter for it (valid only after Execute runs) and restoring the
// original on cleanup — tests that only care about how flags/env resolve
// into a Config use this so they never touch the network or block on
// stdin.
func stubRunChat(t *testing.T) func() *config.Config {
	t.Helper()
	var got *config.Config
	orig := runChat
	runChat = func(_ *cobra.Command, cfg *config.Config) error {
		got = cfg
		return nil
	}
	t.Cleanup(func() { runChat = orig })
	return func() *config.Config { return got }
}

func TestChatHelpListsEveryFlag(t *testing.T) {
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs([]string{"--help"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat --help returned error: %v", err)
	}

	out := buf.String()
	wantFlags := []string{
		"--provider", "--model", "--max-turns", "--token-budget",
		"--workspace", "--yes", "--tools", "--stream", "--tui", "--log-level",
		"--num-ctx", "--allow-write", "--session", "--allow-commands", "--context-window",
	}
	for _, flag := range wantFlags {
		if !strings.Contains(out, flag) {
			t.Errorf("chat --help output missing flag %q\noutput:\n%s", flag, out)
		}
	}
}

func TestChatProviderEnvFallback(t *testing.T) {
	getCfg := stubRunChat(t)
	env := map[string]string{"PICO_CODE_PROVIDER": "ollama"}
	getenv := func(k string) string { return env[k] }

	root := newRootCmd(getenv)
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs(nil)

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if got := getCfg().Provider; got != config.ProviderOllama {
		t.Errorf("Provider = %q, want env fallback to select %q", got, config.ProviderOllama)
	}
}

func TestChatFlagOverridesEnv(t *testing.T) {
	getCfg := stubRunChat(t)
	env := map[string]string{"PICO_CODE_PROVIDER": "ollama"}
	getenv := func(k string) string { return env[k] }

	root := newRootCmd(getenv)
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"--provider=anthropic"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if got := getCfg().Provider; got != config.ProviderAnthropic {
		t.Errorf("Provider = %q, want the explicit flag to override env and select %q", got, config.ProviderAnthropic)
	}
}

func TestChatToolsDefaultsToPromptedForOllama(t *testing.T) {
	getCfg := stubRunChat(t)
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"--provider=ollama"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if got := getCfg().Tools; got != config.ToolsPrompted {
		t.Errorf("Tools = %q, want --provider=ollama with no explicit --tools to default to %q", got, config.ToolsPrompted)
	}
}

func TestChatToolsDefaultsToNativeForAnthropic(t *testing.T) {
	getCfg := stubRunChat(t)
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"--provider=anthropic"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if got := getCfg().Tools; got != config.ToolsNative {
		t.Errorf("Tools = %q, want --provider=anthropic to keep defaulting to %q", got, config.ToolsNative)
	}
}

func TestChatExplicitToolsOverridesOllamaDefault(t *testing.T) {
	getCfg := stubRunChat(t)
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetArgs([]string{"--provider=ollama", "--tools=native"})

	if err := root.Execute(); err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if got := getCfg().Tools; got != config.ToolsNative {
		t.Errorf("Tools = %q, want an explicit --tools=native to override the ollama default", got)
	}
}

func TestChatUnknownProviderIsClearError(t *testing.T) {
	root := newRootCmd(func(string) string { return "" })
	buf := &bytes.Buffer{}
	root.SetOut(buf)
	root.SetErr(buf)
	root.SetArgs([]string{"--provider=bogus"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected an error for an unknown provider")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("expected error to name the offending value, got: %v", err)
	}
}
