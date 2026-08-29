package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// fakeChatProvider returns one canned reply, enough to drive
// runPlainChat's single-shot piped-input path without a real backend.
// disallowStream, when set, makes Stream fail instead of succeed — used to
// prove a --stream=false turn never calls it at all.
type fakeChatProvider struct {
	reply          string
	disallowStream bool
}

func (f *fakeChatProvider) Name() string { return "fake" }

func (f *fakeChatProvider) Chat(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: []llm.Block{llm.Text{Text: f.reply}}},
		StopReason: "end_turn",
	}, nil
}

func (f *fakeChatProvider) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Event, error) {
	if f.disallowStream {
		return nil, errors.New("fakeChatProvider: Stream must not be called when streaming is disabled")
	}
	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		if !send(llm.TextDelta{Text: f.reply}) {
			return
		}
		send(llm.MessageDone{StopReason: "end_turn"})
	}), nil
}

// TestPipedChatProducesCleanANSIFreeOutput is 7.1's AC: piping input to a
// non-interactive chat session produces clean, ANSI-free output.
func TestPipedChatProducesCleanANSIFreeOutput(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetIn(strings.NewReader("hi\n"))
	var out bytes.Buffer
	cmd.SetOut(&out)

	cfg := &config.Config{Yes: true, Stream: true}
	provider := &fakeChatProvider{reply: "hello there"}
	h := history.New()

	err := runPlainChat(cmd, cfg, provider, tools.NewRegistry(), h, agent.Guards{}, noSession(t))
	if err != nil {
		t.Fatalf("runPlainChat() error = %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "hello there") {
		t.Errorf("output = %q, want it to contain the reply", got)
	}
	if strings.ContainsRune(got, '\x1b') {
		t.Errorf("output contains an ANSI escape byte: %q", got)
	}
	if err := h.Validate(); err != nil {
		t.Errorf("history.Validate() error = %v", err)
	}
}

// TestUsageCommandReportsCumulativeAcrossTurns is 8.1's AC ("expose /usage
// in the chat"), driven through runREPL so it never needs a real terminal.
func TestUsageCommandReportsCumulativeAcrossTurns(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	in := strings.NewReader("hello\n/usage\n")
	runTurn := newTurnRunner(&config.Config{Stream: true}, ag, &out)
	if err := runREPL(context.Background(), in, &out, ag, h, noSession(t), runTurn); err != nil {
		t.Fatalf("runREPL() error = %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "1 turn") {
		t.Errorf("output = %q, want it to report 1 completed turn", got)
	}
	if strings.ContainsRune(got, '\x1b') {
		t.Errorf("output contains an ANSI escape byte: %q", got)
	}
}

func TestSlashCommandParsesNameAndArg(t *testing.T) {
	name, arg, ok := slashCommand("/save my-session")
	if !ok || name != "save" || arg != "my-session" {
		t.Errorf("slashCommand(%q) = (%q, %q, %v), want (\"save\", \"my-session\", true)", "/save my-session", name, arg, ok)
	}

	if _, _, ok := slashCommand("not a command"); ok {
		t.Error("slashCommand on ordinary text should return ok = false")
	}
}

// noSession returns an inactive session (no name) rooted at a temp
// directory, for tests that need a *session parameter but don't care about
// persistence.
func noSession(t *testing.T) *session {
	t.Helper()
	sess, err := newSession(t.TempDir(), "")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	return sess
}

// TestSessionAutoSavesAfterEachTurnAndResumes is 8.3's AC in the plain
// REPL: killing the process mid-session and resuming has the model see the
// full prior context. "Killing the process" is simulated by driving runREPL
// once for the first half of the conversation, then constructing a brand
// new History via the session's on-disk file (as a fresh process would)
// and driving runREPL again.
func TestSessionAutoSavesAfterEachTurnAndResumes(t *testing.T) {
	dir := t.TempDir()

	sess1, err := newSession(dir, "work")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	h1, err := sess1.loadOrCreateHistory()
	if err != nil {
		t.Fatalf("loadOrCreateHistory() error = %v", err)
	}
	provider := &fakeChatProvider{reply: "first reply"}
	ag1 := agent.New(provider, tools.NewRegistry(), h1, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	runTurn1 := newTurnRunner(&config.Config{Stream: true}, ag1, io.Discard)
	if err := runREPL(context.Background(), strings.NewReader("first message\n"), io.Discard, ag1, h1, sess1, runTurn1); err != nil {
		t.Fatalf("runREPL() (process 1) error = %v", err)
	}

	// A brand new process: nothing but the session name and the file it
	// wrote to disk.
	sess2, err := newSession(dir, "work")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	h2, err := sess2.loadOrCreateHistory()
	if err != nil {
		t.Fatalf("loadOrCreateHistory() (process 2) error = %v", err)
	}

	if err := h2.Validate(); err != nil {
		t.Fatalf("Validate() on resumed history error = %v", err)
	}
	if got, want := len(h2.Snapshot()), len(h1.Snapshot()); got != want {
		t.Fatalf("resumed history has %d messages, want %d (the full prior context)", got, want)
	}
}

func TestSessionCommandsNewSaveLoad(t *testing.T) {
	dir := t.TempDir()
	sess, err := newSession(dir, "")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	h := history.New()
	provider := &fakeChatProvider{reply: "reply"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	in := strings.NewReader("hello there\n/save first\n/new\n/load first\n")
	runTurn := newTurnRunner(&config.Config{Stream: true}, ag, &out)
	if err := runREPL(context.Background(), in, &out, ag, h, sess, runTurn); err != nil {
		t.Fatalf("runREPL() error = %v", err)
	}

	got := out.String()
	if !strings.Contains(got, `saved session "first"`) {
		t.Errorf("output missing save confirmation: %q", got)
	}
	if !strings.Contains(got, "started a new, unsaved session") {
		t.Errorf("output missing /new confirmation: %q", got)
	}
	if !strings.Contains(got, `loaded session "first"`) {
		t.Errorf("output missing load confirmation: %q", got)
	}
	// /load restored the turn /new had cleared.
	if len(h.Snapshot()) == 0 {
		t.Error("history is empty after /load, want the saved turn restored")
	}
}

// TestSessionLoadOfCorruptFileIsReadableErrorNotPanic is 8.3's other AC: a
// corrupt session file fails with a readable error rather than a panic.
func TestSessionLoadOfCorruptFileIsReadableErrorNotPanic(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "broken.json"), []byte("not json"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	sess, err := newSession(dir, "broken")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	if _, err := sess.loadOrCreateHistory(); err == nil {
		t.Fatal("loadOrCreateHistory() = nil error, want a readable error for a corrupt session file")
	}
}

// TestStreamFalseUsesNonStreamingPath proves --stream=false actually has
// an effect: the fake provider's Stream errors, so a passing turn here
// means newTurnRunner drove it through Chat/ag.Run instead.
func TestStreamFalseUsesNonStreamingPath(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetIn(strings.NewReader("hi\n"))
	var out bytes.Buffer
	cmd.SetOut(&out)

	cfg := &config.Config{Yes: true, Stream: false}
	provider := &fakeChatProvider{reply: "non-streamed reply", disallowStream: true}
	h := history.New()

	if err := runPlainChat(cmd, cfg, provider, tools.NewRegistry(), h, agent.Guards{}, noSession(t)); err != nil {
		t.Fatalf("runPlainChat() error = %v", err)
	}
	if got := out.String(); !strings.Contains(got, "non-streamed reply") {
		t.Errorf("output = %q, want it to contain the reply", got)
	}
}

// TestStreamTrueWithPromptedToolsFallsBackToNonStreaming is the other half
// of newTurnRunner's --tools=prompted handling: even with --stream=true,
// prompted mode can't stream, so the turn must still go through Chat.
func TestStreamTrueWithPromptedToolsFallsBackToNonStreaming(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetIn(strings.NewReader("hi\n"))
	var out bytes.Buffer
	cmd.SetOut(&out)

	cfg := &config.Config{Yes: true, Stream: true, Tools: config.ToolsPrompted}
	provider := &fakeChatProvider{reply: "fallback reply", disallowStream: true}
	h := history.New()

	if err := runPlainChat(cmd, cfg, provider, tools.NewRegistry(), h, agent.Guards{}, noSession(t)); err != nil {
		t.Fatalf("runPlainChat() error = %v", err)
	}
	if got := out.String(); !strings.Contains(got, "fallback reply") {
		t.Errorf("output = %q, want it to contain the reply", got)
	}
}

func TestResolveProviderWrapsForPromptedTools(t *testing.T) {
	provider := &fakeChatProvider{reply: "unused"}
	got, err := resolveProvider(&config.Config{Tools: config.ToolsPrompted}, provider)
	if err != nil {
		t.Fatalf("resolveProvider() error = %v", err)
	}
	if got == llm.Provider(provider) {
		t.Error("resolveProvider() returned the provider unwrapped, want it wrapped for prompted-tools mode")
	}
}

func TestResolveProviderLeavesNativeToolsUnwrapped(t *testing.T) {
	provider := &fakeChatProvider{reply: "unused"}
	got, err := resolveProvider(&config.Config{Tools: config.ToolsNative}, provider)
	if err != nil {
		t.Fatalf("resolveProvider() error = %v", err)
	}
	if got != llm.Provider(provider) {
		t.Error("resolveProvider() should return native-mode providers unwrapped")
	}
}

func TestResolveProviderRejectsTUIWithPromptedTools(t *testing.T) {
	provider := &fakeChatProvider{reply: "unused"}
	_, err := resolveProvider(&config.Config{Tools: config.ToolsPrompted, TUI: true}, provider)
	if err == nil {
		t.Fatal("resolveProvider() error = nil, want an error for --tui with --tools=prompted")
	}
	if !strings.Contains(err.Error(), "tui") || !strings.Contains(err.Error(), "prompted") {
		t.Errorf("resolveProvider() error = %q, want it to name both --tui and --tools=prompted", err)
	}
}

func TestBuildRegistryRegistersRunCommandOnlyWhenAllowlisted(t *testing.T) {
	dir := t.TempDir()

	withCommands, err := buildRegistry(&config.Config{Workspace: dir, AllowCommands: []string{"echo"}})
	if err != nil {
		t.Fatalf("buildRegistry() error = %v", err)
	}
	if _, err := withCommands.Get("run_command"); err != nil {
		t.Errorf("Get(\"run_command\") error = %v, want it registered when --allow-commands is non-empty", err)
	}

	without, err := buildRegistry(&config.Config{Workspace: dir})
	if err != nil {
		t.Fatalf("buildRegistry() error = %v", err)
	}
	if _, err := without.Get("run_command"); !errors.Is(err, tools.ErrToolNotFound) {
		t.Errorf("Get(\"run_command\") error = %v, want ErrToolNotFound when --allow-commands is empty", err)
	}
}

// TestRunTUICommandUsage is the TUI counterpart of
// TestUsageCommandReportsCumulativeAcrossTurns: /usage must work the same
// way whether it's typed into the plain REPL or the TUI's textarea.
func TestRunTUICommandUsage(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)
	runTurn := newTurnRunner(&config.Config{Stream: true}, ag, io.Discard)
	if err := runTurn(context.Background(), "hello"); err != nil {
		t.Fatalf("runTurn() error = %v", err)
	}

	out := runTUICommand(ag, h, noSession(t), "/usage", "usage", "")

	if !strings.Contains(out, "> /usage") {
		t.Errorf("output = %q, want it to echo the command line", out)
	}
	if !strings.Contains(out, "1 turn") {
		t.Errorf("output = %q, want it to report 1 completed turn", out)
	}
}

// TestRunTUICommandNewSaveLoad is the TUI counterpart of
// TestSessionCommandsNewSaveLoad.
func TestRunTUICommandNewSaveLoad(t *testing.T) {
	dir := t.TempDir()
	sess, err := newSession(dir, "")
	if err != nil {
		t.Fatalf("newSession() error = %v", err)
	}
	h := history.New()
	h.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: "hello there"}}})
	provider := &fakeChatProvider{reply: "reply"}
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	saveOut := runTUICommand(ag, h, sess, "/save first", "save", "first")
	if !strings.Contains(saveOut, `saved session "first"`) {
		t.Errorf("save output = %q, want a save confirmation", saveOut)
	}

	newOut := runTUICommand(ag, h, sess, "/new", "new", "")
	if !strings.Contains(newOut, "started a new, unsaved session") {
		t.Errorf("new output = %q, want a /new confirmation", newOut)
	}
	if len(h.Snapshot()) != 0 {
		t.Errorf("history has %d messages after /new, want 0", len(h.Snapshot()))
	}

	loadOut := runTUICommand(ag, h, sess, "/load first", "load", "first")
	if !strings.Contains(loadOut, `loaded session "first"`) {
		t.Errorf("load output = %q, want a load confirmation", loadOut)
	}
	if len(h.Snapshot()) == 0 {
		t.Error("history is empty after /load, want the saved turn restored")
	}
}

// TestEffectiveLogLevelQuietsTUIWithoutExplicitFlag is a regression test:
// stderr shares the TUI's alt-screen terminal, so an info-level line
// printed before program.Run() takes over used to flash onto the screen as
// stray text (e.g. "ollama: context window configured"). --tui should
// default to a quieter level unless the caller passed --log-level.
func TestEffectiveLogLevelQuietsTUIWithoutExplicitFlag(t *testing.T) {
	got := effectiveLogLevel(&config.Config{TUI: true, LogLevel: config.LogLevelInfo}, false)
	if got != config.LogLevelWarn {
		t.Errorf("effectiveLogLevel(TUI, unset flag) = %q, want %q", got, config.LogLevelWarn)
	}
}

func TestEffectiveLogLevelRespectsExplicitFlagEvenInTUI(t *testing.T) {
	got := effectiveLogLevel(&config.Config{TUI: true, LogLevel: config.LogLevelDebug}, true)
	if got != config.LogLevelDebug {
		t.Errorf("effectiveLogLevel(TUI, explicit flag) = %q, want the explicitly requested %q", got, config.LogLevelDebug)
	}
}

func TestEffectiveLogLevelUnchangedForPlainMode(t *testing.T) {
	got := effectiveLogLevel(&config.Config{TUI: false, LogLevel: config.LogLevelInfo}, false)
	if got != config.LogLevelInfo {
		t.Errorf("effectiveLogLevel(plain) = %q, want cfg.LogLevel (%q) unmodified", got, config.LogLevelInfo)
	}
}

func TestCompactionPolicyUsesContextWindowForNonOllamaProviders(t *testing.T) {
	got := compactionPolicy(&config.Config{Provider: config.ProviderAnthropic, ContextWindow: 123_456, NumCtx: 4096})
	if got.ContextWindow != 123_456 {
		t.Errorf("ContextWindow = %d, want the configured --context-window value (4096 is Ollama's NumCtx and must not leak in here)", got.ContextWindow)
	}
}

func TestCompactionPolicyUsesNumCtxForOllama(t *testing.T) {
	got := compactionPolicy(&config.Config{Provider: config.ProviderOllama, ContextWindow: 123_456, NumCtx: 8192})
	if got.ContextWindow != 8192 {
		t.Errorf("ContextWindow = %d, want NumCtx (8192), not --context-window (which Ollama ignores)", got.ContextWindow)
	}
}
