package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/user"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/llm/prompted"
	"github.com/reno/pico-code/internal/tools"
	"github.com/reno/pico-code/internal/ui"
)

// version is stamped into the home screen's frame title. Bumped by hand;
// there is no release pipeline to derive it from yet.
const version = "v0.1.0"

const (
	systemPrompt       = "You are pico, a terminal coding agent. Be direct and concise."
	defaultMaxTokens   = 4096
	defaultToolTimeout = 2 * time.Minute
)

// configureChat wires chat's flags and RunE directly onto cmd — the root
// command itself, since starting a chat session is pico's only behavior and
// doesn't need its own subcommand. Flag values win over the
// PICO_CODE_PROVIDER environment fallback, which wins over the built-in
// default; ANTHROPIC_API_KEY and OLLAMA_HOST are read directly by
// config.Load since they are credentials, not flags.
//
// --tools has its own provider-dependent default: local models frequently
// advertise native tool support without reliably using it (CLAUDE.md's
// "narrate a tool call in prose instead of calling it"), so an explicit
// --provider=ollama with no explicit --tools defaults to prompted rather
// than native. An explicit --tools always wins.
func configureChat(cmd *cobra.Command, getenv func(string) string) {
	var flags struct {
		provider      string
		model         string
		maxTurns      int
		tokenBudget   int
		workspace     string
		yes           bool
		tools         string
		stream        bool
		tui           bool
		logLevel      string
		numCtx        int
		allowWrite    bool
		session       string
		allowCommands []string
		contextWindow int
		mcpConfig     string
		think         bool
	}

	cmd.RunE = func(cmd *cobra.Command, _ []string) error {
			provider := flags.provider
			if !cmd.Flags().Changed("provider") {
				if v := getenv("PICO_CODE_PROVIDER"); v != "" {
					provider = v
				}
			}

			toolsMode := flags.tools
			if !cmd.Flags().Changed("tools") && provider == string(config.ProviderOllama) {
				toolsMode = string(config.ToolsPrompted)
			}

			cfg, err := config.Load(config.Flags{
				Provider:      provider,
				Model:         flags.model,
				MaxTurns:      flags.maxTurns,
				TokenBudget:   flags.tokenBudget,
				Workspace:     flags.workspace,
				Yes:           flags.yes,
				Tools:         toolsMode,
				Stream:        flags.stream,
				TUI:           flags.tui,
				LogLevel:      flags.logLevel,
				NumCtx:        flags.numCtx,
				AllowWrite:    flags.allowWrite,
				Session:       flags.session,
				AllowCommands: flags.allowCommands,
				ContextWindow: flags.contextWindow,
				MCPConfig:     flags.mcpConfig,
				Think:         flags.think,
			}, getenv)
			if err != nil {
				return fmt.Errorf("resolving config: %w", err)
			}

			return runChat(cmd, cfg)
		}

	f := cmd.Flags()
	f.StringVar(&flags.provider, "provider", "anthropic", "LLM backend to use (anthropic|ollama|openai)")
	f.StringVar(&flags.model, "model", "", "model identifier for the selected provider")
	f.IntVar(&flags.maxTurns, "max-turns", 25, "maximum agent loop turns before stopping")
	f.IntVar(&flags.tokenBudget, "token-budget", 100_000, "maximum cumulative tokens before stopping")
	f.StringVar(&flags.workspace, "workspace", ".", "root directory filesystem tools are confined to")
	f.BoolVar(&flags.yes, "yes", false, "skip interactive approval prompts")
	f.StringVar(&flags.tools, "tools", "native", "tool-calling mode (native|prompted); defaults to prompted when --provider=ollama and --tools isn't set explicitly")
	f.BoolVar(&flags.stream, "stream", true, "stream provider responses")
	f.BoolVar(&flags.tui, "tui", false, "use the bubbletea TUI instead of plain output")
	f.StringVar(&flags.logLevel, "log-level", "info", "log level (debug|info|warn|error)")
	f.IntVar(&flags.numCtx, "num-ctx", 4096, "context window size passed to Ollama's num_ctx (ignored by other providers)")
	f.BoolVar(&flags.allowWrite, "allow-write", false, "register the write_file tool (off by default)")
	f.StringVar(&flags.session, "session", "", "name a session to resume or start; saved after every turn")
	f.StringSliceVar(&flags.allowCommands, "allow-commands", nil, "comma-separated binary allowlist for run_command; registers the tool only if non-empty")
	f.IntVar(&flags.contextWindow, "context-window", defaultContextWindow, "context window size compaction measures usage against (ignored by Ollama, which uses --num-ctx instead)")
	f.StringVar(&flags.mcpConfig, "mcp-config", "", `path to a JSON file listing MCP servers, shaped {"mcpServers": {name: {command, args, env}}}`)
	f.BoolVar(&flags.think, "think", false, "ask the model for a reasoning trace ahead of its reply, when the provider supports one (currently Ollama only)")
}

// runChat is a package var so tests can substitute a stub instead of
// driving a real provider/agent turn end to end.
var runChat = func(cmd *cobra.Command, cfg *config.Config) error {
	// The TUI shares stderr with the alt-screen terminal, so a routine
	// info-level line (e.g. "ollama: context window configured") printed
	// before program.Run() takes over flashes onto the screen as stray
	// text. Default to a quieter level for --tui unless the caller asked
	// for a specific one explicitly.
	logLevel := effectiveLogLevel(cfg, cmd.Flags().Changed("log-level"))
	if err := setupLogging(cmd.ErrOrStderr(), logLevel); err != nil {
		return err
	}

	provider, err := llm.New(cfg)
	if err != nil {
		return fmt.Errorf("resolving provider: %w", friendlyAgentError(cfg, err))
	}
	provider, err = resolveProvider(cfg, provider)
	if err != nil {
		return err
	}

	registry, err := buildRegistry(cfg)
	if err != nil {
		return err
	}

	sessDir, err := defaultSessionsDir()
	if err != nil {
		return err
	}
	sess, err := newSession(sessDir, cfg.Session)
	if err != nil {
		return err
	}
	sess.model = cfg.Model

	mcpServers, err := loadMCPServers(cfg.MCPConfig)
	if err != nil {
		return err
	}
	sess.mcp = newMCPManager(cmd.Context(), registry, mcpServers, mcpDiscoverTimeout)
	defer sess.mcp.shutdown()

	h, err := sess.loadOrCreateHistory()
	if err != nil {
		return err
	}

	guards := agent.Guards{MaxTurns: cfg.MaxTurns, TokenBudget: cfg.TokenBudget}

	if cfg.TUI {
		return runTUIChat(cmd, cfg, provider, registry, h, guards, sess)
	}
	return runPlainChat(cmd, cfg, provider, registry, h, guards, sess)
}

// defaultContextWindow is --context-window's default: a conservative
// stand-in for a real model's actual context size, used as-is unless the
// flag overrides it. Ollama ignores it entirely and uses NumCtx instead —
// that's already an explicit, required value (CLAUDE.md), so compaction
// reuses it rather than tracking a second number that could drift from it.
const (
	defaultContextWindow      = 200_000
	compactionTriggerFraction = 0.75
	compactionKeepTurns       = 6
)

// effectiveLogLevel is the level setupLogging actually uses: cfg.LogLevel,
// except a TUI session with no explicit --log-level defaults to warn, since
// stderr shares the alt-screen terminal and a routine info line would flash
// onto the screen as stray text before program.Run() takes it over.
func effectiveLogLevel(cfg *config.Config, logLevelFlagChanged bool) config.LogLevel {
	if cfg.TUI && !logLevelFlagChanged {
		return config.LogLevelWarn
	}
	return cfg.LogLevel
}

func compactionPolicy(cfg *config.Config) agent.CompactionPolicy {
	return agent.CompactionPolicy{ContextWindow: contextWindow(cfg), TriggerFraction: compactionTriggerFraction, KeepTurns: compactionKeepTurns}
}

// contextWindow is the token budget compaction measures against: NumCtx
// for Ollama, whose window is already explicit and required, and the
// configured ContextWindow for every other provider.
func contextWindow(cfg *config.Config) int {
	if cfg.Provider == config.ProviderOllama {
		return cfg.NumCtx
	}
	return cfg.ContextWindow
}

// resolveProvider wraps provider with prompted-tools support when cfg.Tools
// asks for it, rejecting that combined with --tui: the TUI always drives
// the agent through RunStream, and prompted.Provider.Stream always errors
// (it needs the full reply before it can look for a fenced tool call), so
// the combination would fail confusingly on the very first turn instead of
// with a clear message at startup.
func resolveProvider(cfg *config.Config, provider llm.Provider) (llm.Provider, error) {
	if cfg.Tools != config.ToolsPrompted {
		return provider, nil
	}
	if cfg.TUI {
		return nil, errors.New("--tui does not support --tools=prompted: the TUI requires streaming, and prompted mode can't stream")
	}
	return prompted.Wrap(provider), nil
}

// buildRegistry wires the built-in tools this session offers. run_command
// is only registered when --allow-commands names at least one binary — an
// empty allowlist would make the tool present but unconditionally useless,
// worse than leaving it out.
func buildRegistry(cfg *config.Config) (*tools.Registry, error) {
	sandbox, err := tools.NewSandbox(cfg.Workspace, nil)
	if err != nil {
		return nil, fmt.Errorf("resolving workspace sandbox: %w", err)
	}

	registry := tools.NewRegistry()
	readTool, err := tools.NewReadFileTool(sandbox)
	if err != nil {
		return nil, err
	}
	if err := registry.Register(readTool); err != nil {
		return nil, err
	}
	listTool, err := tools.NewListDirTool(sandbox)
	if err != nil {
		return nil, err
	}
	if err := registry.Register(listTool); err != nil {
		return nil, err
	}
	searchTool, err := tools.NewSearchFilesTool(sandbox)
	if err != nil {
		return nil, err
	}
	if err := registry.Register(searchTool); err != nil {
		return nil, err
	}
	if cfg.AllowWrite {
		writeTool, err := tools.NewWriteFileTool(sandbox)
		if err != nil {
			return nil, err
		}
		if err := registry.Register(writeTool); err != nil {
			return nil, err
		}
	}
	if len(cfg.AllowCommands) > 0 {
		runTool, err := tools.NewRunCommandTool(cfg.AllowCommands, defaultToolTimeout)
		if err != nil {
			return nil, err
		}
		if err := registry.Register(runTool); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

// subAgentToolNames are the tools a sub-agent is allowed to call. Deliberately
// read-only: run_command and write_file need interactive approval, but a
// sub-agent runs under agent.AutoApprove (there's no clean way to surface an
// approval prompt from inside a nested tool call), so including either here
// would silently bypass CLAUDE.md's approval-per-call rule.
var subAgentToolNames = []string{"read_file", "list_dir", "search_files"}

// wireSubAgent registers a sub_agent tool into registry once ag exists,
// giving it the same provider and a restricted, read-only tool set so a
// sub-agent's own budget is carved out of ag's configured Guards. It has to
// run after agent.New rather than inside buildRegistry: the sub-agent tool
// needs a live *agent.Agent to read Guards()/CumulativeUsage() from, and
// building that Agent needs the registry to already exist.
func wireSubAgent(registry *tools.Registry, provider llm.Provider, ag *agent.Agent) error {
	var allowed []tools.Tool
	for _, name := range subAgentToolNames {
		t, err := registry.Get(name)
		if err != nil {
			continue // not registered in this session (e.g. an empty test registry)
		}
		allowed = append(allowed, t)
	}
	sa, err := agent.NewSubAgentTool(provider, allowed, systemPrompt, defaultMaxTokens, defaultToolTimeout, ag)
	if err != nil {
		return err
	}
	return registry.Register(sa)
}

// runPlainChat drives the agent through ui.PlainRenderer. Piped/non-TTY
// stdin (the common case for scripting and for 7.1's AC) is read in full as
// one message and answered once; a TTY gets a minimal read-eval-print loop.
func runPlainChat(cmd *cobra.Command, cfg *config.Config, provider llm.Provider, registry *tools.Registry, h *history.History, guards agent.Guards, sess *session) error {
	approver := agent.AutoApprove
	if !cfg.Yes {
		approver = agent.ConsoleApprover{In: cmd.InOrStdin(), Out: cmd.OutOrStdout()}
	}
	ag := agent.New(provider, registry, h, systemPrompt, defaultMaxTokens, guards, defaultToolTimeout, approver)
	if err := wireSubAgent(registry, provider, ag); err != nil {
		return err
	}
	ag.SetCompactionPolicy(compactionPolicy(cfg))
	ag.SetThink(cfg.Think)
	out := cmd.OutOrStdout()
	ctx := cmd.Context()
	in := cmd.InOrStdin()

	runTurn := newTurnRunner(cfg, ag, out)

	if !isInteractive(in) {
		data, err := io.ReadAll(in)
		if err != nil {
			return fmt.Errorf("reading stdin: %w", err)
		}
		input := strings.TrimSpace(string(data))
		if input == "" {
			return nil
		}
		if err := runTurn(ctx, input); err != nil {
			return err
		}
		return sess.saveIfActive(h)
	}

	_, _ = fmt.Fprint(out, ui.Banner(bannerInfo(cfg, sess), terminalWidth(out)))
	return runREPL(ctx, in, out, ag, h, sess, runTurn)
}

// bannerInfo projects the resolved config and on-disk sessions onto the
// fields the home screen shows, so ui never has to import config, read the
// filesystem, or look at the environment.
func bannerInfo(cfg *config.Config, sess *session) ui.BannerInfo {
	info := ui.BannerInfo{
		Version:  version,
		Greeting: ui.Mockery(),
		Provider: string(cfg.Provider),
		Model:    cfg.Model,
		Usage:    fmt.Sprintf("0 / %s tokens", humanTokens(contextWindow(cfg))),
	}
	// The home screen shows the directory you launched from, which is what
	// a shell prompt would show; cfg.Workspace is the tools' sandbox root
	// and can differ from it.
	wd, err := os.Getwd()
	if err != nil {
		wd = cfg.Workspace
	}
	info.Directory = tildeHome(wd)
	if u, err := user.Current(); err == nil {
		info.User = u.Username
	}
	if sess != nil {
		info.Sessions = sess.recent(bannerSessionCount)
	}
	return info
}

// tildeHome abbreviates a leading home directory to "~", the way a shell
// prompt does, so the path fits the home screen's column.
func tildeHome(path string) string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" || !strings.HasPrefix(path, home) {
		return path
	}
	return "~" + strings.TrimPrefix(path, home)
}

// bannerSessionCount is how many recent sessions fit the home screen's
// right column without pushing the box taller than the skull beside it.
const bannerSessionCount = 4

// terminalWidth reports out's width, falling back to 0 (which Banner
// clamps to its minimum) when out is not a terminal — a pipe, or a test's
// buffer.
func terminalWidth(out io.Writer) int {
	f, ok := out.(*os.File)
	if !ok {
		return 0
	}
	w, _, err := term.GetSize(int(f.Fd()))
	if err != nil {
		return 0
	}
	return w
}

// humanTokens renders a token count the way a context window is usually
// quoted ("128k") rather than in full.
func humanTokens(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%dk", n/1000)
	}
	return strconv.Itoa(n)
}

// newTurnRunner picks Run or RunStream per cfg.Stream, printing Run's
// final text itself since PlainRenderer only ever sees a stream.
// tools=prompted forces non-streaming regardless of --stream: the
// prompted.Provider decorator needs a full reply before it can look for a
// fenced tool call, so its Stream always errors.
func newTurnRunner(cfg *config.Config, ag *agent.Agent, out io.Writer) func(context.Context, string) error {
	stream := cfg.Stream
	if cfg.Tools == config.ToolsPrompted && cfg.Stream {
		slog.Info("chat: streaming disabled", "reason", "tools=prompted can't stream")
		stream = false
	}
	if !stream {
		return func(ctx context.Context, input string) error {
			text, err := ag.Run(ctx, input)
			if err != nil {
				return friendlyAgentError(cfg, err)
			}
			_, err = fmt.Fprintln(out, text)
			return err
		}
	}
	renderer := &ui.PlainRenderer{Out: out}
	return func(ctx context.Context, input string) error {
		text, err := ag.RunStream(ctx, input, renderer)
		if err != nil {
			return friendlyAgentError(cfg, err)
		}
		// A guard trip or an empty-reply explanation (16.1's --think can
		// trigger the latter by exhausting the budget on thinking) is
		// synthesized by the agent loop after the last round's Render
		// call already returned, so it was never streamed as TextDelta
		// events — renderer.WroteAny() is false for that round. An
		// ordinary reply, by contrast, already printed itself live; this
		// would otherwise double it.
		if text != "" && !renderer.WroteAny() {
			_, err = fmt.Fprintln(out, text)
		}
		return err
	}
}

// runREPL reads one line at a time from in, dispatching a leading slash
// command to handleCommand and everything else to runTurn (saving to
// sess's file afterward, if one is active), until in hits EOF or ctx is
// cancelled. Split out from runPlainChat so a test can drive it directly
// without needing a real terminal for isInteractive.
func runREPL(ctx context.Context, in io.Reader, out io.Writer, ag *agent.Agent, h *history.History, sess *session, runTurn func(context.Context, string) error) error {
	prompt := func() { _, _ = fmt.Fprint(out, "> ") }

	scanner := bufio.NewScanner(in)
	prompt()
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			prompt()
			continue
		}
		if name, arg, ok := slashCommand(line); ok {
			_, err := handleCommand(ctx, out, ag, h, sess, name, arg)
			switch {
			case errors.Is(err, errExit):
				return nil
			case errors.Is(err, errClearScrollback):
				clearScreen(out)
			case err != nil:
				return err
			}
			prompt()
			continue
		}
		if err := runTurn(ctx, line); err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := sess.saveIfActive(h); err != nil {
			return err
		}
		prompt()
	}
	return scanner.Err()
}

// clearScreen writes an ANSI sequence that clears the visible screen and
// the terminal's scrollback buffer and homes the cursor, for the /clear
// command in the interactive plain REPL. runREPL only reaches this from an
// interactive terminal (never the piped, single-shot path in runPlainChat),
// so 7.1's ANSI-free AC for piped output is unaffected.
func clearScreen(out io.Writer) {
	_, _ = fmt.Fprint(out, "\x1b[H\x1b[2J\x1b[3J")
}

// isInteractive reports whether in is a terminal. A non-*os.File reader
// (e.g. a test's bytes.Buffer) is treated as non-interactive.
func isInteractive(in io.Reader) bool {
	f, ok := in.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// runTUIChat drives the agent through the bubbletea Model: a background
// goroutine reads submitted input and calls RunStream, sending its
// progress and result back into the program as messages, while
// program.Run() owns the terminal until Ctrl+D quits it.
func runTUIChat(cmd *cobra.Command, cfg *config.Config, provider llm.Provider, registry *tools.Registry, h *history.History, guards agent.Guards, sess *session) error {
	ctx := cmd.Context()
	submit := make(chan string, 1)
	// Resolved here, before tea.Program takes the terminal into raw mode:
	// querying the background color once bubbletea owns stdin races with
	// its input reader and leaks stray escape bytes onto the screen.
	glamourStyle := "light"
	if lipgloss.HasDarkBackground() {
		glamourStyle = "dark"
	}
	program := tea.NewProgram(ui.NewModel(submit, bannerInfo(cfg, sess), glamourStyle), tea.WithAltScreen(), tea.WithContext(ctx))

	renderer := &ui.TUIRenderer{Program: program}
	approver := &ui.TUIApprover{Program: program}
	ag := agent.New(provider, registry, h, systemPrompt, defaultMaxTokens, guards, defaultToolTimeout, approver)
	if err := wireSubAgent(registry, provider, ag); err != nil {
		return err
	}
	ag.SetCompactionPolicy(compactionPolicy(cfg))
	ag.SetThink(cfg.Think)

	go func() {
		for input := range submit {
			if name, arg, ok := slashCommand(input); ok {
				output, cmdErr := runTUICommand(ctx, ag, h, sess, input, name, arg)
				if errors.Is(cmdErr, errClearScrollback) {
					ui.ClearScrollback(program)
				} else {
					ui.CommandOutput(program, output)
				}
				if errors.Is(cmdErr, errExit) {
					program.Quit()
				}
				continue
			}

			turnCtx, cancel := context.WithCancel(ctx)
			ui.TurnStarted(program, cancel)
			text, err := ag.RunStream(turnCtx, input, renderer)
			cancel()
			if err != nil {
				err = friendlyAgentError(cfg, err)
			} else {
				err = sess.saveIfActive(h)
			}
			ui.TurnDone(program, text, err)
		}
	}()

	_, err := program.Run()
	close(submit)
	return err
}

// runTUICommand runs a parsed slash command and formats its output for the
// TUI's transcript, prefixed with the command line itself (the plain
// REPL's prompt naturally shows what was typed; the TUI doesn't echo user
// input into its transcript at all, so a command needs to say what it is).
// The returned error is never a write failure (bytes.Buffer.Write can't
// fail) — it's only ever nil or one of handleCommand's sentinels, which the
// caller uses to clear the transcript or quit the program instead of
// appending the output. Split out from runTUIChat's driver goroutine so
// it's testable without a running *tea.Program.
func runTUICommand(ctx context.Context, ag *agent.Agent, h *history.History, sess *session, input, name, arg string) (string, error) {
	var buf bytes.Buffer
	_, _ = fmt.Fprintf(&buf, "> %s\n", input)
	_, err := handleCommand(ctx, &buf, ag, h, sess, name, arg)
	return buf.String(), err
}
