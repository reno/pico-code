package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
	"github.com/reno/pico-code/internal/ui"
)

const (
	systemPrompt       = "You are pico code, a terminal coding agent. Be direct and concise."
	defaultMaxTokens   = 4096
	defaultToolTimeout = 2 * time.Minute
)

// newChatCmd builds the chat subcommand. Flag values win over the
// PICO_CODE_PROVIDER environment fallback, which wins over the built-in
// default; ANTHROPIC_API_KEY and OLLAMA_HOST are read directly by
// config.Load since they are credentials, not flags.
func newChatCmd(getenv func(string) string) *cobra.Command {
	var flags struct {
		provider    string
		model       string
		maxTurns    int
		tokenBudget int
		workspace   string
		yes         bool
		tools       string
		stream      bool
		tui         bool
		logLevel    string
		numCtx      int
		allowWrite  bool
		session     string
	}

	cmd := &cobra.Command{
		Use:   "chat",
		Short: "Start an interactive chat session with the agent",
		RunE: func(cmd *cobra.Command, _ []string) error {
			provider := flags.provider
			if !cmd.Flags().Changed("provider") {
				if v := getenv("PICO_CODE_PROVIDER"); v != "" {
					provider = v
				}
			}

			cfg, err := config.Load(config.Flags{
				Provider:    provider,
				Model:       flags.model,
				MaxTurns:    flags.maxTurns,
				TokenBudget: flags.tokenBudget,
				Workspace:   flags.workspace,
				Yes:         flags.yes,
				Tools:       flags.tools,
				Stream:      flags.stream,
				TUI:         flags.tui,
				LogLevel:    flags.logLevel,
				NumCtx:      flags.numCtx,
				AllowWrite:  flags.allowWrite,
				Session:     flags.session,
			}, getenv)
			if err != nil {
				return fmt.Errorf("resolving config: %w", err)
			}

			return runChat(cmd, cfg)
		},
	}

	f := cmd.Flags()
	f.StringVar(&flags.provider, "provider", "anthropic", "LLM backend to use (anthropic|ollama)")
	f.StringVar(&flags.model, "model", "", "model identifier for the selected provider")
	f.IntVar(&flags.maxTurns, "max-turns", 25, "maximum agent loop turns before stopping")
	f.IntVar(&flags.tokenBudget, "token-budget", 100_000, "maximum cumulative tokens before stopping")
	f.StringVar(&flags.workspace, "workspace", ".", "root directory filesystem tools are confined to")
	f.BoolVar(&flags.yes, "yes", false, "skip interactive approval prompts")
	f.StringVar(&flags.tools, "tools", "native", "tool-calling mode (native|prompted)")
	f.BoolVar(&flags.stream, "stream", true, "stream provider responses")
	f.BoolVar(&flags.tui, "tui", false, "use the bubbletea TUI instead of plain output")
	f.StringVar(&flags.logLevel, "log-level", "info", "log level (debug|info|warn|error)")
	f.IntVar(&flags.numCtx, "num-ctx", 4096, "context window size passed to Ollama's num_ctx (ignored by other providers)")
	f.BoolVar(&flags.allowWrite, "allow-write", false, "register the write_file tool (off by default)")
	f.StringVar(&flags.session, "session", "", "name a session to resume or start; saved after every turn")

	return cmd
}

// runChat is a package var so tests can substitute a stub instead of
// driving a real provider/agent turn end to end.
var runChat = func(cmd *cobra.Command, cfg *config.Config) error {
	if err := setupLogging(cmd.ErrOrStderr(), cfg.LogLevel); err != nil {
		return err
	}

	provider, err := llm.New(cfg)
	if err != nil {
		return fmt.Errorf("resolving provider: %w", err)
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

// defaultAnthropicContextWindow is a conservative stand-in for the real
// per-model context window: config has no field for it (only Ollama's
// NumCtx, since CLAUDE.md requires that one explicitly), so compaction
// against Anthropic uses this constant rather than an actual model limit.
const (
	defaultAnthropicContextWindow = 200_000
	compactionTriggerFraction     = 0.75
	compactionKeepTurns           = 6
)

func compactionPolicy(cfg *config.Config) agent.CompactionPolicy {
	window := defaultAnthropicContextWindow
	if cfg.Provider == config.ProviderOllama {
		window = cfg.NumCtx
	}
	return agent.CompactionPolicy{ContextWindow: window, TriggerFraction: compactionTriggerFraction, KeepTurns: compactionKeepTurns}
}

// buildRegistry wires the built-in tools this session offers. run_command
// is deliberately left out: CLAUDE.md requires it to have a binary
// allowlist from config, and no config flag for one exists yet — a real
// gap, not an oversight, worth a TASKS.md follow-up rather than a silent
// empty allowlist that would make the tool present but useless.
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
	if cfg.AllowWrite {
		writeTool, err := tools.NewWriteFileTool(sandbox)
		if err != nil {
			return nil, err
		}
		if err := registry.Register(writeTool); err != nil {
			return nil, err
		}
	}
	return registry, nil
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
	ag.SetCompactionPolicy(compactionPolicy(cfg))
	renderer := ui.PlainRenderer{Out: cmd.OutOrStdout()}
	ctx := cmd.Context()
	in := cmd.InOrStdin()

	if !isInteractive(in) {
		data, err := io.ReadAll(in)
		if err != nil {
			return fmt.Errorf("reading stdin: %w", err)
		}
		input := strings.TrimSpace(string(data))
		if input == "" {
			return nil
		}
		if _, err := ag.RunStream(ctx, input, renderer); err != nil {
			return err
		}
		return sess.saveIfActive(h)
	}

	return runREPL(ctx, in, cmd.OutOrStdout(), ag, h, sess, renderer)
}

// runREPL reads one line at a time from in, dispatching a leading slash
// command to handleCommand and everything else to ag.RunStream (saving to
// sess's file afterward, if one is active), until in hits EOF or ctx is
// cancelled. Split out from runPlainChat so a test can drive it directly
// without needing a real terminal for isInteractive.
func runREPL(ctx context.Context, in io.Reader, out io.Writer, ag *agent.Agent, h *history.History, sess *session, renderer ui.Renderer) error {
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
			if _, err := handleCommand(out, ag, h, sess, name, arg); err != nil {
				return err
			}
			prompt()
			continue
		}
		if _, err := ag.RunStream(ctx, line, renderer); err != nil {
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
	program := tea.NewProgram(ui.NewModel(submit), tea.WithAltScreen(), tea.WithContext(ctx))

	renderer := &ui.TUIRenderer{Program: program}
	approver := &ui.TUIApprover{Program: program}
	ag := agent.New(provider, registry, h, systemPrompt, defaultMaxTokens, guards, defaultToolTimeout, approver)
	ag.SetCompactionPolicy(compactionPolicy(cfg))

	go func() {
		for input := range submit {
			turnCtx, cancel := context.WithCancel(ctx)
			ui.TurnStarted(program, cancel)
			text, err := ag.RunStream(turnCtx, input, renderer)
			cancel()
			if err == nil {
				err = sess.saveIfActive(h)
			}
			ui.TurnDone(program, text, err)
		}
	}()

	_, err := program.Run()
	close(submit)
	return err
}
