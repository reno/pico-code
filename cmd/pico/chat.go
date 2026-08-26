package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/config"
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

	return cmd
}

// runChat is a placeholder until the agent loop (phase 4) exists.
func runChat(cmd *cobra.Command, cfg *config.Config) error {
	_, err := fmt.Fprintf(cmd.OutOrStdout(), "pico chat: provider=%s model=%s (agent loop not yet implemented)\n", cfg.Provider, cfg.Model)
	return err
}
