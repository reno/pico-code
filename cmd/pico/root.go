package main

import (
	"github.com/spf13/cobra"
)

// newRootCmd builds the pico root command with all subcommands attached.
// getenv is injected so tests can exercise env fallback without touching
// the process environment.
func newRootCmd(getenv func(string) string) *cobra.Command {
	root := &cobra.Command{
		Use:           "pico",
		Short:         "pico code is a small CLI AI agent",
		SilenceUsage:  true,
		SilenceErrors: false,
	}
	root.AddCommand(newChatCmd(getenv))
	return root
}
