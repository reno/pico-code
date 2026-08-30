package main

import (
	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/version"
)

// newRootCmd builds the pico root command. Starting a chat session is
// pico's only behavior, so its flags and RunE are wired directly onto root
// instead of a "chat" subcommand — getenv is injected so tests can exercise
// env fallback without touching the process environment.
func newRootCmd(getenv func(string) string) *cobra.Command {
	root := &cobra.Command{
		Use:           "pico",
		Short:         "pico is a small CLI AI agent",
		Version:       version.Version,
		SilenceUsage:  true,
		SilenceErrors: false,
	}
	configureChat(root, getenv)
	return root
}
