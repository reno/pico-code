package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/reno/pico-code/internal/agent"
)

// slashCommand parses a leading "/name arg" out of line. ok is false for
// ordinary chat input.
func slashCommand(line string) (name, arg string, ok bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "/") {
		return "", "", false
	}
	fields := strings.SplitN(strings.TrimPrefix(line, "/"), " ", 2)
	name = fields[0]
	if len(fields) > 1 {
		arg = strings.TrimSpace(fields[1])
	}
	return name, arg, name != ""
}

// handleCommand runs a parsed slash command, writing its output to out. It
// always returns handled == true, even for an unknown command (reported as
// an error line rather than sent to the model as chat input).
func handleCommand(out io.Writer, ag *agent.Agent, name, _ string) (handled bool, err error) {
	switch name {
	case "usage":
		printUsage(out, ag)
	default:
		_, err = fmt.Fprintf(out, "unknown command /%s\n", name)
	}
	return true, err
}

func printUsage(out io.Writer, ag *agent.Agent) {
	turns := ag.TurnUsages()
	cumulative := ag.CumulativeUsage()
	_, _ = fmt.Fprintf(out, "cumulative: %d input, %d output token(s) across %d turn(s)\n",
		cumulative.InputTokens, cumulative.OutputTokens, len(turns))
	if len(turns) > 0 {
		last := turns[len(turns)-1]
		_, _ = fmt.Fprintf(out, "last turn: %d input, %d output token(s)\n", last.InputTokens, last.OutputTokens)
	}
}
