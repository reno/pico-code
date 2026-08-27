package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/history"
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
func handleCommand(out io.Writer, ag *agent.Agent, h *history.History, sess *session, name, arg string) (handled bool, err error) {
	switch name {
	case "usage":
		printUsage(out, ag)
	case "new":
		h.Reset()
		sess.name = ""
		_, err = fmt.Fprintln(out, "started a new, unsaved session")
	case "save":
		err = runSaveCommand(out, h, sess, arg)
	case "load":
		err = runLoadCommand(out, h, sess, arg)
	default:
		_, err = fmt.Fprintf(out, "unknown command /%s\n", name)
	}
	return true, err
}

func runSaveCommand(out io.Writer, h *history.History, sess *session, arg string) error {
	name := arg
	if name == "" {
		name = sess.name
	}
	if name == "" {
		_, err := fmt.Fprintln(out, "usage: /save <name> (no active session to save without one)")
		return err
	}
	if err := h.Save(sess.path(name)); err != nil {
		_, werr := fmt.Fprintf(out, "save failed: %v\n", err)
		return werr
	}
	sess.name = name
	_, err := fmt.Fprintf(out, "saved session %q\n", name)
	return err
}

func runLoadCommand(out io.Writer, h *history.History, sess *session, arg string) error {
	if arg == "" {
		_, err := fmt.Fprintln(out, "usage: /load <name>")
		return err
	}
	if err := h.LoadInto(sess.path(arg)); err != nil {
		_, werr := fmt.Fprintf(out, "load failed: %v\n", err)
		return werr
	}
	sess.name = arg
	_, err := fmt.Fprintf(out, "loaded session %q (%d message(s))\n", arg, len(h.Snapshot()))
	return err
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
