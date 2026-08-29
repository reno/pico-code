package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// errExit and errClearScrollback are handleCommand's two sentinel results:
// signals to the caller's own loop rather than failures. Neither commands.go
// action can be expressed by writing to out alone — exiting has to unwind
// the plain REPL's scanner loop or quit the bubbletea program, and clearing
// has to reset a real terminal or the TUI's transcript state, both owned by
// the caller, not by this UI-agnostic file.
var (
	errExit            = errors.New("exit requested")
	errClearScrollback = errors.New("clear scrollback requested")
)

// commandSpec describes one slash command. Adding a command to the chat
// means adding one entry to commandTable — handleCommand and /help both
// drive off it, so nothing else in this file needs to change.
type commandSpec struct {
	name    string
	args    string
	summary string
	run     func(ctx context.Context, out io.Writer, ag *agent.Agent, h *history.History, sess *session, arg string) error
}

// commandTable is populated in init(), not a var initializer: runHelpCommand
// reads commandTable, and a direct function-value reference inside the
// initializer would make that a compile-time initialization cycle.
var commandTable []commandSpec

func init() {
	commandTable = []commandSpec{
		{name: "help", summary: "list available commands", run: runHelpCommand},
		{name: "usage", summary: "show cumulative and last-turn token usage", run: runUsageCommand},
		{name: "new", summary: "start a new, unsaved session", run: runNewCommand},
		{name: "save", args: "[name]", summary: "save the current session to disk", run: runSaveCommand},
		{name: "load", args: "<name>", summary: "load a saved session from disk", run: runLoadCommand},
		{name: "clear", summary: "clear the terminal scrollback (the conversation history is kept)", run: runClearCommand},
		{name: "compact", summary: "summarize older turns now and report the token estimate", run: runCompactCommand},
		{name: "exit", summary: "save the session and exit, like Ctrl+D", run: runExitCommand},
		{name: "cd", args: "<path>", summary: "re-root the workspace sandbox", run: runCdCommand},
		{name: "model", args: "<name> [context-window]", summary: "switch models, validated against the active provider", run: runModelCommand},
		{name: "mcp", args: "[reconnect <name>]", summary: "list configured MCP servers and their status, or reconnect one", run: runMcpCommand},
	}
}

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
// always returns handled == true, even for an unknown command (reported with
// its closest match by name rather than sent to the model as chat input).
// err is either a genuine failure or one of the sentinels above; callers
// that don't special-case those sentinels should still treat a non-nil err
// as fatal to the current turn/command, same as any other error.
func handleCommand(ctx context.Context, out io.Writer, ag *agent.Agent, h *history.History, sess *session, name, arg string) (handled bool, err error) {
	for _, c := range commandTable {
		if c.name == name {
			return true, c.run(ctx, out, ag, h, sess, arg)
		}
	}
	_, err = fmt.Fprintf(out, "unknown command /%s, did you mean /%s?\n", name, closestCommand(name))
	return true, err
}

// runHelpCommand renders commandTable as an aligned list of "/name args"
// followed by its summary.
func runHelpCommand(_ context.Context, out io.Writer, _ *agent.Agent, _ *history.History, _ *session, _ string) error {
	tw := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	for _, c := range commandTable {
		usage := "/" + c.name
		if c.args != "" {
			usage += " " + c.args
		}
		if _, err := fmt.Fprintf(tw, "%s\t%s\n", usage, c.summary); err != nil {
			return err
		}
	}
	return tw.Flush()
}

func runUsageCommand(_ context.Context, out io.Writer, ag *agent.Agent, _ *history.History, _ *session, _ string) error {
	printUsage(out, ag)
	return nil
}

func runNewCommand(_ context.Context, out io.Writer, _ *agent.Agent, h *history.History, sess *session, _ string) error {
	h.Reset()
	sess.name = ""
	_, err := fmt.Fprintln(out, "started a new, unsaved session")
	return err
}

func runSaveCommand(_ context.Context, out io.Writer, _ *agent.Agent, h *history.History, sess *session, arg string) error {
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

func runLoadCommand(_ context.Context, out io.Writer, _ *agent.Agent, h *history.History, sess *session, arg string) error {
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

// runClearCommand never touches out itself: clearing a real terminal's
// scrollback and clearing the TUI's transcript are different operations
// owned by the two callers (runREPL, runTUICommand), which react to
// errClearScrollback instead.
func runClearCommand(_ context.Context, _ io.Writer, _ *agent.Agent, _ *history.History, _ *session, _ string) error {
	return errClearScrollback
}

// runCompactCommand forces the same summarization pass maybeCompact runs
// automatically (agent.CompactionPolicy, CLAUDE.md phase 8.2) and reports
// the estimated token count before and after.
func runCompactCommand(ctx context.Context, out io.Writer, ag *agent.Agent, _ *history.History, _ *session, _ string) error {
	before, after, err := ag.ForceCompact(ctx)
	if err != nil {
		_, werr := fmt.Fprintf(out, "compact failed: %v\n", err)
		return werr
	}
	if before == after {
		_, err := fmt.Fprintf(out, "nothing to compact (estimated %d token(s))\n", before)
		return err
	}
	_, err = fmt.Fprintf(out, "compacted: %d -> %d estimated token(s)\n", before, after)
	return err
}

// runExitCommand saves the session first, then signals errExit so the
// caller unwinds through the same path as Ctrl+D. If the save fails, it
// reports the failure and does not signal exit, so a save failure never
// silently drops the session.
func runExitCommand(_ context.Context, out io.Writer, _ *agent.Agent, h *history.History, sess *session, _ string) error {
	if err := sess.saveIfActive(h); err != nil {
		_, werr := fmt.Fprintf(out, "exit: session save failed: %v\n", err)
		return werr
	}
	return errExit
}

// sandboxProvider is satisfied by any registered tool that exposes the
// shared workspace Sandbox — every filesystem tool (read_file, list_dir,
// write_file) is constructed with the same *tools.Sandbox pointer, so
// reaching it through any one of them re-roots every filesystem tool at
// once. read_file is always registered (buildRegistry never gates it
// behind a flag, unlike write_file/run_command), so it's the one this
// looks up.
type sandboxProvider interface {
	Sandbox() *tools.Sandbox
}

// runCdCommand re-roots the workspace sandbox to arg, refusing (and leaving
// the sandbox untouched) if it doesn't exist or can't be resolved —
// Sandbox.Reroot already enforces that the same way NewSandbox does at
// startup.
func runCdCommand(_ context.Context, out io.Writer, ag *agent.Agent, _ *history.History, _ *session, arg string) error {
	if arg == "" {
		_, err := fmt.Fprintln(out, "usage: /cd <path>")
		return err
	}
	t, err := ag.Tools().Get("read_file")
	if err != nil {
		_, werr := fmt.Fprintf(out, "cd failed: %v\n", err)
		return werr
	}
	sb, ok := t.(sandboxProvider)
	if !ok {
		_, werr := fmt.Fprintln(out, "cd failed: workspace is not sandboxed")
		return werr
	}
	if err := sb.Sandbox().Reroot(arg); err != nil {
		_, werr := fmt.Fprintf(out, "cd failed: %v\n", err)
		return werr
	}
	_, err = fmt.Fprintf(out, "workspace root changed to %s\n", arg)
	return err
}

// runModelCommand switches the active provider's model, validating the
// name first (llm.ModelSwitcher.ValidateModel) so an unknown model errors
// without mutating the provider or the compaction policy. arg is "<name>"
// or "<name> <context-window>"; the window is otherwise left unchanged,
// since none of this codebase's providers can safely auto-detect one for
// every backend (Anthropic can via its Models API; Ollama can't, per
// ValidateModel's doc comment) and CLAUDE.md already treats a model's
// context window as an explicit, user-set value (--context-window,
// --num-ctx) rather than one inferred from the model name.
func runModelCommand(ctx context.Context, out io.Writer, ag *agent.Agent, h *history.History, _ *session, arg string) error {
	fields := strings.Fields(arg)
	if len(fields) == 0 || len(fields) > 2 {
		_, err := fmt.Fprintln(out, "usage: /model <name> [context-window]")
		return err
	}
	name := fields[0]

	switcher, ok := ag.Provider().(llm.ModelSwitcher)
	if !ok {
		_, err := fmt.Fprintf(out, "model failed: provider %q does not support switching models\n", ag.Provider().Name())
		return err
	}
	if err := switcher.ValidateModel(ctx, name); err != nil {
		_, werr := fmt.Fprintf(out, "model failed: %v\n", err)
		return werr
	}

	policy := ag.CompactionPolicy()
	if len(fields) == 2 {
		window, err := strconv.Atoi(fields[1])
		if err != nil || window <= 0 {
			_, werr := fmt.Fprintf(out, "model failed: invalid context window %q\n", fields[1])
			return werr
		}
		policy.ContextWindow = window
	}

	switcher.SetModel(name)
	ag.SetCompactionPolicy(policy)

	if policy.ContextWindow > 0 {
		if used := history.EstimateTokens(h.Snapshot()); used > policy.ContextWindow {
			_, err := fmt.Fprintf(out, "switched model to %s (context window %d); warning: current usage (~%d tokens) already exceeds it, compaction will run on the next turn\n", name, policy.ContextWindow, used)
			return err
		}
	}
	_, err := fmt.Fprintf(out, "switched model to %s\n", name)
	return err
}

// runMcpCommand lists configured MCP servers and their status (connected
// with a tool count, or failed with the error), or — given "reconnect
// <name>" — re-runs discovery for just that server first. sess.mcp is nil
// whenever --mcp-config named no servers (or wasn't passed at all), which
// this reports the same way as "servers configured but none reachable"
// would: there's nothing more a user needs to distinguish between the two.
func runMcpCommand(ctx context.Context, out io.Writer, _ *agent.Agent, _ *history.History, sess *session, arg string) error {
	if fields := strings.Fields(arg); len(fields) > 0 {
		if sess.mcp == nil {
			_, err := fmt.Fprintln(out, "no MCP servers configured")
			return err
		}
		if fields[0] != "reconnect" || len(fields) != 2 {
			_, err := fmt.Fprintln(out, "usage: /mcp [reconnect <name>]")
			return err
		}
		if err := sess.mcp.reconnect(ctx, fields[1]); err != nil {
			_, werr := fmt.Fprintf(out, "reconnect failed: %v\n", err)
			return werr
		}
	}

	if sess.mcp == nil {
		_, err := fmt.Fprintln(out, "no MCP servers configured")
		return err
	}
	statuses := sess.mcp.statuses()
	if len(statuses) == 0 {
		_, err := fmt.Fprintln(out, "no MCP servers configured")
		return err
	}

	tw := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	for _, s := range statuses {
		if s.connected {
			if _, err := fmt.Fprintf(tw, "%s\tconnected\t%d tool(s)\n", s.name, s.toolCount); err != nil {
				return err
			}
			continue
		}
		if _, err := fmt.Fprintf(tw, "%s\tfailed\t%v\n", s.name, s.err); err != nil {
			return err
		}
	}
	return tw.Flush()
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

// closestCommand returns the commandTable entry whose name is nearest to
// name by Levenshtein edit distance, for suggesting a fix on a likely typo.
func closestCommand(name string) string {
	best := commandTable[0].name
	bestDist := levenshtein(name, best)
	for _, c := range commandTable[1:] {
		if d := levenshtein(name, c.name); d < bestDist {
			bestDist = d
			best = c.name
		}
	}
	return best
}

// levenshtein computes the edit distance between a and b.
func levenshtein(a, b string) int {
	ar, br := []rune(a), []rune(b)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		for j := 1; j <= len(br); j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[len(br)]
}
