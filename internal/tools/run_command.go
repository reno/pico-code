package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// ErrCommandNotAllowed is returned when the requested binary is not in the
// configured allowlist.
var ErrCommandNotAllowed = errors.New("tools: command not allowed")

// ErrShellMetacharacter is returned when the command or an argument
// contains a character that would be meaningful to a shell — run_command
// never invokes one (CLAUDE.md: "no shell interpretation"), so this always
// indicates an attempt to smuggle one in.
var ErrShellMetacharacter = errors.New("tools: shell metacharacter not allowed")

const shellMetaChars = "|&;<>()$`\\\"'*?[]#~!{}\n"

// RunCommandInput is run_command's argument shape.
type RunCommandInput struct {
	Command string   `json:"command" jsonschema:"description=The binary to execute; must be in the configured allowlist"`
	Args    []string `json:"args,omitempty" jsonschema:"description=Arguments passed to the command, unexpanded"`
}

// RunCommandTool runs an allowlisted binary directly (exec.CommandContext,
// no shell), with a timeout and output truncation.
type RunCommandTool struct {
	schema    json.RawMessage
	allowlist map[string]bool
	timeout   time.Duration
}

// NewRunCommandTool returns a RunCommandTool that only executes binaries
// named in allowlist, each bounded by timeout (0 = unlimited).
func NewRunCommandTool(allowlist []string, timeout time.Duration) (*RunCommandTool, error) {
	schema, err := GenerateSchema(RunCommandInput{})
	if err != nil {
		return nil, fmt.Errorf("tools: run_command schema: %w", err)
	}
	allowed := make(map[string]bool, len(allowlist))
	for _, b := range allowlist {
		allowed[b] = true
	}
	return &RunCommandTool{schema: schema, allowlist: allowed, timeout: timeout}, nil
}

// Name implements tools.Tool.
func (t *RunCommandTool) Name() string { return "run_command" }

// Description implements tools.Tool.
func (t *RunCommandTool) Description() string {
	return "Run an allowlisted command with no shell interpretation."
}

// Schema implements tools.Tool.
func (t *RunCommandTool) Schema() json.RawMessage { return t.schema }

// NeedsApproval implements tools.ApprovalRequired: every call to
// run_command needs sign-off.
func (t *RunCommandTool) NeedsApproval() bool { return true }

// Run executes the requested command, refusing anything outside the
// allowlist or containing a shell metacharacter before starting it. The
// result always reports an exit code — a nonzero exit is data the model
// sees, not a tool failure — except when the process is killed by timeout
// or never starts.
func (t *RunCommandTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in RunCommandInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("tools: run_command: %w", err)
	}

	if !t.allowlist[in.Command] {
		return "", fmt.Errorf("%w: %q", ErrCommandNotAllowed, in.Command)
	}
	if strings.ContainsAny(in.Command, shellMetaChars) {
		return "", fmt.Errorf("%w: in command %q", ErrShellMetacharacter, in.Command)
	}
	for _, arg := range in.Args {
		if strings.ContainsAny(arg, shellMetaChars) {
			return "", fmt.Errorf("%w: in argument %q", ErrShellMetacharacter, arg)
		}
	}

	runCtx := ctx
	if t.timeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, t.timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(runCtx, in.Command, in.Args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	runErr := cmd.Run()

	exitCode := 0
	var exitErr *exec.ExitError
	switch {
	case runCtx.Err() != nil:
		// A signal-killed process also surfaces as an *exec.ExitError, so
		// this must be checked before that case or a timeout gets
		// misreported as a normal nonzero exit.
		return truncateBytes(out.String(), defaultByteBudget), fmt.Errorf("tools: run_command %q timed out: %w", in.Command, runCtx.Err())
	case runErr == nil:
	case errors.As(runErr, &exitErr):
		exitCode = exitErr.ExitCode()
	default:
		return "", fmt.Errorf("tools: run_command %q: %w", in.Command, runErr)
	}

	result := fmt.Sprintf("exit code: %d\n%s", exitCode, truncateBytes(out.String(), defaultByteBudget))
	return result, nil
}

var _ ApprovalRequired = (*RunCommandTool)(nil)
