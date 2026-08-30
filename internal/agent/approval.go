package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Approver decides whether a tool call needing approval (see
// tools.ApprovalRequired) proceeds. CLAUDE.md invariant 5 makes the agent
// loop the sole owner of this decision — a Tool itself never prompts.
type Approver interface {
	Approve(ctx context.Context, toolName string, input json.RawMessage, preview string) (bool, error)
}

type autoApprove struct{}

func (autoApprove) Approve(context.Context, string, json.RawMessage, string) (bool, error) {
	return true, nil
}

// AutoApprove approves every call without prompting — used for --yes.
var AutoApprove Approver = autoApprove{}

// ConsoleApprover prompts on Out and reads a y/N answer from In.
type ConsoleApprover struct {
	In  io.Reader
	Out io.Writer
}

// Approve implements Approver.
func (c ConsoleApprover) Approve(_ context.Context, toolName string, input json.RawMessage, preview string) (bool, error) {
	if _, err := fmt.Fprintf(c.Out, "Approve call to %q with input %s?\n", toolName, input); err != nil {
		return false, fmt.Errorf("agent: write approval prompt: %w", err)
	}
	if preview != "" {
		if _, err := fmt.Fprintln(c.Out, preview); err != nil {
			return false, fmt.Errorf("agent: write approval preview: %w", err)
		}
	}
	if _, err := fmt.Fprint(c.Out, "[y/N]: "); err != nil {
		return false, fmt.Errorf("agent: write approval prompt: %w", err)
	}

	line, err := bufio.NewReader(c.In).ReadString('\n')
	if err != nil && err != io.EOF {
		return false, fmt.Errorf("agent: read approval answer: %w", err)
	}
	answer := strings.ToLower(strings.TrimSpace(line))
	return answer == "y" || answer == "yes", nil
}

var _ Approver = ConsoleApprover{}
