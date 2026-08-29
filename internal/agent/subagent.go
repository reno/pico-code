package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// subAgentMaxTurns bounds a sub-agent's own turn count, independent of the
// parent's MaxTurns unless the parent's is smaller — a sub-agent should
// never be allowed to outlast the loop that spawned it.
const subAgentMaxTurns = 10

// SubAgentInput is sub_agent's argument shape: one self-contained task. The
// sub-agent has no access to the parent conversation's history, so
// everything it needs to know has to be in Task.
type SubAgentInput struct {
	Task string `json:"task" jsonschema:"description=A complete, self-contained task for the sub-agent. It cannot see this conversation, so state everything it needs to know."`
}

// SubAgentTool is a tools.Tool whose Run drives a nested Agent for a
// delegated task: a fresh, empty history (CLAUDE.md invariant 3 applies to
// it independently — the parent's history never contains the child's
// messages, only this tool's single flattened string result), a restricted
// tool set, and a turn/token budget carved from the parent Agent's
// configured Guards.
//
// It satisfies tools.Tool but lives in internal/agent rather than
// internal/tools because building the nested loop needs agent.New, and
// internal/agent already imports internal/tools — the reverse import would
// cycle.
type SubAgentTool struct {
	provider    llm.Provider
	registry    *tools.Registry
	system      string
	maxTokens   int
	toolTimeout time.Duration
	parent      *Agent
	schema      json.RawMessage
}

// NewSubAgentTool returns a SubAgentTool that hands a delegated task to a
// nested Agent talking to the same provider as parent. allowedTools is the
// sub-agent's entire tool set — there is no built-in default, since what's
// safe to delegate (and, per CLAUDE.md's safety rules, safe to run without
// an interactive approval prompt: Run always approves via AutoApprove) is a
// caller decision. Omit sub_agent itself to keep nesting bounded.
func NewSubAgentTool(provider llm.Provider, allowedTools []tools.Tool, system string, maxTokens int, toolTimeout time.Duration, parent *Agent) (*SubAgentTool, error) {
	registry := tools.NewRegistry()
	for _, t := range allowedTools {
		if err := registry.Register(t); err != nil {
			return nil, fmt.Errorf("agent: sub_agent tool set: %w", err)
		}
	}
	schema, err := tools.GenerateSchema(SubAgentInput{})
	if err != nil {
		return nil, fmt.Errorf("agent: sub_agent schema: %w", err)
	}
	return &SubAgentTool{
		provider:    provider,
		registry:    registry,
		system:      system,
		maxTokens:   maxTokens,
		toolTimeout: toolTimeout,
		parent:      parent,
		schema:      schema,
	}, nil
}

// Name implements tools.Tool.
func (t *SubAgentTool) Name() string { return "sub_agent" }

// Description implements tools.Tool.
func (t *SubAgentTool) Description() string {
	return "Delegate a self-contained task to a nested sub-agent with its own turn and token budget and a restricted tool set. Only the sub-agent's final answer comes back, not its intermediate tool calls."
}

// Schema implements tools.Tool.
func (t *SubAgentTool) Schema() json.RawMessage { return t.schema }

// Run drives the nested Agent to completion and returns its final text.
// Cancelling ctx tears the nested Agent down the same way it would the
// parent — nothing extra is wired for that, since it's the same ctx chain
// (the nested Agent's own tool goroutines are abandoned on cancellation by
// the same doRunTool logic recursively).
//
// A guard tripping inside the nested Agent (its budget exhausted before it
// produced a final answer) comes back as an error here rather than as a
// normal result, so the parent records ToolResult{IsError:true} instead of
// silently accepting a cut-off answer as if it were complete.
func (t *SubAgentTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in SubAgentInput
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("agent: sub_agent: %w", err)
	}

	child := New(t.provider, t.registry, history.New(), t.system, t.maxTokens, t.childGuards(), t.toolTimeout, AutoApprove)
	text, err := child.Run(ctx, in.Task)
	if err != nil {
		return "", fmt.Errorf("agent: sub_agent: %w", err)
	}
	if child.GuardTripped() {
		return "", fmt.Errorf("agent: sub_agent exhausted its budget before finishing: %s", text)
	}
	return tools.TruncateBytes(text, tools.DefaultByteBudget), nil
}

// childGuards carves a budget for the nested Agent out of the parent's
// configured Guards and usage so far. MaxTurns is capped at
// subAgentMaxTurns, further capped by the parent's own MaxTurns if that's
// smaller and set. TokenBudget is what's left of the parent's configured
// budget after its completed turns' usage — an approximation, since the
// parent's own in-flight turn (the one that issued this very call) hasn't
// recorded its usage yet, only prior turns have. An unset (0 = unlimited)
// parent TokenBudget carries over as unlimited for the child too, rather
// than being read as "nothing left."
func (t *SubAgentTool) childGuards() Guards {
	pg := t.parent.Guards()

	maxTurns := subAgentMaxTurns
	if pg.MaxTurns > 0 && pg.MaxTurns < maxTurns {
		maxTurns = pg.MaxTurns
	}

	var tokenBudget int
	if pg.TokenBudget > 0 {
		spent := t.parent.CumulativeUsage()
		tokenBudget = pg.TokenBudget - spent.InputTokens - spent.OutputTokens
		if tokenBudget < 1 {
			tokenBudget = 1
		}
	}

	return Guards{MaxTurns: maxTurns, TokenBudget: tokenBudget}
}

var _ tools.Tool = (*SubAgentTool)(nil)
