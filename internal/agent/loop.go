// Package agent implements the core loop: call the provider, and if its
// reply carries no tool calls, hand back the text; otherwise run every tool
// call from that reply, feed the results back, and repeat. It owns policy
// (CLAUDE.md layer ownership) — providers only translate wire formats and
// tools only execute; the loop decides when to stop and how results get
// reassembled.
package agent

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// Agent drives one conversation: a provider to talk to, a tool registry to
// satisfy its tool calls, and the history both read from and append to.
type Agent struct {
	provider  llm.Provider
	tools     *tools.Registry
	history   *history.History
	system    string
	maxTokens int
}

// New returns an Agent ready to run turns against provider, using registry
// to execute any tool call it issues, appending to h. system and maxTokens
// are sent on every Request the loop builds.
func New(provider llm.Provider, registry *tools.Registry, h *history.History, system string, maxTokens int) *Agent {
	return &Agent{provider: provider, tools: registry, history: h, system: system, maxTokens: maxTokens}
}

// Run appends userInput to history as a user turn, then drives the
// call-provider / run-tools cycle to completion: each round appends the
// provider's reply to history, and if that reply has no ToolUse blocks,
// Run returns its text. Otherwise every ToolUse in that reply runs
// concurrently, the results are appended — in the original call order,
// regardless of which tool finished first — as the next user message, and
// the loop repeats.
func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: userInput}}})

	for {
		resp, err := a.provider.Chat(ctx, llm.Request{
			System:    a.system,
			Messages:  a.history.Snapshot(),
			Tools:     a.tools.Definitions(),
			MaxTokens: a.maxTokens,
		})
		if err != nil {
			return "", fmt.Errorf("agent: chat: %w", err)
		}
		a.history.Append(resp.Message)

		calls := toolUseBlocks(resp.Message)
		if len(calls) == 0 {
			return textOf(resp.Message), nil
		}

		a.history.Append(llm.Message{Role: llm.RoleUser, Blocks: a.runTools(ctx, calls)})
	}
}

// runTools executes every call concurrently and returns their ToolResult
// blocks in calls' original order — CLAUDE.md invariant 3 requires results
// answer their ToolUse in call order, independent of completion order.
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolUse) []llm.Block {
	results := make([]llm.Block, len(calls))
	var g errgroup.Group
	for i, call := range calls {
		g.Go(func() error {
			results[i] = a.runTool(ctx, call)
			return nil
		})
	}
	// Every goroutine above always returns nil: a tool failure becomes an
	// IsError ToolResult (CLAUDE.md invariant 4, "tool failures are data,
	// not control flow"), never a Go error that could short-circuit the
	// group and leave a result slot empty.
	_ = g.Wait()
	return results
}

func (a *Agent) runTool(ctx context.Context, call llm.ToolUse) llm.Block {
	out, err := a.tools.Run(ctx, call.Name, call.Input)
	if err != nil {
		return llm.ToolResult{ToolUseID: call.ID, Content: err.Error(), IsError: true}
	}
	return llm.ToolResult{ToolUseID: call.ID, Content: out, IsError: false}
}

func toolUseBlocks(m llm.Message) []llm.ToolUse {
	var calls []llm.ToolUse
	for _, b := range m.Blocks {
		if tu, ok := b.(llm.ToolUse); ok {
			calls = append(calls, tu)
		}
	}
	return calls
}

func textOf(m llm.Message) string {
	var b strings.Builder
	for _, blk := range m.Blocks {
		t, ok := blk.(llm.Text)
		if !ok {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(t.Text)
	}
	return b.String()
}
