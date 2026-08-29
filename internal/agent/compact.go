package agent

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
)

// CompactionPolicy configures when Run/RunStream summarize the oldest
// turns to keep context small. The zero value (ContextWindow == 0)
// disables it.
type CompactionPolicy struct {
	// ContextWindow is the provider's context window, in tokens.
	ContextWindow int
	// TriggerFraction is the fraction of ContextWindow that, once
	// estimated usage exceeds it, triggers a compaction.
	TriggerFraction float64
	// KeepTurns is how many of the most recent user turns are kept
	// verbatim; everything older is summarized into one message.
	KeepTurns int
}

// SetCompactionPolicy configures p for future turns; the zero value
// disables compaction. It is not a New parameter since most callers
// (every existing test, and any caller not tracking context budget) don't
// need it.
func (a *Agent) SetCompactionPolicy(p CompactionPolicy) {
	a.compaction = p
}

// CompactionPolicy returns the policy currently in effect, so a caller (the
// /model command) can change just ContextWindow without having to know or
// clobber TriggerFraction and KeepTurns.
func (a *Agent) CompactionPolicy() CompactionPolicy {
	return a.compaction
}

// maybeCompact summarizes the oldest turns into a single synthetic message
// once history's estimated size crosses the configured threshold. It is
// best-effort: a failed summarization call (or a disabled/inapplicable
// policy) leaves history untouched rather than failing the turn — the
// worst case is simply not compacting yet.
func (a *Agent) maybeCompact(ctx context.Context) {
	p := a.compaction
	if p.ContextWindow <= 0 || p.TriggerFraction <= 0 || p.KeepTurns <= 0 {
		return
	}

	messages := a.history.Snapshot()
	threshold := int(float64(p.ContextWindow) * p.TriggerFraction)
	if history.EstimateTokens(messages) <= threshold {
		return
	}

	if err := a.compactNow(ctx, messages, p.KeepTurns); err != nil {
		slog.Warn("agent: skipping compaction", "error", err)
	}
}

// ForceCompact runs the same summarization pass as maybeCompact, but
// unconditionally rather than waiting for the trigger threshold — for the
// /compact command (CLAUDE.md phase 11.2). It reports the estimated token
// count before and after so the caller can show the delta. If there are
// KeepTurns or fewer turns to summarize (including when no policy has been
// configured), it is a no-op and before == after.
func (a *Agent) ForceCompact(ctx context.Context) (before, after int, err error) {
	messages := a.history.Snapshot()
	before = history.EstimateTokens(messages)

	if err := a.compactNow(ctx, messages, a.compaction.KeepTurns); err != nil {
		return before, before, err
	}
	return before, history.EstimateTokens(a.history.Snapshot()), nil
}

// compactNow summarizes everything before the boundary that keeps the last
// keepTurns turns verbatim, then commits it via history.Compact. A no-op if
// there are keepTurns or fewer turns.
func (a *Agent) compactNow(ctx context.Context, messages []llm.Message, keepTurns int) error {
	if keepTurns <= 0 {
		return nil
	}
	boundary := history.CompactionBoundary(messages, keepTurns)
	if boundary <= 0 {
		return nil
	}

	summary, err := a.summarize(ctx, messages[:boundary])
	if err != nil {
		return fmt.Errorf("agent: summarize: %w", err)
	}
	return a.history.Compact(keepTurns, summary)
}

func (a *Agent) summarize(ctx context.Context, elided []llm.Message) (string, error) {
	prompt := buildSummarizationPrompt(elided)
	resp, err := a.provider.Chat(ctx, llm.Request{
		System:    a.system,
		Messages:  []llm.Message{{Role: llm.RoleUser, Blocks: []llm.Block{llm.Text{Text: prompt}}}},
		MaxTokens: a.maxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("agent: summarize: %w", err)
	}
	return textOf(resp.Message), nil
}

// summarizationPromptTemplate is pinned by a golden test: changing its
// wording is a deliberate choice, not an incidental refactor.
const summarizationPromptTemplate = `Summarize the earlier part of this conversation between you (the assistant) and a user, so the summary can replace the original messages in your context. Preserve any facts, decisions, file paths, commands, or conclusions that later turns might depend on. Do not include pleasantries or filler. Write the summary as plain prose, not a transcript.

Conversation to summarize:
%s`

func buildSummarizationPrompt(messages []llm.Message) string {
	return fmt.Sprintf(summarizationPromptTemplate, renderTranscript(messages))
}

func renderTranscript(messages []llm.Message) string {
	var b strings.Builder
	for _, m := range messages {
		for _, blk := range m.Blocks {
			switch v := blk.(type) {
			case llm.Text:
				fmt.Fprintf(&b, "%s: %s\n", m.Role, v.Text)
			case llm.ToolUse:
				fmt.Fprintf(&b, "%s: [called tool %s with %s]\n", m.Role, v.Name, v.Input)
			case llm.ToolResult:
				status := "ok"
				if v.IsError {
					status = "error"
				}
				fmt.Fprintf(&b, "%s: [tool result (%s): %s]\n", m.Role, status, v.Content)
			}
		}
	}
	return strings.TrimRight(b.String(), "\n")
}
