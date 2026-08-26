// Package prompted implements the --tools=prompted fallback for models
// with no native tool-calling support. It wraps any llm.Provider, moving a
// Request's tool schemas into the system prompt instead of the wrapped
// provider's native tools field, and parses a fenced ```json block out of
// the reply into the same canonical ToolUse blocks native mode would
// produce. The loop never has to know which mode is active — CLAUDE.md.
package prompted

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/reno/pico-code/internal/llm"
)

// parseErrorToolName is not a real, registerable tool name — it's how a
// malformed tool-call attempt still produces a ToolUse block. The loop's
// existing "unknown tool name" handling (tools.Registry.Run returns
// ErrToolNotFound, the loop turns that into a ToolResult{IsError:true},
// phase 4) is what actually delivers the correction to the model. No new
// machinery needed here for that half of the job.
const parseErrorToolName = "__prompted_parse_error__"

var fencedJSONRe = regexp.MustCompile("(?s)```(?:json)?\\s*\\n?(.*?)\\n?```")

// Provider wraps inner to add prompted tool-calling.
type Provider struct {
	inner llm.Provider
}

var _ llm.Provider = (*Provider)(nil)

// Wrap returns a Provider that layers prompted tool-calling on top of
// inner.
func Wrap(inner llm.Provider) *Provider {
	return &Provider{inner: inner}
}

// Name delegates to the wrapped provider.
func (p *Provider) Name() string { return p.inner.Name() }

// Chat injects tool schemas into the system prompt (when req has any) and
// calls the wrapped provider with Tools cleared, then parses its reply for
// a fenced tool call before returning.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if len(req.Tools) == 0 {
		return p.inner.Chat(ctx, req)
	}

	promptedReq := req
	promptedReq.System = injectToolPrompt(req.System, req.Tools)
	promptedReq.Tools = nil

	resp, err := p.inner.Chat(ctx, promptedReq)
	if err != nil {
		return nil, err
	}
	return parseToolCall(resp), nil
}

// Stream is not implemented: prompted mode needs the full text before it
// can look for a fenced block, so it needs its own accumulate-then-parse
// design rather than reusing native streaming (no phase currently plans
// this — TASKS.md only scopes 6.x to the native adapters).
func (p *Provider) Stream(_ context.Context, _ llm.Request) (<-chan llm.Event, error) {
	return nil, errors.New("prompted: streaming is not implemented")
}

func injectToolPrompt(system string, tools []llm.ToolDefinition) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(system)
		b.WriteString("\n\n")
	}
	b.WriteString("You can call tools. To call one, respond with a single fenced code block ")
	b.WriteString("labeled json containing exactly this shape:\n")
	b.WriteString("```json\n{\"tool\": \"<tool name>\", \"input\": <object matching the tool's schema>}\n```\n")
	b.WriteString("Call at most one tool per response. If no tool call is needed, respond normally with no fenced block.\n\n")
	b.WriteString("Available tools:\n")
	for _, td := range tools {
		fmt.Fprintf(&b, "- %s: %s\n  input schema: %s\n", td.Name, td.Description, td.InputSchema)
	}
	return b.String()
}

// parseToolCall scans resp's first Text block for a fenced tool call and,
// if found, replaces it with a ToolUse block (surrounding prose becomes its
// own Text block on either side). A response with no fenced block passes
// through unchanged — the model simply didn't call a tool this turn.
func parseToolCall(resp *llm.Response) *llm.Response {
	blocks := make([]llm.Block, 0, len(resp.Message.Blocks)+1)
	found := false

	for _, b := range resp.Message.Blocks {
		text, ok := b.(llm.Text)
		if !ok || found {
			blocks = append(blocks, b)
			continue
		}

		prefix, suffix, call, ok := extractToolCall(text.Text)
		if !ok {
			blocks = append(blocks, b)
			continue
		}
		found = true
		if prefix != "" {
			blocks = append(blocks, llm.Text{Text: prefix})
		}
		blocks = append(blocks, call)
		if suffix != "" {
			blocks = append(blocks, llm.Text{Text: suffix})
		}
	}

	stopReason := resp.StopReason
	if found {
		stopReason = "tool_use"
	}
	return &llm.Response{
		Message:    llm.Message{Role: resp.Message.Role, Blocks: blocks},
		StopReason: stopReason,
		Usage:      resp.Usage,
	}
}

// extractToolCall finds the first fenced ```json block in text. ok is false
// only when no fence is present at all; a present-but-malformed block still
// returns ok=true with a call block that carries the parse error, so the
// caller always has something to hand back to the model.
func extractToolCall(text string) (prefix, suffix string, call llm.Block, ok bool) {
	loc := fencedJSONRe.FindStringSubmatchIndex(text)
	if loc == nil {
		return "", "", nil, false
	}
	prefix = strings.TrimSpace(text[:loc[0]])
	suffix = strings.TrimSpace(text[loc[1]:])
	inner := text[loc[2]:loc[3]]

	var parsed struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal([]byte(inner), &parsed); err != nil {
		return prefix, suffix, malformedToolCall(inner, err), true
	}
	if parsed.Tool == "" {
		return prefix, suffix, malformedToolCall(inner, errors.New(`missing "tool" field`)), true
	}

	input := parsed.Input
	if input == nil {
		input = json.RawMessage(`{}`)
	}
	return prefix, suffix, llm.ToolUse{ID: syntheticID(), Name: parsed.Tool, Input: input}, true
}

func malformedToolCall(raw string, parseErr error) llm.Block {
	input, err := json.Marshal(map[string]string{"error": parseErr.Error(), "raw": raw})
	if err != nil {
		// map[string]string of two plain strings always marshals; this is
		// unreachable, but a hardcoded fallback beats a panic if it ever isn't.
		input = json.RawMessage(`{"error":"malformed tool call"}`)
	}
	return llm.ToolUse{ID: syntheticID(), Name: parseErrorToolName, Input: input}
}

// syntheticID gives a synthesized ToolUse a unique-enough ID. Global
// uniqueness isn't required — history.Validate only pairs a ToolUse against
// the ToolResult in the very next message — so a timestamp is sufficient
// without needing a package-level counter (CLAUDE.md: no package-level
// mutable state outside cmd/).
func syntheticID() string {
	return fmt.Sprintf("prompted_%d", time.Now().UnixNano())
}
