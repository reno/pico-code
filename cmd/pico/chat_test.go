package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/llm"
	"github.com/reno/pico-code/internal/tools"
)

// fakeChatProvider returns one canned reply, enough to drive
// runPlainChat's single-shot piped-input path without a real backend.
type fakeChatProvider struct{ reply string }

func (f *fakeChatProvider) Name() string { return "fake" }

func (f *fakeChatProvider) Chat(context.Context, llm.Request) (*llm.Response, error) {
	return nil, nil
}

func (f *fakeChatProvider) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Event, error) {
	return llm.StreamEvents(ctx, func(send func(llm.Event) bool) {
		if !send(llm.TextDelta{Text: f.reply}) {
			return
		}
		send(llm.MessageDone{StopReason: "end_turn"})
	}), nil
}

// TestPipedChatProducesCleanANSIFreeOutput is 7.1's AC: piping input to a
// non-interactive chat session produces clean, ANSI-free output.
func TestPipedChatProducesCleanANSIFreeOutput(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetIn(strings.NewReader("hi\n"))
	var out bytes.Buffer
	cmd.SetOut(&out)

	cfg := &config.Config{Yes: true}
	provider := &fakeChatProvider{reply: "hello there"}
	h := history.New()

	err := runPlainChat(cmd, cfg, provider, tools.NewRegistry(), h, agent.Guards{})
	if err != nil {
		t.Fatalf("runPlainChat() error = %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "hello there") {
		t.Errorf("output = %q, want it to contain the reply", got)
	}
	if strings.ContainsRune(got, '\x1b') {
		t.Errorf("output contains an ANSI escape byte: %q", got)
	}
	if err := h.Validate(); err != nil {
		t.Errorf("history.Validate() error = %v", err)
	}
}
