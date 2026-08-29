package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/tools"
)

const helpGoldenPath = "testdata/golden/help.txt"

// TestHelpCommandMatchesGolden is 11.1's AC: /help renders commandTable
// aligned. Run with RECORD=1 to (re)write the golden file.
func TestHelpCommandMatchesGolden(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(&out, ag, h, noSession(t), "help", ""); err != nil {
		t.Fatalf("handleCommand(help) error = %v", err)
	}
	got := out.String()

	if os.Getenv("RECORD") == "1" {
		if err := os.MkdirAll(filepath.Dir(helpGoldenPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(helpGoldenPath, []byte(got), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	want, err := os.ReadFile(helpGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to create it)", helpGoldenPath, err)
	}
	if diff := cmp.Diff(string(want), got); diff != "" {
		t.Errorf("/help output mismatch (-want +got):\n%s", diff)
	}
}

// TestCommandTableEntriesHaveSummaries guards against a bare entry slipping
// into commandTable with no help text, since /help's usefulness depends on
// every command documenting itself.
func TestCommandTableEntriesHaveSummaries(t *testing.T) {
	for _, c := range commandTable {
		if strings.TrimSpace(c.summary) == "" {
			t.Errorf("command %q has no summary", c.name)
		}
		if c.run == nil {
			t.Errorf("command %q has no run function", c.name)
		}
	}
}

// TestUnknownCommandSuggestsClosestMatch is 11.1's AC: an unknown command
// reports the closest match instead of a bare error.
func TestUnknownCommandSuggestsClosestMatch(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer
	if _, err := handleCommand(&out, ag, h, noSession(t), "hlep", ""); err != nil {
		t.Fatalf("handleCommand(hlep) error = %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "did you mean /help") {
		t.Errorf("output = %q, want it to suggest /help", got)
	}
}

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"help", "help", 0},
		{"hlep", "help", 2},
		{"sav", "save", 1},
		{"kitten", "sitting", 3},
	}
	for _, tt := range tests {
		if got := levenshtein(tt.a, tt.b); got != tt.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}
