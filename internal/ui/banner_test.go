package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func sampleInfo() BannerInfo {
	return BannerInfo{
		Version:   "v0.1.0",
		Greeting:  "meatbag",
		User:      "renan",
		Provider:  "ollama",
		Model:     "qwen3:8b",
		Directory: "/tmp/ws",
		Sessions:  []string{"refactor-loop", "add-tools"},
		Usage:     "0 / 128k tokens",
	}
}

func TestSkullFitsItsDeclaredWidth(t *testing.T) {
	for i, line := range strings.Split(skull, "\n") {
		if got := lipgloss.Width(line); got > skullWidth {
			t.Errorf("skull row %d width = %d, want <= %d", i, got, skullWidth)
		}
	}
}

// Every rendered row must be exactly the banner's width, or the frame's
// right edge tears — the failure mode any change to the column math hits.
func TestBannerRowsAreFlush(t *testing.T) {
	for _, width := range []int{0, 40, 64, 80, 120, 200} {
		lines := strings.Split(strings.TrimRight(Banner(sampleInfo(), width), "\n"), "\n")
		want := lipgloss.Width(lines[0])
		if want < minBannerWidth || want > maxBannerWidth {
			t.Errorf("width %d: banner rendered %d cells wide, outside [%d,%d]", width, want, minBannerWidth, maxBannerWidth)
		}
		for i, line := range lines {
			if got := lipgloss.Width(line); got != want {
				t.Errorf("width %d: row %d is %d cells, want %d", width, i, got, want)
			}
		}
	}
}

func TestBannerShowsSessionDetails(t *testing.T) {
	out := Banner(sampleInfo(), 120)
	for _, want := range []string{
		"pico code v0.1.0", "Welcome back, meatbag!", "ollama", "qwen3:8b",
		"renan", "/tmp/ws", "type /help", "refactor-loop", "add-tools", "128k", "█",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Banner() missing %q", want)
		}
	}
}

func TestBannerHandlesNoSessions(t *testing.T) {
	info := sampleInfo()
	info.Sessions = nil
	if !strings.Contains(Banner(info, 120), "no saved sessions yet") {
		t.Error("Banner() should say so when there are no saved sessions")
	}
}

// A long path must be truncated rather than pushing the frame's edge out,
// which is what TestBannerRowsAreFlush would otherwise catch only by luck.
func TestBannerTruncatesOverlongText(t *testing.T) {
	info := sampleInfo()
	info.Directory = "/" + strings.Repeat("very-long-directory-name/", 20)
	out := Banner(info, 80)
	if !strings.Contains(out, "…") {
		t.Error("Banner() should ellipsize text too wide for its column")
	}
}

// The list feeds a fixed-width column, so every entry has to fit it and
// be distinct. The list is hand-curated and its length is deliberate —
// this asserts it is usable, not how long it is.
func TestMockeriesAreUsable(t *testing.T) {
	if len(mockeries) == 0 {
		t.Fatal("mockeries is empty; the home screen has nothing to greet you with")
	}
	seen := make(map[string]bool, len(mockeries))
	for _, m := range mockeries {
		if n := lipgloss.Width(m); n == 0 || n > mockeryMaxLen {
			t.Errorf("mockery %q is %d cells, want 1..%d", m, n, mockeryMaxLen)
		}
		if seen[m] {
			t.Errorf("mockery %q appears twice", m)
		}
		seen[m] = true
	}
}

// Every mockery must survive the greeting uncut at 80 columns, the
// narrowest width mockeryMaxLen is calibrated for. Checking only a wide
// terminal here would hide exactly the entries that are too long.
func TestEveryMockeryFitsTheGreeting(t *testing.T) {
	info := sampleInfo()
	info.Directory = "/ws"
	info.Sessions = []string{"one"}
	for _, width := range []int{80, 100, 120} {
		for _, m := range mockeries {
			info.Greeting = m
			if strings.Contains(Banner(info, width), "…") {
				t.Errorf("greeting for %q was ellipsized at width %d", m, width)
			}
		}
	}
}

// A deep path must keep its tail — the project name — not its head,
// which is the same "/Users/<name>/..." prefix for everything.
func TestBannerKeepsTheTailOfALongPath(t *testing.T) {
	info := sampleInfo()
	info.Directory = "~/Documents/Github/some/deeply/nested/pico-code"
	out := Banner(info, 80)
	if !strings.Contains(out, "pico-code") {
		t.Errorf("Banner() dropped the end of a long path:\n%s", out)
	}
	if strings.Contains(out, "~/Documents/Github/some/deeply") {
		t.Error("Banner() kept the head of a long path, want the tail")
	}
}

func TestMockeryReturnsAKnownName(t *testing.T) {
	got := Mockery()
	for _, m := range mockeries {
		if got == m {
			return
		}
	}
	t.Errorf("Mockery() = %q, not one of the known mockeries", got)
}

func TestModelRendersBannerAfterResize(t *testing.T) {
	m := NewModel(make(chan string, 1), sampleInfo())
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	if !strings.Contains(updated.View(), "Welcome back, meatbag!") {
		t.Error("View() should show the banner once the width is known")
	}
}
