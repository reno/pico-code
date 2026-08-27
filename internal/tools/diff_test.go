package tools

import (
	"strings"
	"testing"
)

func TestLineDiffNoChanges(t *testing.T) {
	if got := lineDiff("same", "same"); got != "(no changes)" {
		t.Errorf("lineDiff() = %q, want %q", got, "(no changes)")
	}
}

func TestLineDiffShowsAddedRemovedAndContext(t *testing.T) {
	got := lineDiff("a\nb\nc", "a\nX\nc")
	for _, want := range []string{"  a", "- b", "+ X", "  c"} {
		if !strings.Contains(got, want) {
			t.Errorf("lineDiff() = %q, want it to contain %q", got, want)
		}
	}
}

func TestLineDiffFallsBackForLargeInput(t *testing.T) {
	big := strings.Repeat("line\n", maxDiffLines+1)
	got := lineDiff("", big)
	if !strings.Contains(got, "old (1 lines)") {
		t.Errorf("lineDiff() = %q, want the fallback header", got[:200])
	}
}
