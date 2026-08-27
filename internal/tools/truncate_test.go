package tools

import (
	"strings"
	"testing"
)

func TestTruncateBytesUnderBudgetUnchanged(t *testing.T) {
	s := "hello world"
	if got := truncateBytes(s, 1024); got != s {
		t.Errorf("truncateBytes() = %q, want unchanged %q", got, s)
	}
}

func TestTruncateBytesOverBudget(t *testing.T) {
	s := strings.Repeat("A", 10*1024*1024)
	budget := 32 * 1024

	got := truncateBytes(s, budget)
	if len(got) > budget {
		t.Fatalf("truncateBytes() length = %d, want <= %d", len(got), budget)
	}
	if !strings.Contains(got, "elided") {
		t.Errorf("truncateBytes() = %q, want it to contain an elision marker", got[:200])
	}
	if !strings.HasPrefix(got, "AAAA") {
		t.Error("truncateBytes() should keep a head slice of the original content")
	}
	if !strings.HasSuffix(got, "AAAA") {
		t.Error("truncateBytes() should keep a tail slice of the original content")
	}
}
