package tools

import "fmt"

const defaultByteBudget = 32 * 1024

// truncateBytes returns s unchanged if it's within budget; otherwise a
// head+tail slice with an elision marker in between, kept to budget so a
// large tool output can never blow the context window.
func truncateBytes(s string, budget int) string {
	if len(s) <= budget {
		return s
	}
	marker := fmt.Sprintf("\n\n... [%d bytes elided] ...\n\n", len(s)-budget)
	keep := budget - len(marker)
	if keep < 0 {
		keep = 0
	}
	head := keep / 2
	tail := keep - head
	return s[:head] + marker + s[len(s)-tail:]
}
