package tools

import "fmt"

const defaultByteBudget = 32 * 1024

// DefaultByteBudget is the byte budget this package's own tools truncate
// output to, exposed so another internal package building something that
// behaves like a Tool (the sub-agent tool in internal/agent) can match it
// instead of picking its own number.
const DefaultByteBudget = defaultByteBudget

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

// TruncateBytes exposes truncateBytes to another internal package building
// something that behaves like a Tool but can't live in this package (the
// sub-agent tool in internal/agent — internal/agent already imports
// internal/tools, so the reverse import would cycle).
func TruncateBytes(s string, budget int) string { return truncateBytes(s, budget) }
