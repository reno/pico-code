package tools

import (
	"fmt"
	"strings"
)

const maxDiffLines = 2000

// lineDiff renders a minimal line-based diff of oldText to newText for an
// approval preview. It falls back to a plain before/after dump for inputs
// too large for the O(n*m) LCS below to be worth computing.
func lineDiff(oldText, newText string) string {
	if oldText == newText {
		return "(no changes)"
	}

	oldLines := strings.Split(oldText, "\n")
	newLines := strings.Split(newText, "\n")
	if len(oldLines) > maxDiffLines || len(newLines) > maxDiffLines {
		return truncateBytes(fmt.Sprintf("--- old (%d lines) ---\n%s\n+++ new (%d lines) +++\n%s",
			len(oldLines), oldText, len(newLines), newText), defaultByteBudget)
	}

	lcs := longestCommonSubsequence(oldLines, newLines)

	var b strings.Builder
	oi, ni, li := 0, 0, 0
	for oi < len(oldLines) || ni < len(newLines) {
		switch {
		case li < len(lcs) && oi < len(oldLines) && oldLines[oi] == lcs[li] && ni < len(newLines) && newLines[ni] == lcs[li]:
			fmt.Fprintf(&b, "  %s\n", oldLines[oi])
			oi++
			ni++
			li++
		case oi < len(oldLines) && (li >= len(lcs) || oldLines[oi] != lcs[li]):
			fmt.Fprintf(&b, "- %s\n", oldLines[oi])
			oi++
		default:
			fmt.Fprintf(&b, "+ %s\n", newLines[ni])
			ni++
		}
	}
	return truncateBytes(b.String(), defaultByteBudget)
}

func longestCommonSubsequence(a, b []string) []string {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			switch {
			case a[i] == b[j]:
				dp[i][j] = dp[i+1][j+1] + 1
			case dp[i+1][j] >= dp[i][j+1]:
				dp[i][j] = dp[i+1][j]
			default:
				dp[i][j] = dp[i][j+1]
			}
		}
	}

	var lcs []string
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			lcs = append(lcs, a[i])
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			i++
		default:
			j++
		}
	}
	return lcs
}
