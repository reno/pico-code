package ui

import (
	"math/rand/v2"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// skull is the home screen's mark. Rows are padded to skullWidth by the
// caller rather than by trailing spaces in this literal, which editors and
// linters strip.
const skull = ` ▄▄▄▄▄
███████
█  █  █
███ ███
 █ █ █`

const skullWidth = 7

// Banner width bounds. Below minBannerWidth the two columns have no room
// for content; above maxBannerWidth the box stretches into unreadable
// line lengths on a wide terminal. The greeting fits whole from 80
// columns up (greetingCol + divider + minRightCol); narrower than that it
// is ellipsized rather than allowed to push the frame out of shape.
const (
	minBannerWidth = 64
	maxBannerWidth = 128
)

// mockeryMaxLen bounds an entry in mockeries (declared in mockeries.go).
// The greeting is one full-width line, "Welcome back, <mockery>!", so the
// budget is the frame's inner width at the narrowest terminal worth
// designing for (80 columns) less that wrapping.
const mockeryMaxLen = 30

// greetingCol is the left column width needed to print the longest
// greeting whole; minRightCol is what the right column needs for its
// widest line, the help hint. Their sum plus the divider is what
// minBannerWidth has to accommodate.
const (
	greetingCol = mockeryMaxLen + len("Welcome back, !")
	minRightCol = len("  type /help to see all commands")
)

// Mockery returns one random affectionate insult. Call it once per process
// and store the result in BannerInfo — Banner itself must stay pure so a
// resize re-render produces the identical greeting.
func Mockery() string {
	return mockeries[rand.IntN(len(mockeries))]
}

// BannerInfo is everything the home screen displays. Empty fields degrade
// to a placeholder row rather than collapsing the layout, so the box keeps
// its shape on a first run with no history.
type BannerInfo struct {
	Version   string
	Greeting  string
	User      string
	Provider  string
	Model     string
	Directory string
	Sessions  []string
	Usage     string
}

// cell is one line of a column: plain text whose display width the layout
// controls, plus the style to paint it with. Keeping text plain until the
// final render is what lets the padding math stay simple — styling adds
// escape sequences but no display width.
type cell struct {
	text  string
	style lipgloss.Style
}

// Banner renders the home screen: a titled frame split into a left column
// (greeting, skull, session identity) and a right column (help hint,
// recent sessions, usage). width is the terminal width; it is clamped, so
// callers may pass a raw, even zero, value.
func Banner(info BannerInfo, width int) string {
	width = clamp(width, minBannerWidth, maxBannerWidth)
	inner := width - 2

	// The left column is sized to hold the greeting outright rather than
	// to a fixed fraction: a truncated mockery is the one thing on this
	// screen that has to read whole. The right column keeps a floor so
	// widening the left cannot squeeze the help line out.
	leftW := maxInt(inner*2/5, greetingCol)
	if limit := inner - 1 - minRightCol; leftW > limit {
		leftW = maxInt(limit, 1)
	}
	rightW := inner - 1 - leftW

	left := leftColumn(info, leftW)
	right := rightColumn(info, rightW)

	// Usage is anchored to the frame's last row rather than following the
	// session list, so it stays put as that list grows and shrinks.
	height := maxInt(len(left), len(right))
	if info.Usage != "" && height <= len(right) {
		height = len(right) + 1
	}
	for len(left) < height {
		left = append(left, cell{})
	}
	for len(right) < height {
		right = append(right, cell{})
	}
	if info.Usage != "" {
		right[height-1] = cell{
			text:  "  Usage: " + info.Usage,
			style: lipgloss.NewStyle().Foreground(lipgloss.Color("240")),
		}
	}

	frame := lipgloss.NewStyle().Foreground(lipgloss.Color("#ADFF2F"))
	bar := frame.Render("│")

	var b strings.Builder
	b.WriteString(topBorder(info.Version, inner, frame))
	b.WriteString("\n")

	for i := range left {
		b.WriteString(bar)
		b.WriteString(left[i].style.Render(pad(left[i].text, leftW)))
		b.WriteString(bar)
		b.WriteString(right[i].style.Render(pad(right[i].text, rightW)))
		b.WriteString(bar)
		b.WriteString("\n")
	}
	b.WriteString(frame.Render("╰" + strings.Repeat("─", inner) + "╯"))
	b.WriteString("\n\n")
	return b.String()
}

// greeting is the one-line welcome. An empty mockery still greets you,
// just less specifically.
func greeting(info BannerInfo) string {
	if info.Greeting == "" {
		return "Welcome back!"
	}
	return "Welcome back, " + info.Greeting + "!"
}

// topBorder draws the frame's top edge with the title inlaid, mirroring
// the "╭─── name version ───╮" shape.
func topBorder(version string, inner int, frame lipgloss.Style) string {
	title := "pico"
	if version != "" {
		title += " " + version
	}
	lead := 3
	fill := inner - lead - lipgloss.Width(title) - 2
	if fill < 0 {
		return frame.Render("╭" + strings.Repeat("─", inner) + "╮")
	}
	name := lipgloss.NewStyle().Foreground(lipgloss.Color("#ADFF2F")).Bold(true).Render("pico")
	if version != "" {
		name += lipgloss.NewStyle().Foreground(lipgloss.Color("245")).Render(" " + version)
	}
	return frame.Render("╭"+strings.Repeat("─", lead)+" ") +
		name +
		frame.Render(" "+strings.Repeat("─", fill)+"╮")
}

func leftColumn(info BannerInfo, w int) []cell {
	white := lipgloss.NewStyle().Foreground(lipgloss.Color("#FFFFFF")).Bold(true)
	dim := lipgloss.NewStyle().Foreground(lipgloss.Color("245"))

	out := []cell{
		{},
		{text: center(greeting(info), w), style: lipgloss.NewStyle().Bold(true)},
	}

	for _, row := range strings.Split(skull, "\n") {
		out = append(out, cell{text: center(row+strings.Repeat(" ", skullWidth-lipgloss.Width(row)), w), style: white})
	}

	out = append(out, cell{})
	if id := join(" · ", info.Provider, info.Model, info.User); id != "" {
		out = append(out, cell{text: "  " + id, style: dim})
	}
	if info.Directory != "" {
		out = append(out, cell{text: "  " + truncPath(info.Directory, w-2), style: dim})
	}
	return out
}

func rightColumn(info BannerInfo, w int) []cell {
	dim := lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	faint := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	green := lipgloss.NewStyle().Foreground(lipgloss.Color("#ADFF2F"))

	out := []cell{
		{},
		{text: "  Tips for getting started", style: green},
		{text: "  type /help to see all commands", style: dim},
		{text: "  " + strings.Repeat("─", maxInt(w-4, 1)), style: green},
		{text: "  Last sessions", style: green},
	}

	if len(info.Sessions) == 0 {
		out = append(out, cell{text: "  no saved sessions yet", style: faint})
	}
	for _, s := range info.Sessions {
		out = append(out, cell{text: "  " + s, style: dim})
	}
	return out
}

// pad truncates text to w display cells (with an ellipsis when it had to
// cut) and right-pads it, so every assembled row is exactly w wide.
func pad(text string, w int) string {
	if w <= 0 {
		return ""
	}
	if lipgloss.Width(text) > w {
		r := []rune(text)
		for len(r) > 0 && lipgloss.Width(string(r)+"…") > w {
			r = r[:len(r)-1]
		}
		text = string(r) + "…"
	}
	return text + strings.Repeat(" ", w-lipgloss.Width(text))
}

// truncPath shortens a path from the left, keeping the tail — the part
// that says which project you are in. pad would cut the other end and
// leave every deep path looking identical.
func truncPath(path string, w int) string {
	if w <= 1 || lipgloss.Width(path) <= w {
		return path
	}
	r := []rune(path)
	for len(r) > 0 && lipgloss.Width("…"+string(r)) > w {
		r = r[1:]
	}
	return "…" + string(r)
}

func center(text string, w int) string {
	gap := w - lipgloss.Width(text)
	if gap <= 0 {
		return text
	}
	return strings.Repeat(" ", gap/2) + text
}

func join(sep string, parts ...string) string {
	var kept []string
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, sep)
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
