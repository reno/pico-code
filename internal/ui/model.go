package ui

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

// userLineStyle marks a submitted message in the transcript and the input
// textbox itself as the user's own text: a gray field with white text, so a
// submitted turn reads as visually distinct from the assistant's reply that
// follows it.
var userLineStyle = lipgloss.NewStyle().Background(lipgloss.Color("235")).Foreground(lipgloss.Color("15"))

// statusLineStyle colors the turn status line (spinner + word + counters)
// shown while a turn is in flight.
var statusLineStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#ADFF2F"))

// spinnerStyle bolds the spinner glyph on top of statusLineStyle's color, so
// the arc's strokes read heavier than the status text next to it.
var spinnerStyle = statusLineStyle.Bold(true)

// statusParenStyle mutes the parenthesized elapsed-time/tokens/interrupt-key
// part of the status line, so only the turn word carries the accent color.
var statusParenStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))

// rotatingSpinner is a growing/shrinking arc chasing itself around a
// circle — segments of a ring in visible motion rather than a static glyph
// swap.
var rotatingSpinner = spinner.Spinner{
	Frames: []string{"◜", "◠", "◝", "◞", "◡", "◟"},
	FPS:    time.Second / 8, //nolint:mnd
}

// dot styles color the status marker printed before every message and
// action in the transcript: white for a message (user or assistant), gray
// while an action is still running, green once it succeeds, red once it
// fails.
var (
	dotMessageStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
	dotPendingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	dotSuccessStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
	dotErrorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
)

// sessionState is the TUI's coarse state machine: idle waits for input,
// streaming/toolRunning reflect where the current turn is, and approval
// suspends everything else until a modal is answered.
type sessionState int

const (
	stateIdle sessionState = iota
	stateStreaming
	stateToolRunning
	stateApproval
)

// toolBlock is one tool call's rendered status within the current turn.
type toolBlock struct {
	id, name string
	status   string // "running", "ok", "error"
	output   string
}

type approvalRequest struct {
	toolName string
	preview  string
	resp     chan<- bool
}

// Model is the bubbletea model driving the interactive TUI. It never talks
// to a Provider or the agent loop directly — TUIRenderer and TUIApprover
// (this package) push it messages over the tea.Program those own, keeping
// UI rendering separate from agent policy per CLAUDE.md invariant 5.
type Model struct {
	submit chan<- string

	state      sessionState
	priorState sessionState

	viewport viewport.Model
	textarea textarea.Model
	spinner  spinner.Model
	renderer *glamour.TermRenderer

	bannerInfo   BannerInfo
	banner       string // rendered home screen, redrawn on resize
	glamourStyle string // "dark" or "light", resolved once before the program starts

	completed string // finalized, glamour-rendered transcript
	liveText  string // in-progress text, appended per textDeltaMsg
	liveTools []toolBlock
	cancel    context.CancelFunc

	turnElapsed int // seconds ticked since the current turn started
	turnTokens  int // estimated tokens streamed so far this turn

	wordRand    *rand.Rand // overridden directly by tests for determinism
	currentWord string     // this turn's word, rotated by turnTickMsg
	wordTick    int        // ticks since the last rotation

	approvalQueue []approvalRequest
	approval      *approvalRequest

	err   error
	ready bool
	width int
}

// NewModel returns a Model that sends submitted user input on submit.
// info is the home screen's content; the banner itself is rendered on the
// first resize, since its frame has to be drawn to the terminal's width.
// glamourStyle is "dark" or "light", resolved by the caller *before*
// bubbletea takes over the terminal: querying the background color from
// inside the running program (glamour.WithAutoStyle's default behavior)
// races with bubbletea's own raw-mode input reader and leaks stray escape
// bytes onto the screen. An empty glamourStyle defaults to "dark".
func NewModel(submit chan<- string, info BannerInfo, glamourStyle string) Model {
	ta := textarea.New()
	ta.Placeholder = "Ask pico code…"
	ta.SetPromptFunc(2, func(lineIdx int) string {
		if lineIdx == 0 {
			return "> "
		}
		return ""
	})
	ta.Focus()
	ta.ShowLineNumbers = false
	ta.SetHeight(3)
	ta.FocusedStyle.Prompt = userLineStyle
	ta.FocusedStyle.Text = userLineStyle
	ta.FocusedStyle.CursorLine = userLineStyle
	ta.FocusedStyle.EndOfBuffer = userLineStyle
	ta.FocusedStyle.Placeholder = userLineStyle
	ta.BlurredStyle.Prompt = userLineStyle
	ta.BlurredStyle.Text = userLineStyle
	ta.BlurredStyle.CursorLine = userLineStyle
	ta.BlurredStyle.EndOfBuffer = userLineStyle
	ta.BlurredStyle.Placeholder = userLineStyle

	sp := spinner.New()
	sp.Spinner = rotatingSpinner
	sp.Style = spinnerStyle

	if glamourStyle == "" {
		glamourStyle = "dark"
	}

	return Model{
		submit:       submit,
		textarea:     ta,
		spinner:      sp,
		viewport:     viewport.New(80, 20),
		bannerInfo:   info,
		glamourStyle: glamourStyle,
		wordRand:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Init implements tea.Model.
func (m Model) Init() tea.Cmd {
	return textarea.Blink
}

// Update implements tea.Model.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		return m.handleResize(msg)
	case tea.KeyMsg:
		return m.handleKey(msg)
	case spinner.TickMsg:
		if m.state == stateIdle {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	case turnStartedMsg:
		m.state = stateStreaming
		m.cancel = msg.cancel
		m.liveText = ""
		m.liveTools = nil
		m.err = nil
		m.turnElapsed = 0
		m.turnTokens = 0
		m.wordTick = 0
		m.currentWord = nextWord(m.wordRand, m.currentWord)
		m.refreshViewport()
		return m, tea.Batch(m.spinner.Tick, tickEvery())
	case turnTickMsg:
		if m.state == stateIdle {
			return m, nil
		}
		m.turnElapsed++
		m.wordTick++
		if m.wordTick%wordRotationTicks == 0 {
			m.currentWord = nextWord(m.wordRand, m.currentWord)
		}
		return m, tickEvery()
	case textDeltaMsg:
		m.liveText += msg.text
		m.turnTokens = estimateStreamedTokens(m.liveText)
		m.refreshViewport()
		return m, nil
	case toolStartedMsg:
		m.state = stateToolRunning
		m.liveTools = append(m.liveTools, toolBlock{id: msg.id, name: msg.name, status: "running"})
		m.refreshViewport()
		return m, nil
	case toolFinishedMsg:
		for i := range m.liveTools {
			if m.liveTools[i].id == msg.id {
				m.liveTools[i].status = "ok"
				if msg.isError {
					m.liveTools[i].status = "error"
				}
				m.liveTools[i].output = msg.output
			}
		}
		if !m.anyToolRunning() {
			m.state = stateStreaming
		}
		m.refreshViewport()
		return m, nil
	case turnDoneMsg:
		m.finalizeTurn(msg.text, msg.err)
		return m, nil
	case commandOutputMsg:
		m.completed += msg.text
		m.refreshViewport()
		return m, nil
	case clearScrollbackMsg:
		m.completed = ""
		m.refreshViewport()
		return m, nil
	case approvalRequestMsg:
		m.approvalQueue = append(m.approvalQueue, approvalRequest{toolName: msg.toolName, preview: msg.preview, resp: msg.resp})
		if m.approval == nil {
			m.activateNextApproval()
		}
		return m, nil
	}
	return m, nil
}

func (m *Model) anyToolRunning() bool {
	for _, tb := range m.liveTools {
		if tb.status == "running" {
			return true
		}
	}
	return false
}

func (m *Model) activateNextApproval() {
	if len(m.approvalQueue) == 0 {
		m.approval = nil
		m.state = m.priorState
		return
	}
	if m.state != stateApproval {
		m.priorState = m.state
	}
	req := m.approvalQueue[0]
	m.approvalQueue = m.approvalQueue[1:]
	m.approval = &req
	m.state = stateApproval
}

func (m *Model) finalizeTurn(text string, err error) {
	m.state = stateIdle
	m.cancel = nil
	if err != nil {
		m.err = err
	} else if text != "" {
		rendered := text
		if m.renderer != nil {
			if out, rerr := m.renderer.Render(text); rerr == nil {
				rendered = out
			}
		}
		m.completed += dotMessageStyle.Render("●") + " " + strings.TrimLeft(rendered, "\n")
	}
	m.liveText = ""
	m.liveTools = nil
	m.refreshViewport()
	m.textarea.Focus()
}

func (m Model) handleResize(msg tea.WindowSizeMsg) (Model, tea.Cmd) {
	m.width = msg.Width
	m.ready = true
	m.textarea.SetWidth(msg.Width)
	m.banner = Banner(m.bannerInfo, msg.Width)
	m.viewport.Width = msg.Width
	m.viewport.Height = msg.Height - m.textarea.Height() - 2
	if r, err := glamour.NewTermRenderer(glamour.WithStandardStyle(m.glamourStyle), glamour.WithWordWrap(msg.Width)); err == nil {
		m.renderer = r
	}
	m.refreshViewport()
	return m, nil
}

func (m Model) handleKey(msg tea.KeyMsg) (Model, tea.Cmd) {
	if msg.Type == tea.KeyCtrlD {
		if m.cancel != nil {
			m.cancel()
		}
		return m, tea.Quit
	}

	if m.state == stateApproval {
		return m.handleApprovalKey(msg)
	}

	if msg.Type == tea.KeyCtrlC {
		if m.cancel != nil {
			m.cancel()
		}
		return m, nil
	}

	if msg.Type == tea.KeyEsc && m.state != stateIdle {
		if m.cancel != nil {
			m.cancel()
		}
		return m, nil
	}

	if m.state == stateIdle && msg.Type == tea.KeyEnter {
		input := strings.TrimSpace(m.textarea.Value())
		if input == "" {
			return m, nil
		}
		select {
		case m.submit <- input:
			m.completed += userLineStyle.Render(padLines("> "+input, m.width)) + "\n\n"
			m.textarea.Reset()
			m.refreshViewport()
		default:
		}
		return m, nil
	}

	if m.state != stateIdle {
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd
	}

	var cmds []tea.Cmd
	var cmd tea.Cmd
	m.textarea, cmd = m.textarea.Update(msg)
	cmds = append(cmds, cmd)
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)
	return m, tea.Batch(cmds...)
}

func (m Model) handleApprovalKey(msg tea.KeyMsg) (Model, tea.Cmd) {
	if m.approval == nil {
		return m, nil
	}
	switch msg.Type {
	case tea.KeyRunes:
		switch string(msg.Runes) {
		case "y", "Y":
			m.approval.resp <- true
			m.activateNextApproval()
		case "n", "N":
			m.approval.resp <- false
			m.activateNextApproval()
		}
	case tea.KeyEsc:
		m.approval.resp <- false
		m.activateNextApproval()
	}
	return m, nil
}

// padLines right-pads every line of s with spaces out to w display cells, so
// a background color applied to the result fills the full line width rather
// than stopping at the last visible character.
func padLines(s string, w int) string {
	if w <= 0 {
		return s
	}
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		if gap := w - lipgloss.Width(line); gap > 0 {
			lines[i] = line + strings.Repeat(" ", gap)
		}
	}
	return strings.Join(lines, "\n")
}

// refreshViewport rebuilds the viewport's content from the finalized
// transcript plus whatever the in-progress turn has produced so far.
func (m *Model) refreshViewport() {
	var b strings.Builder
	b.WriteString(m.banner)
	b.WriteString(m.completed)
	if len(m.liveText) > 0 || len(m.liveTools) > 0 {
		b.WriteString(m.liveText)
		b.WriteString("\n")
	}
	for _, tb := range m.liveTools {
		dot := dotPendingStyle.Render("●")
		icon := "⏳"
		switch tb.status {
		case "ok":
			dot = dotSuccessStyle.Render("●")
			icon = "✓"
		case "error":
			dot = dotErrorStyle.Render("●")
			icon = "✗"
		}
		fmt.Fprintf(&b, "%s %s %s\n", dot, icon, tb.name)
	}
	if m.err != nil {
		fmt.Fprintf(&b, "\nerror: %v\n", m.err)
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoBottom()
}

// View implements tea.Model.
func (m Model) View() string {
	if !m.ready {
		return "initializing…"
	}
	if m.state == stateApproval && m.approval != nil {
		return m.viewport.View() + "\n" + m.approvalModalView()
	}

	status := ""
	if m.state != stateIdle {
		status = m.spinner.View() + " " + m.styledStatusText()
	}
	return m.viewport.View() + "\n" + status + "\n" + padTextareaView(m.textarea.View(), m.width)
}

// padTextareaView tops up every line of the textarea's rendered view with
// gray-background spaces out to width w. In placeholder mode, the
// component's own inner viewport already pads short lines out to its
// declared width, but with plain, unstyled spaces — the tail of the
// placeholder line reads as a bare gap instead of gray. Any literal
// trailing spaces are stripped and rebuilt with userLineStyle so the whole
// line, not just its text, carries the background.
func padTextareaView(view string, w int) string {
	if w <= 0 {
		return view
	}
	lines := strings.Split(view, "\n")
	for i, line := range lines {
		trimmed := strings.TrimRight(line, " ")
		spaces := len(line) - len(trimmed)
		if gap := w - lipgloss.Width(trimmed); gap > spaces {
			spaces = gap
		}
		if spaces > 0 {
			lines[i] = trimmed + userLineStyle.Render(strings.Repeat(" ", spaces))
		}
	}
	return strings.Join(lines, "\n")
}

// statusText renders the turn status line: a word describing what's
// happening, elapsed time since the turn started, an estimate of tokens
// streamed so far, and the key that interrupts it.
func (m Model) statusText() string {
	return fmt.Sprintf("%s… (%ds · ↑ %s tokens · esc to interrupt)", m.turnWord(), m.turnElapsed, formatTokenCount(m.turnTokens))
}

// styledStatusText renders statusText's content with the parenthesized
// elapsed-time/tokens/interrupt-key part muted to gray, so only the turn
// word carries the status line's accent color.
func (m Model) styledStatusText() string {
	word := statusLineStyle.Render(m.turnWord() + "…")
	rest := fmt.Sprintf(" (%ds · ↑ %s tokens · esc to interrupt)", m.turnElapsed, formatTokenCount(m.turnTokens))
	return word + statusParenStyle.Render(rest)
}

func (m Model) turnWord() string {
	switch m.state {
	case stateStreaming, stateToolRunning:
		return m.currentWord
	default:
		return ""
	}
}

// wordRotationTicks is how many once-per-second turnTickMsgs pass between
// word rotations — a few seconds, per 10.2, long enough to read.
const wordRotationTicks = 6

// turnWords is the vocabulary the status line's word is drawn from while a
// turn is in flight — deliberately whimsical, since it stands in for real
// progress we don't have a finer-grained signal for.
var turnWords = []string{
	"computing",
	"compiling",
	"parsing",
	"indexing",
	"vectorizing",
	"reticulating",
	"percolating",
	"marinating",
	"bikeshedding",
	"yak-shaving",
	"spelunking",
	"noodling",
	"cogitating",
	"ruminating",
	"pondering",
	"frobnicating",
}

// nextWord draws a word from turnWords using r, excluding prev so the
// status line never shows the same word twice in a row.
func nextWord(r *rand.Rand, prev string) string {
	if len(turnWords) == 1 {
		return turnWords[0]
	}
	for {
		w := turnWords[r.Intn(len(turnWords))]
		if w != prev {
			return w
		}
	}
}

// tickEvery schedules the next turnTickMsg one second out. Update
// re-issues it after every tick for as long as a turn is in flight, and
// lets it lapse once the turn returns to stateIdle.
func tickEvery() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return turnTickMsg{} })
}

// estimateStreamedTokens approximates a token count from streamed text
// length for the status line's live counter — the same ~4-chars-per-token
// rule of thumb used as a fallback elsewhere, not a billing figure (that
// comes from the provider's own Usage once MessageDone arrives).
func estimateStreamedTokens(s string) int {
	if s == "" {
		return 0
	}
	if n := len(s) / 4; n > 0 {
		return n
	}
	return 1
}

// formatTokenCount renders a token count the way the status line quotes
// one in flight ("1.2k"), with one decimal place so the counter visibly
// moves between whole-thousand boundaries instead of jumping.
func formatTokenCount(n int) string {
	if n < 1000 {
		return strconv.Itoa(n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}

func (m Model) approvalModalView() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Approve call to %q? [y/N]\n", m.approval.toolName)
	if m.approval.preview != "" {
		b.WriteString(m.approval.preview)
		b.WriteString("\n")
	}
	return b.String()
}
