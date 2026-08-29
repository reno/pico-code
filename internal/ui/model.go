package ui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
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

	bannerInfo BannerInfo
	banner     string // rendered home screen, redrawn on resize

	completed string // finalized, glamour-rendered transcript
	liveText  string // in-progress text, appended per textDeltaMsg
	liveTools []toolBlock
	cancel    context.CancelFunc

	approvalQueue []approvalRequest
	approval      *approvalRequest

	err   error
	ready bool
	width int
}

// NewModel returns a Model that sends submitted user input on submit.
// info is the home screen's content; the banner itself is rendered on the
// first resize, since its frame has to be drawn to the terminal's width.
func NewModel(submit chan<- string, info BannerInfo) Model {
	ta := textarea.New()
	ta.Placeholder = "Ask pico code…"
	ta.Focus()
	ta.ShowLineNumbers = false
	ta.SetHeight(3)

	sp := spinner.New()
	sp.Spinner = spinner.Dot

	return Model{
		submit:     submit,
		textarea:   ta,
		spinner:    sp,
		viewport:   viewport.New(80, 20),
		bannerInfo: info,
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
		m.refreshViewport()
		return m, m.spinner.Tick
	case textDeltaMsg:
		m.liveText += msg.text
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
		m.completed += rendered
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
	if r, err := glamour.NewTermRenderer(glamour.WithAutoStyle(), glamour.WithWordWrap(msg.Width)); err == nil {
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

	if m.state == stateIdle && msg.Type == tea.KeyEnter {
		input := strings.TrimSpace(m.textarea.Value())
		if input == "" {
			return m, nil
		}
		select {
		case m.submit <- input:
			m.textarea.Reset()
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
		icon := "⏳"
		switch tb.status {
		case "ok":
			icon = "✓"
		case "error":
			icon = "✗"
		}
		fmt.Fprintf(&b, "%s %s\n", icon, tb.name)
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

	status := "> "
	if m.state != stateIdle {
		status = m.spinner.View() + " " + m.statusText()
	}
	return m.viewport.View() + "\n" + status + "\n" + m.textarea.View()
}

func (m Model) statusText() string {
	switch m.state {
	case stateStreaming:
		return "thinking…"
	case stateToolRunning:
		return "running tools…"
	default:
		return ""
	}
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
