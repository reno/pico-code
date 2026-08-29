package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/reno/pico-code/internal/agent"
	"github.com/reno/pico-code/internal/history"
	"github.com/reno/pico-code/internal/mcp"
	"github.com/reno/pico-code/internal/tools"
)

// helperServerConfig builds an mcp.ServerConfig whose command is this test
// binary itself, re-exec'd with -test.run=TestMCPHelperProcess so it runs
// as a fake, always-answers MCP server (see mcp_helper_process_test.go).
func helperServerConfig(t *testing.T, name string) mcp.ServerConfig {
	t.Helper()
	return mcp.ServerConfig{
		Name:    name,
		Command: os.Args[0],
		Args:    []string{"-test.run=TestMCPHelperProcess"},
		Env:     []string{"MCP_HELPER_PROCESS=1"},
	}
}

// brokenServerConfig names a command that can't possibly exist, so
// mcp.Start fails immediately (no process is ever spawned) — a fast,
// deterministic "failed" server for tests, independent of any timeout.
func brokenServerConfig(name string) mcp.ServerConfig {
	return mcp.ServerConfig{Name: name, Command: "/definitely/does/not/exist/pico-code-test-binary"}
}

func TestMCPManagerDiscoversAndRegistersNamespacedTools(t *testing.T) {
	reg := tools.NewRegistry()
	mgr := newMCPManager(context.Background(), reg, []mcp.ServerConfig{helperServerConfig(t, "weather")}, 2*time.Second)
	t.Cleanup(mgr.shutdown)

	statuses := mgr.statuses()
	if len(statuses) != 1 {
		t.Fatalf("len(statuses) = %d, want 1", len(statuses))
	}
	if !statuses[0].connected {
		t.Fatalf("statuses[0].connected = false, want true (err = %v)", statuses[0].err)
	}
	if statuses[0].toolCount != 2 {
		t.Errorf("statuses[0].toolCount = %d, want 2", statuses[0].toolCount)
	}

	for _, name := range []string{"mcp__weather__get_weather", "mcp__weather__get_forecast"} {
		if _, err := reg.Get(name); err != nil {
			t.Errorf("reg.Get(%q) error = %v, want it registered", name, err)
		}
	}
}

func TestMCPManagerFailedServerReportsErrorAndRegistersNoTools(t *testing.T) {
	reg := tools.NewRegistry()
	mgr := newMCPManager(context.Background(), reg, []mcp.ServerConfig{brokenServerConfig("broken")}, time.Second)
	t.Cleanup(mgr.shutdown)

	statuses := mgr.statuses()
	if len(statuses) != 1 {
		t.Fatalf("len(statuses) = %d, want 1", len(statuses))
	}
	if statuses[0].connected {
		t.Error("statuses[0].connected = true, want false for a command that doesn't exist")
	}
	if statuses[0].err == nil {
		t.Error("statuses[0].err = nil, want the discovery failure")
	}
	if len(reg.Definitions()) != 0 {
		t.Errorf("len(reg.Definitions()) = %d, want 0 (a failed server contributes no tools)", len(reg.Definitions()))
	}
}

// TestMCPManagerOneBrokenServerDoesNotBlockAGoodOne is 13.1's guarantee
// (Discover) still holding once wrapped by the manager: a hung/broken
// server never delays another server's own result.
func TestMCPManagerOneBrokenServerDoesNotBlockAGoodOne(t *testing.T) {
	reg := tools.NewRegistry()
	cfgs := []mcp.ServerConfig{helperServerConfig(t, "good"), brokenServerConfig("bad")}

	start := time.Now()
	mgr := newMCPManager(context.Background(), reg, cfgs, 2*time.Second)
	elapsed := time.Since(start)
	t.Cleanup(mgr.shutdown)

	if elapsed > 2*time.Second {
		t.Fatalf("newMCPManager() took %s, want it bounded by the discovery timeout", elapsed)
	}
	statuses := mgr.statuses()
	if !statuses[0].connected || statuses[1].connected {
		t.Errorf("statuses = %+v, want [connected, failed]", statuses)
	}
}

// TestMCPManagerReconnectReplacesToolsWithoutDuplicateError is 13.3's
// reconnect AC: re-running discovery for an already-connected server must
// not trip Registry's duplicate-name detection (13.2) on its own
// previously registered tools.
func TestMCPManagerReconnectReplacesToolsWithoutDuplicateError(t *testing.T) {
	reg := tools.NewRegistry()
	mgr := newMCPManager(context.Background(), reg, []mcp.ServerConfig{helperServerConfig(t, "weather")}, 2*time.Second)
	t.Cleanup(mgr.shutdown)

	oldPid := mgr.servers[0].client.Pid()

	if err := mgr.reconnect(context.Background(), "weather"); err != nil {
		t.Fatalf("reconnect() error = %v", err)
	}

	statuses := mgr.statuses()
	if !statuses[0].connected || statuses[0].toolCount != 2 {
		t.Errorf("statuses[0] = %+v, want connected with 2 tools after reconnect", statuses[0])
	}
	if _, err := reg.Get("mcp__weather__get_weather"); err != nil {
		t.Errorf("reg.Get() error = %v, want the tool still registered after reconnect", err)
	}

	newPid := mgr.servers[0].client.Pid()
	if newPid == oldPid {
		t.Error("Pid unchanged after reconnect, want a fresh process")
	}
	if processAlive(oldPid) {
		t.Error("the pre-reconnect process is still alive, want it closed")
	}
}

func TestMCPManagerReconnectUnknownServerErrors(t *testing.T) {
	mgr := newMCPManager(context.Background(), tools.NewRegistry(), nil, time.Second)
	if err := mgr.reconnect(context.Background(), "nope"); err == nil {
		t.Fatal("reconnect() error = nil, want an error for an unconfigured server name")
	}
}

// TestMCPManagerShutdownLeavesNoOrphanedProcess is 13.3's AC: a test
// asserts no orphaned child process survives shutdown.
func TestMCPManagerShutdownLeavesNoOrphanedProcess(t *testing.T) {
	reg := tools.NewRegistry()
	cfgs := []mcp.ServerConfig{helperServerConfig(t, "a"), helperServerConfig(t, "b")}
	mgr := newMCPManager(context.Background(), reg, cfgs, 2*time.Second)

	var pids []int
	for _, s := range mgr.servers {
		if s.client == nil {
			t.Fatalf("server %q never connected", s.cfg.Name)
		}
		pids = append(pids, s.client.Pid())
	}
	if len(pids) != 2 {
		t.Fatalf("len(pids) = %d, want 2", len(pids))
	}
	for _, pid := range pids {
		if !processAlive(pid) {
			t.Fatalf("pid %d not alive before shutdown, test is not exercising anything", pid)
		}
	}

	mgr.shutdown()

	for _, pid := range pids {
		if processAlive(pid) {
			t.Errorf("pid %d still alive after shutdown, want it terminated", pid)
		}
	}
}

// processAlive reports whether pid names a live process, via the
// zero-signal probe (Unix: signal 0 checks existence/permission without
// actually signaling).
func processAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func TestLoadMCPServersEmptyPath(t *testing.T) {
	servers, err := loadMCPServers("")
	if err != nil {
		t.Fatalf("loadMCPServers(\"\") error = %v", err)
	}
	if servers != nil {
		t.Errorf("loadMCPServers(\"\") = %v, want nil", servers)
	}
}

func TestLoadMCPServersParsesAndSorts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mcp.json")
	data := `{"mcpServers": {
		"zeta": {"command": "zeta-bin", "args": ["--flag"], "env": {"A": "1"}},
		"alpha": {"command": "alpha-bin"}
	}}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	servers, err := loadMCPServers(path)
	if err != nil {
		t.Fatalf("loadMCPServers() error = %v", err)
	}
	if len(servers) != 2 {
		t.Fatalf("len(servers) = %d, want 2", len(servers))
	}
	if servers[0].Name != "alpha" || servers[1].Name != "zeta" {
		t.Errorf("servers = %+v, want alpha before zeta (sorted)", servers)
	}
	if servers[1].Command != "zeta-bin" || len(servers[1].Args) != 1 || servers[1].Args[0] != "--flag" {
		t.Errorf("servers[1] = %+v, want command zeta-bin and args [--flag]", servers[1])
	}
	if len(servers[1].Env) != 1 || servers[1].Env[0] != "A=1" {
		t.Errorf("servers[1].Env = %v, want [A=1]", servers[1].Env)
	}
}

func TestLoadMCPServersMissingFileErrors(t *testing.T) {
	if _, err := loadMCPServers(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("loadMCPServers() error = nil, want an error for a missing file")
	}
}

func TestLoadMCPServersInvalidJSONErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if _, err := loadMCPServers(path); err == nil {
		t.Fatal("loadMCPServers() error = nil, want an error for invalid JSON")
	}
}

const mcpGoldenPath = "testdata/golden/mcp.txt"

// TestMcpCommandMatchesGolden is 13.3's AC: a golden test of /mcp output
// across states — no servers configured, a mixed connected+failed
// listing, and the listing again after /mcp reconnect on the previously
// good server (proving reconnect works end to end through the command
// itself, including that it doesn't trip the duplicate-name detection on
// its own stale entries). Run with RECORD=1 to (re)write the golden file.
func TestMcpCommandMatchesGolden(t *testing.T) {
	provider := &fakeChatProvider{reply: "ok"}
	h := history.New()
	ag := agent.New(provider, tools.NewRegistry(), h, "", 1024, agent.Guards{}, 0, agent.AutoApprove)

	var out bytes.Buffer

	noneSess := noSession(t)
	if _, err := handleCommand(context.Background(), &out, ag, h, noneSess, "mcp", ""); err != nil {
		t.Fatalf("handleCommand(mcp) [no servers] error = %v", err)
	}
	out.WriteString("---\n")

	reg := tools.NewRegistry()
	mgr := newMCPManager(context.Background(), reg, []mcp.ServerConfig{helperServerConfig(t, "weather"), brokenServerConfig("broken")}, 2*time.Second)
	t.Cleanup(mgr.shutdown)
	sess := noSession(t)
	sess.mcp = mgr

	if _, err := handleCommand(context.Background(), &out, ag, h, sess, "mcp", ""); err != nil {
		t.Fatalf("handleCommand(mcp) [mixed] error = %v", err)
	}
	out.WriteString("---\n")

	if _, err := handleCommand(context.Background(), &out, ag, h, sess, "mcp", "reconnect weather"); err != nil {
		t.Fatalf("handleCommand(mcp reconnect) error = %v", err)
	}
	got := out.String()

	if os.Getenv("RECORD") == "1" {
		if err := os.MkdirAll(filepath.Dir(mcpGoldenPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(mcpGoldenPath, []byte(got), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	want, err := os.ReadFile(mcpGoldenPath)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v (run with RECORD=1 to create it)", mcpGoldenPath, err)
	}
	if got != string(want) {
		t.Errorf("/mcp output mismatch:\n--- want ---\n%s\n--- got ---\n%s", want, got)
	}
}
