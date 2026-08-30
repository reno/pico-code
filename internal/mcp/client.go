// Package mcp is a hand-rolled MCP client speaking JSON-RPC 2.0 over a
// server subprocess's stdio, one newline-delimited JSON message per line.
// There is no MCP SDK on CLAUDE.md's dependency allowlist, and 13.1's
// scope (initialize plus tools/list) is narrow enough that hand-rolling it
// follows the same precedent as the OpenAI adapter's raw net/http: no
// framework, explicit over clever.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"
)

// ServerConfig names one MCP server to launch over stdio.
type ServerConfig struct {
	Name    string
	Command string
	Args    []string
	// Env is appended to the child's inherited environment as "KEY=VALUE"
	// pairs, the same shape os/exec.Cmd.Env expects.
	Env []string
}

// Client is a live connection to one MCP server over stdio, established by
// Start after a completed initialize handshake.
type Client struct {
	name   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader

	// mu serializes request/response: stdio is a single pipe with no
	// message IDs multiplexed by the transport itself, so only one call
	// can be in flight at a time.
	mu     sync.Mutex
	nextID int64
}

// Start spawns cfg's process and performs the initialize handshake,
// failing (and killing the process) if the handshake doesn't complete
// within timeout. The process itself is tied to ctx for its whole
// lifetime — CLAUDE.md invariant 6, cancelling ctx must abort a running
// operation — while timeout only bounds the handshake step, so a slow but
// eventually-successful server isn't punished by the same deadline that
// protects startup from a server that never answers at all.
func Start(ctx context.Context, cfg ServerConfig, timeout time.Duration) (*Client, error) {
	cmd := exec.CommandContext(ctx, cfg.Command, cfg.Args...)
	if len(cfg.Env) > 0 {
		cmd.Env = append(os.Environ(), cfg.Env...)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: %q: stdin pipe: %w", cfg.Name, err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: %q: stdout pipe: %w", cfg.Name, err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: %q: start: %w", cfg.Name, err)
	}

	c := &Client{name: cfg.Name, cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}

	hctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := c.handshake(hctx); err != nil {
		_ = c.kill()
		return nil, fmt.Errorf("mcp: %q: handshake: %w", cfg.Name, err)
	}

	return c, nil
}

func (c *Client) handshake(ctx context.Context) error {
	params := initializeParams{
		ProtocolVersion: protocolVersion,
		Capabilities:    json.RawMessage(`{}`),
		ClientInfo:      clientInfo{Name: clientName, Version: clientVersion},
	}
	var result initializeResult
	if err := c.call(ctx, "initialize", params, &result); err != nil {
		return err
	}
	// notifications/initialized has no response — the MCP spec forbids
	// the server from replying to it — so this is fire-and-forget, not a
	// call.
	return c.notify("notifications/initialized", struct{}{})
}

// ListTools calls tools/list and returns the tools the server advertised.
func (c *Client) ListTools(ctx context.Context) ([]ToolInfo, error) {
	var result listToolsResult
	if err := c.call(ctx, "tools/list", struct{}{}, &result); err != nil {
		return nil, fmt.Errorf("mcp: %q: tools/list: %w", c.name, err)
	}
	return result.Tools, nil
}

// Call invokes tool on the server via tools/call with arguments and
// returns the server's raw result. Call has no opinion on what the
// result means for a caller building a string tool answer — flattening
// Content and applying a truncation budget is internal/tools' policy, not
// this transport's.
func (c *Client) Call(ctx context.Context, tool string, arguments json.RawMessage) (*CallToolResult, error) {
	var result CallToolResult
	params := callToolParams{Name: tool, Arguments: arguments}
	if err := c.call(ctx, "tools/call", params, &result); err != nil {
		return nil, fmt.Errorf("mcp: %q: tools/call %s: %w", c.name, tool, err)
	}
	return &result, nil
}

// Pid returns the server process's OS process ID, so a caller responsible
// for shutdown lifecycle (cmd/pico's /mcp manager and its tests) can
// verify a process is actually gone after Close, rather than just trusting
// Close returned.
func (c *Client) Pid() int { return c.cmd.Process.Pid }

// Close terminates the server process and waits for it to exit.
func (c *Client) Close() error {
	_ = c.stdin.Close()
	if c.cmd.Process == nil {
		return nil
	}
	_ = c.cmd.Process.Kill()
	return c.cmd.Wait()
}

// kill is Close's counterpart for a Client that never finished the
// handshake and so was never handed to a caller — same cleanup, private
// because half-initialized Clients don't escape this package.
func (c *Client) kill() error {
	_ = c.stdin.Close()
	if c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
	}
	return c.cmd.Wait()
}

// call sends a JSON-RPC request and waits for its matching response,
// returning ctx.Err() promptly if ctx is done first. The read happens in
// a goroutine because bufio.Reader.ReadBytes has no ctx awareness of its
// own; the goroutine's result is dropped on the floor if ctx wins the
// race, which is safe because the channel is buffered by 1 and the
// process is always killed shortly after a timeout (Start) or Close
// (everywhere else), which unblocks the pending read via EOF.
func (c *Client) call(ctx context.Context, method string, params, out any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.nextID++
	id := c.nextID
	if err := c.writeLine(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params}); err != nil {
		return fmt.Errorf("write %s request: %w", method, err)
	}

	type result struct {
		resp rpcResponse
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		line, err := c.stdout.ReadBytes('\n')
		if err != nil {
			ch <- result{err: fmt.Errorf("read %s response: %w", method, err)}
			return
		}
		var resp rpcResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			ch <- result{err: fmt.Errorf("decode %s response: %w", method, err)}
			return
		}
		ch <- result{resp: resp}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return r.err
		}
		if r.resp.Error != nil {
			return r.resp.Error
		}
		if out == nil {
			return nil
		}
		if err := json.Unmarshal(r.resp.Result, out); err != nil {
			return fmt.Errorf("decode %s result: %w", method, err)
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// notify sends a JSON-RPC notification and does not wait for (or expect)
// a response.
func (c *Client) notify(method string, params any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.writeLine(rpcNotification{JSONRPC: "2.0", Method: method, Params: params}); err != nil {
		return fmt.Errorf("write %s notification: %w", method, err)
	}
	return nil
}

// writeLine marshals v to one line of newline-delimited JSON on stdin, per
// MCP's stdio transport spec: messages are newline-delimited and must not
// contain an embedded newline. Callers hold c.mu.
func (c *Client) writeLine(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	data = append(data, '\n')
	_, err = c.stdin.Write(data)
	return err
}
