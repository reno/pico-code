// Package recordutil implements the RECORD=1 convention CLAUDE.md's
// testing section describes: with RECORD=1 set and real credentials in the
// environment, an adapter's test proxies one live exchange through a
// Recorder that captures the raw request/response and writes them to disk
// as a scrubbed golden fixture. Without RECORD=1 (the default, and the
// only mode `make test` ever runs), nothing in this package is exercised —
// tests replay already-recorded fixtures via httptest.Server instead, so
// the suite stays offline and key-free.
package recordutil

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
)

// Enabled reports whether RECORD=1 is set.
func Enabled() bool {
	return os.Getenv("RECORD") == "1"
}

// apiKeyPattern matches an API-key-shaped literal (Anthropic's sk-ant-...,
// OpenAI-style sk-..., and similar) so Scrub catches a stray key even if a
// caller forgot to list it explicitly among secrets.
var apiKeyPattern = regexp.MustCompile(`sk-[A-Za-z0-9_-]{10,}`)

// Scrub redacts every occurrence of each secret in s, plus anything
// matching apiKeyPattern, so a recorded fixture can never contain a live
// credential.
func Scrub(s string, secrets ...string) string {
	for _, secret := range secrets {
		if secret == "" {
			continue
		}
		s = strings.ReplaceAll(s, secret, "REDACTED")
	}
	return apiKeyPattern.ReplaceAllString(s, "sk-REDACTED")
}

// Exchange is one recorded request/response pair, already read into memory
// so a Recorder's caller can inspect and scrub it after the real
// RoundTrip completes.
type Exchange struct {
	RequestBody  []byte
	ResponseBody []byte
	StatusCode   int
}

// Recorder wraps an http.RoundTripper, forwarding every request unchanged
// but also calling OnExchange with the raw request/response bytes —
// letting a test capture a live exchange without altering how the request
// is actually sent.
type Recorder struct {
	Transport  http.RoundTripper
	OnExchange func(Exchange)
}

// RoundTrip implements http.RoundTripper.
func (r *Recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	var reqBody []byte
	if req.Body != nil {
		var err error
		reqBody, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		req.Body = io.NopCloser(bytes.NewReader(reqBody))
	}

	transport := r.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(respBody))

	if r.OnExchange != nil {
		r.OnExchange(Exchange{RequestBody: reqBody, ResponseBody: respBody, StatusCode: resp.StatusCode})
	}
	return resp, nil
}
