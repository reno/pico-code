package recordutil

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestScrubRedactsExactSecretsAndKeyShapedLiterals(t *testing.T) {
	in := `{"header":"Bearer top-secret-value","key":"sk-ant-abc123XYZ456","other":"fine"}`
	got := Scrub(in, "top-secret-value")

	if strings.Contains(got, "top-secret-value") {
		t.Errorf("Scrub() = %q, still contains the exact secret", got)
	}
	if strings.Contains(got, "sk-ant-abc123XYZ456") {
		t.Errorf("Scrub() = %q, still contains an sk-shaped key literal", got)
	}
	if !strings.Contains(got, "other") || !strings.Contains(got, "fine") {
		t.Errorf("Scrub() = %q, want unrelated content left alone", got)
	}
}

func TestScrubIgnoresEmptySecrets(t *testing.T) {
	got := Scrub("hello world", "", "world")
	if got != "hello REDACTED" {
		t.Errorf("Scrub() = %q, want empty secrets skipped and non-empty ones applied", got)
	}
}

func TestRecorderCapturesExchangeAndForwardsResponseUnchanged(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if string(body) != `{"hello":"world"}` {
			t.Errorf("server saw body = %q", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	var captured Exchange
	client := &http.Client{Transport: &Recorder{
		OnExchange: func(e Exchange) { captured = e },
	}}

	resp, err := client.Post(srv.URL, "application/json", bytes.NewReader([]byte(`{"hello":"world"}`)))
	if err != nil {
		t.Fatalf("Post() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(respBody) != `{"ok":true}` {
		t.Errorf("response body (after recording) = %q, want the server's original body untouched", respBody)
	}

	if string(captured.RequestBody) != `{"hello":"world"}` {
		t.Errorf("captured.RequestBody = %q", captured.RequestBody)
	}
	if string(captured.ResponseBody) != `{"ok":true}` {
		t.Errorf("captured.ResponseBody = %q", captured.ResponseBody)
	}
	if captured.StatusCode != http.StatusOK {
		t.Errorf("captured.StatusCode = %d, want 200", captured.StatusCode)
	}
}

func TestEnabledReflectsRecordEnvVar(t *testing.T) {
	t.Setenv("RECORD", "")
	if Enabled() {
		t.Error("Enabled() = true with RECORD unset, want false")
	}
	t.Setenv("RECORD", "1")
	if !Enabled() {
		t.Error("Enabled() = false with RECORD=1, want true")
	}
}
