package main

import (
	"bytes"
	"log/slog"
	"testing"

	"github.com/reno/pico-code/internal/config"
)

func TestSetupLoggingAppliesConfiguredLevel(t *testing.T) {
	var buf bytes.Buffer
	if err := setupLogging(&buf, config.LogLevelWarn); err != nil {
		t.Fatalf("setupLogging() error = %v", err)
	}
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))) })

	slog.Info("should not appear")
	slog.Warn("should appear")

	out := buf.String()
	if bytes.Contains([]byte(out), []byte("should not appear")) {
		t.Errorf("Info-level line logged despite --log-level=warn: %q", out)
	}
	if !bytes.Contains([]byte(out), []byte("should appear")) {
		t.Errorf("Warn-level line missing: %q", out)
	}
}

func TestParseLogLevelRejectsUnknownValue(t *testing.T) {
	if _, err := parseLogLevel(config.LogLevel("verbose")); err == nil {
		t.Fatal("parseLogLevel() = nil error, want one for an unknown level")
	}
}
