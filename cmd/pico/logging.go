package main

import (
	"fmt"
	"io"
	"log/slog"

	"github.com/reno/pico-code/internal/config"
)

// setupLogging points the default slog logger at w with cfg.LogLevel's
// severity. Called once at startup; only --log-level=debug ever dumps
// provider request/response bodies (9.2), and everything logs to w
// (stderr in production) rather than stdout, so piped/plain chat output
// stays clean regardless of level.
func setupLogging(w io.Writer, level config.LogLevel) error {
	slogLevel, err := parseLogLevel(level)
	if err != nil {
		return err
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slogLevel})))
	return nil
}

func parseLogLevel(level config.LogLevel) (slog.Level, error) {
	switch level {
	case config.LogLevelDebug:
		return slog.LevelDebug, nil
	case config.LogLevelInfo:
		return slog.LevelInfo, nil
	case config.LogLevelWarn:
		return slog.LevelWarn, nil
	case config.LogLevelError:
		return slog.LevelError, nil
	default:
		// config.Load already validates this; reaching here means a caller
		// built a Config by hand rather than through Load.
		return 0, fmt.Errorf("logging: unknown level %q", level)
	}
}
