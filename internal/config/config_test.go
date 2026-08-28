package config

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func baseFlags() Flags {
	return Flags{
		Provider:    "anthropic",
		Model:       "claude-x",
		MaxTurns:    25,
		TokenBudget: 100000,
		Workspace:   ".",
		Tools:       "native",
		Stream:      true,
		LogLevel:    "info",
	}
}

func noEnv(string) string { return "" }

func TestLoad(t *testing.T) {
	tests := []struct {
		name    string
		flags   Flags
		getenv  func(string) string
		want    *Config
		wantErr error
	}{
		{
			name:  "valid anthropic",
			flags: baseFlags(),
			getenv: func(k string) string {
				if k == "ANTHROPIC_API_KEY" {
					return "sk-test"
				}
				return ""
			},
			want: &Config{
				Provider:        ProviderAnthropic,
				Model:           "claude-x",
				MaxTurns:        25,
				TokenBudget:     100000,
				Workspace:       ".",
				Tools:           ToolsNative,
				Stream:          true,
				LogLevel:        "info",
				AnthropicAPIKey: "sk-test",
			},
		},
		{
			name: "valid ollama picks up host",
			flags: func() Flags {
				f := baseFlags()
				f.Provider = "ollama"
				return f
			}(),
			getenv: func(k string) string {
				if k == "OLLAMA_HOST" {
					return "http://localhost:11434"
				}
				return ""
			},
			want: &Config{
				Provider:    ProviderOllama,
				Model:       "claude-x",
				MaxTurns:    25,
				TokenBudget: 100000,
				Workspace:   ".",
				Tools:       ToolsNative,
				Stream:      true,
				LogLevel:    "info",
				OllamaHost:  "http://localhost:11434",
			},
		},
		{
			name: "unknown provider",
			flags: func() Flags {
				f := baseFlags()
				f.Provider = "openai"
				return f
			}(),
			getenv:  noEnv,
			wantErr: ErrUnknownProvider,
		},
		{
			name: "empty provider",
			flags: func() Flags {
				f := baseFlags()
				f.Provider = ""
				return f
			}(),
			getenv:  noEnv,
			wantErr: ErrUnknownProvider,
		},
		{
			name: "empty model",
			flags: func() Flags {
				f := baseFlags()
				f.Model = ""
				return f
			}(),
			getenv:  noEnv,
			wantErr: ErrModelRequired,
		},
		{
			name: "unknown tools mode",
			flags: func() Flags {
				f := baseFlags()
				f.Tools = "auto"
				return f
			}(),
			getenv:  noEnv,
			wantErr: ErrUnknownToolsMode,
		},
		{
			name: "unknown log level",
			flags: func() Flags {
				f := baseFlags()
				f.LogLevel = "verbose"
				return f
			}(),
			getenv:  noEnv,
			wantErr: ErrUnknownLogLevel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Load(tt.flags, tt.getenv)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("Load() error = %v, want wrapping %v", err, tt.wantErr)
				}
				if err.Error() == "" {
					t.Fatal("expected a descriptive error message")
				}
				return
			}
			if err != nil {
				t.Fatalf("Load() unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("Load() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
