package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/reno/pico-code/internal/config"
	"github.com/reno/pico-code/internal/llm"
)

type fakeProvider struct{ name string }

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Chat(_ context.Context, _ llm.Request) (*llm.Response, error) {
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
}

func (f *fakeProvider) Stream(_ context.Context, _ llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event)
	close(ch)
	return ch, nil
}

var _ llm.Provider = (*fakeProvider)(nil)

func TestRegisterAndNew(t *testing.T) {
	name := "fake-" + t.Name()
	llm.Register(name, func(_ *config.Config) (llm.Provider, error) {
		return &fakeProvider{name: name}, nil
	})

	cfg := &config.Config{Provider: config.Provider(name)}
	p, err := llm.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if got := p.Name(); got != name {
		t.Errorf("Name() = %q, want %q", got, name)
	}
}

func TestNewUnregisteredProvider(t *testing.T) {
	cfg := &config.Config{Provider: "does-not-exist"}
	_, err := llm.New(cfg)
	if !errors.Is(err, llm.ErrProviderNotRegistered) {
		t.Fatalf("New() error = %v, want wrapping %v", err, llm.ErrProviderNotRegistered)
	}
}

func TestRegisterDuplicatePanics(t *testing.T) {
	name := "dup-" + t.Name()
	factory := func(_ *config.Config) (llm.Provider, error) { return &fakeProvider{name: name}, nil }
	llm.Register(name, factory)

	defer func() {
		if recover() == nil {
			t.Fatal("expected Register to panic on a duplicate name")
		}
	}()
	llm.Register(name, factory)
}
