.PHONY: build test lint run run-plain

RUN_FLAGS = --provider=ollama --model=qwen3.5:4b --tools=native --think --num-ctx 8192

build:
	go build -o bin/pico ./cmd/pico

test:
	go test ./...

lint:
	gofmt -l .
	go vet ./...
	golangci-lint run

run: build
	./bin/pico $(RUN_FLAGS) --tui

run-plain: build
	./bin/pico $(RUN_FLAGS)
