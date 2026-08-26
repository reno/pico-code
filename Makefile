.PHONY: build test lint run

build:
	go build -o bin/pico ./cmd/pico

test:
	go test ./...

lint:
	gofmt -l .
	go vet ./...
	golangci-lint run

run:
	go run ./cmd/pico chat --provider=ollama
