// Package version holds the pico build version.
package version

// Version defaults to "dev" for local builds; GoReleaser overrides it with
// the release tag via -ldflags at build time.
var Version = "dev"
