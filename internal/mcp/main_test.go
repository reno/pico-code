package mcp

import (
	"testing"

	"go.uber.org/goleak"
)

// TestMain guards against a leaked read goroutine: call's background
// reader is abandoned (not joined) whenever ctx wins the race against a
// pending response, relying on the killed process's closed pipe to
// eventually unblock it. This proves that unblocking actually happens
// (CLAUDE.md invariant 6) rather than leaving a goroutine parked forever.
func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}
