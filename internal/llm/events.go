package llm

// Event is one item in a Provider's Stream. It is sealed to this package
// (via the unexported isEvent method), mirroring Block, so a consumer's
// type switch is exhaustive by construction. Concrete variants (TextDelta,
// ToolUseStart, ToolUseArgsDelta, ToolUseDone, MessageDone, Error) are added
// in phase 6.1 alongside the streaming adapters that produce them; Provider
// only needs the type to exist to declare its Stream method now.
type Event interface {
	isEvent()
}
