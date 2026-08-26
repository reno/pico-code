package prompted

import "testing"

// FuzzExtractToolCall backs the "never a panic" half of 3.3's AC directly:
// go test's fuzzer treats any panic as a failure, so this is a genuine
// proof rather than just confidence from reading the code.
func FuzzExtractToolCall(f *testing.F) {
	seeds := []string{
		"The weather looks nice today, no need to check.",
		"Let me check.\n```json\n{\"tool\":\"get_weather\",\"input\":{\"location\":\"Paris\"}}\n```\nOne moment.",
		"```json\n{'tool': 'get_weather', 'input': {'location': 'Paris'}}\n```",
		"```json\n[\"get_weather\", {\"location\":\"Paris\"}]\n```",
		"```json\n{\"tool\":\"time_travel\",\"input\":{}}\n```",
		"```json\n{\"tool\":\"get_weather\",\"input\":{\"location\":\"Paris\"}}\n```",
		"",
		"```",
		"``` ``` ```",
		"```json```",
		"```json\n\n```",
		"no fence at all, just ``` a stray backtick run ``` in prose",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, text string) {
		prefix, suffix, call, ok := extractToolCall(text)
		if !ok {
			return
		}
		if call == nil {
			t.Fatalf("extractToolCall(%q) returned ok=true with a nil call", text)
		}
		_ = prefix
		_ = suffix
	})
}
