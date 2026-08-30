package config

// ModelPricing is one model's per-token rate, in USD per million tokens.
// These are estimates snapshotted at some point in time from the
// provider's published pricing — prices change, and this table is not
// refreshed automatically. Treat any cost figure derived from it as an
// approximation for budgeting, never as a billing-accurate number.
type ModelPricing struct {
	InputPerMTok      float64
	OutputPerMTok     float64
	CacheWritePerMTok float64
	CacheReadPerMTok  float64
}

// pricing is keyed by the exact model ID string a provider expects (the
// same string --model/cfg.Model carries). A model with no entry here —
// a typo, a brand-new release, a local Ollama model — simply has no cost:
// PriceFor's ok return exists precisely so callers never have to treat an
// unpriced model as an error.
var pricing = map[string]ModelPricing{
	"claude-haiku-4-5":           {InputPerMTok: 1.00, OutputPerMTok: 5.00, CacheWritePerMTok: 1.25, CacheReadPerMTok: 0.10},
	"claude-haiku-4-5-20251001":  {InputPerMTok: 1.00, OutputPerMTok: 5.00, CacheWritePerMTok: 1.25, CacheReadPerMTok: 0.10},
	"claude-sonnet-4-5":          {InputPerMTok: 3.00, OutputPerMTok: 15.00, CacheWritePerMTok: 3.75, CacheReadPerMTok: 0.30},
	"claude-sonnet-4-5-20250929": {InputPerMTok: 3.00, OutputPerMTok: 15.00, CacheWritePerMTok: 3.75, CacheReadPerMTok: 0.30},
	"claude-sonnet-5":            {InputPerMTok: 3.00, OutputPerMTok: 15.00, CacheWritePerMTok: 3.75, CacheReadPerMTok: 0.30},
	"claude-opus-4-5":            {InputPerMTok: 15.00, OutputPerMTok: 75.00, CacheWritePerMTok: 18.75, CacheReadPerMTok: 1.50},
	"claude-opus-4-5-20251101":   {InputPerMTok: 15.00, OutputPerMTok: 75.00, CacheWritePerMTok: 18.75, CacheReadPerMTok: 1.50},
	"claude-opus-5":              {InputPerMTok: 15.00, OutputPerMTok: 75.00, CacheWritePerMTok: 18.75, CacheReadPerMTok: 1.50},
}

// PriceFor returns model's pricing and whether one is known. A false ok
// means "no cost available," not an error — every caller must treat it
// that way (an unrecognized --model is routine, not exceptional).
func PriceFor(model string) (ModelPricing, bool) {
	p, ok := pricing[model]
	return p, ok
}

// Cost computes usage's estimated cost in USD against model's pricing.
// The second return is false whenever model has no pricing entry, in
// which case the float is always zero.
func (p ModelPricing) Cost(inputTokens, outputTokens, cacheWriteTokens, cacheReadTokens int) float64 {
	const perToken = 1.0 / 1_000_000
	return float64(inputTokens)*p.InputPerMTok*perToken +
		float64(outputTokens)*p.OutputPerMTok*perToken +
		float64(cacheWriteTokens)*p.CacheWritePerMTok*perToken +
		float64(cacheReadTokens)*p.CacheReadPerMTok*perToken
}
