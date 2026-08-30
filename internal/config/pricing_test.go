package config

import (
	"math"
	"testing"
)

const costEpsilon = 1e-9

func TestPriceForKnownModel(t *testing.T) {
	price, ok := PriceFor("claude-sonnet-4-5")
	if !ok {
		t.Fatal("PriceFor(claude-sonnet-4-5) ok = false, want true")
	}
	if price.InputPerMTok != 3.00 || price.OutputPerMTok != 15.00 {
		t.Errorf("price = %+v, want InputPerMTok 3.00, OutputPerMTok 15.00", price)
	}
}

// TestPriceForUnknownModelReturnsNotOK is half of 15.1's AC: an unknown
// model raises no error — PriceFor just reports it has nothing.
func TestPriceForUnknownModelReturnsNotOK(t *testing.T) {
	price, ok := PriceFor("not-a-real-model")
	if ok {
		t.Fatalf("PriceFor(not-a-real-model) ok = true, want false")
	}
	if price != (ModelPricing{}) {
		t.Errorf("price = %+v, want the zero value when ok is false", price)
	}
}

// TestModelPricingCostMatchesHandComputedFigure is the other half of
// 15.1's AC: a hand-computed figure to the cent.
func TestModelPricingCostMatchesHandComputedFigure(t *testing.T) {
	p := ModelPricing{InputPerMTok: 3.00, OutputPerMTok: 15.00, CacheWritePerMTok: 3.75, CacheReadPerMTok: 0.30}

	// 100,000 input + 100,000 output: 0.3 + 1.5 = 1.8
	if got, want := p.Cost(100_000, 100_000, 0, 0), 1.8; math.Abs(got-want) > costEpsilon {
		t.Errorf("Cost(100_000, 100_000, 0, 0) = %v, want %v", got, want)
	}
	// + 200,000 cache-write + 1,000,000 cache-read: 1.8 + 0.75 + 0.30 = 2.85
	if got, want := p.Cost(100_000, 100_000, 200_000, 1_000_000), 2.85; math.Abs(got-want) > costEpsilon {
		t.Errorf("Cost(100_000, 100_000, 200_000, 1_000_000) = %v, want %v", got, want)
	}
	if got, want := p.Cost(0, 0, 0, 0), 0.0; got != want {
		t.Errorf("Cost(0, 0, 0, 0) = %v, want %v", got, want)
	}
}
