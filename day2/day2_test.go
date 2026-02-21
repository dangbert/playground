package day2

import (
	"testing"
)

// https://pkg.go.dev/testing#T
func TestIsInvalid(t *testing.T) {
	input := 11
	res := isInvalid(input)
	want := true
	if res != want {
		t.Errorf(`Got %v, expected %v for input %v`, res, want, input)
	}

	input = 12
	res = isInvalid(input)
	want = false
	if res != want {
		t.Errorf(`Got %v, expected %v for input %v`, res, want, input)
	}

	input = 22
	res = isInvalid(input)
	want = true
	if res != want {
		t.Errorf(`Got %v, expected %v for input %v`, res, want, input)
	}
}
