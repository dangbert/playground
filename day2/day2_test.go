package main

import (
	"reflect"
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

func TestFindInvalids(t *testing.T) {
	res := findInvalids(95, 115)
	want := []int{99}

	if !reflect.DeepEqual(res, want) {
		t.Errorf(`Got %c, expected %c`, res, want)
	}

	res = findInvalids(1698522, 1698528)
	want = []int{}

	if !reflect.DeepEqual(res, want) {
		t.Errorf(`Got %v, expected %v`, res, want)
	}
}
