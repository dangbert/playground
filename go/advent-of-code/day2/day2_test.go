package main

import (
	"reflect"
	"testing"
)

// https://pkg.go.dev/testing#T
func TestIsInvalidPart1(t *testing.T) {
	type testCase struct {
		input int
		want  bool
	}
	testCases := []testCase{
		{input: 11, want: true},
		{input: 12, want: false},
		{input: 22, want: true},
		{input: 9998, want: false},
		{input: 9999, want: true},
	}

	for _, cur := range testCases {
		res := isInvalidPart1(cur.input)
		if res != cur.want {
			t.Errorf(`Got %v, expected %v for input %v`, res, cur.want, cur.input)
		}
	}
}

func TestFindInvalids_part1(t *testing.T) {
	res := findInvalids(95, 115, false)
	want := []int{99}

	if !reflect.DeepEqual(res, want) {
		t.Errorf(`Got %c, expected %c`, res, want)
	}

	res = findInvalids(1698522, 1698528, false)
	want = []int{}

	if !reflect.DeepEqual(res, want) {
		t.Errorf(`Got %v, expected %v`, res, want)
	}
}

func TestIsInvalidPart2(t *testing.T) {
	type testCase struct {
		input int
		want  bool
	}
	testCases := []testCase{
		{input: 9, want: false},
		{input: 1010, want: true},
		{input: 101010, want: true},
		{input: 101210, want: false},
		{input: 456456456456, want: true},
		{input: 456456456, want: true},
	}

	for _, cur := range testCases {
		res := isInvalidPart2(cur.input)
		if res != cur.want {
			t.Errorf(`Got %v, expected %v for input %v`, res, cur.want, cur.input)
		}
	}

}
