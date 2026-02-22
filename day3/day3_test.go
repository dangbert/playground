package main

import (
	"testing"
)

func TestIsInvalidPart1(t *testing.T) {
	type testCase struct {
		input string
		want  int
	}
	testCases := []testCase{
		{input: "234234234234278", want: 78},
		{input: "818181911112111", want: 92},
	}

	for _, cur := range testCases {
		res := maxJoltage(cur.input)
		if res != cur.want {
			t.Errorf(`Got %v, expected %v for input %v`, res, cur.want, cur.input)
		}
	}
}
