package greetings

import (
	"testing"
	"regexp"
)

// https://pkg.go.dev/testing#T
func TestHelloName(t *testing.T) {
	name := "Dan"
	want := regexp.MustCompile(`\b`+name+`\b`)

	msg, err := Hello(name)
	if (!want.MatchString(msg) || err != nil) {
		t.Errorf(`Got %q, %v, want match for %#q, nil`, msg, err, want)
	}
}

func TestHelloEmpty(t *testing.T) {
	msg, err := Hello("")

	if (err == nil || msg != "") {
		t.Errorf(`Got %q, %v, want match for "", (an error)`, msg, err)
	}
}
