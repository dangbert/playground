package main

import "fmt"

// https://pkg.go.dev/rsc.io/quote@v1.5.2
import "rsc.io/quote"

import "example.com/greetings"

func main() {
	fmt.Println("what's up")
	fmt.Println(quote.Opt())

	msg := greetings.Hello("jullie")
	fmt.Println(msg)
}
