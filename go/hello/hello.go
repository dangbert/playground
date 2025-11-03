package main

import (
	"fmt"
	"log"
	// https://pkg.go.dev/rsc.io/quote@v1.5.2
	"rsc.io/quote"
	// my module
	"example.com/greetings"
)


func main() {
	log.SetPrefix("[greetings]: ")
	log.SetFlags(0)

	fmt.Println("what's up")
	fmt.Println(quote.Opt())

	msg, err := greetings.Hello("jullie")
	//fmt.Printf("msg: '%v', err: '%v'\n", msg, err)
	fmt.Println(msg)

	msg, err = greetings.Hello("")
	fmt.Println(msg)

	if err != nil {
		log.Fatal(err)
	}
}
