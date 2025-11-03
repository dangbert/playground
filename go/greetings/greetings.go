package greetings

import "fmt"

func Hello(name string) string {
	// ':=' simultaneously declares and intializes a var
	msg := fmt.Sprintf("Hi, %v. Welcome!", name)
	return msg
}
