package greetings

import (
	"errors"
	"fmt"
)

func Hello(name string) (string, error) {
	if (name == "") {
		return "", errors.New("you provided an empty name my guy")
	}

	// ':=' simultaneously declares and intializes a var
	msg := fmt.Sprintf("Hi, %v. Welcome!", name)
	return msg, nil
}
