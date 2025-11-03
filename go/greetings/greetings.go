package greetings

import (
	"errors"
	"fmt"
	"math/rand"
)

func Hello(name string) (string, error) {
	if (name == "") {
		return "", errors.New("you provided an empty name my guy")
	}

	// ':=' simultaneously declares and intializes a var
	msg := fmt.Sprintf(randomFormat(), name)
	return msg, nil
}

func Hellos(names []string) (map[string]string, error) {
	messages := make(map[string]string)

	for _, name := range names {
		msg, err := Hello(name)
		if err != nil {
			return nil, err
		}

		messages[name] = msg
	}
	return messages, nil
}

func randomFormat() string {
	// possible message formats
	formats := []string{
		"Hi, %v. Welcome!",
		"Hoi %v hoe gaat het?",
		"Bom dia %v, tudo bem?",
	}

	idx := rand.Intn(len(formats))
	return formats[idx]
}
