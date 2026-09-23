package main

import (
	"io"
	"log"
	"math/rand"
	"net/http"
	"strconv"
)

func rolldice(w http.ResponseWriter, r *http.Request) {
	var msg string
	player := r.PathValue("player")

	if player != "" {
		msg = player + " rolling the dice"
	} else {
		msg = "anon player rolling the dice"
	}

	roll := 1 + rand.Intn(6) // rand int in range [1,6]
	log.Printf("%s, result: %d", msg, roll)

	res := strconv.Itoa(roll) + "\n"

	if _, err := io.WriteString(w, res); err != nil {
		log.Printf("Write failed %v", err)
	}
}
