package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

func main() {
	fname := "input1.txt"

	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if (err != nil) {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	const DIAL_POSITIONS = 100

	var result int = 0
	var state int = 50  // position of dial


	// iterate lines
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		// fmt.Println(line)

		chars := strings.Split(line, "")
		direction := chars[0]
		amount, err := strconv.Atoi(line[1:])

		if (err != nil) {
			log.Fatalf("failed to parse line '%s'", line)
		}

		// fmt.Printf("type of direction = %T\n", direction)
		// fmt.Printf("direction=%v, amount=%d", direction, amount)

		sign := 1
		if (direction == "L") {
			sign = -1
		}

		// var fullRotations = amount / DIAL_POSITIONS
		var nextState int = (state + sign * amount) % DIAL_POSITIONS

		if (nextState == 0) {
			result += 1
		}
		state = nextState

		//os.Exit(0)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}

	fmt.Printf("\nfinal result = %v", result)
}
