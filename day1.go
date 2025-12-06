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
	// fname := "example.txt"

	// set true for running day1 part 2
	const method2 = true

	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if err != nil {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	const DIAL_POSITIONS = 100

	var result int = 0
	var state int = 50 // position of dial

	// iterate lines
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fmt.Println(line)

		chars := strings.Split(line, "")
		direction := chars[0]
		amount, err := strconv.Atoi(line[1:])

		if err != nil {
			log.Fatalf("failed to parse line '%s'", line)
		}

		// fmt.Printf("type of direction = %T\n", direction)
		// fmt.Printf("direction=%v, amount=%d\n", direction, amount)

		sign := 1
		if direction == "L" {
			sign = -1
		}

		nextState := positiveMod(state+sign*amount, DIAL_POSITIONS)

		if !method2 {
			if nextState == 0 {
				result += 1
			}
			state = nextState
			fmt.Println("flag1")
			continue
		}

		prevResult := result
		fullRotations := amount / DIAL_POSITIONS

		if fullRotations > 0 {
			result += fullRotations
			// now we can consider the leftovers
			amount -= fullRotations * DIAL_POSITIONS
			fmt.Printf("fullRotations = %d\n", fullRotations)
		}

		if sign == 1 {
			if (state+amount) >= DIAL_POSITIONS && state != 0 {
				result += 1
			}
		} else {
			if (state-amount) <= 0 && state != 0 {
				result += 1
			}
		}

		fmt.Printf("state=%v -> %v (result = %v -> %v)\n\n", state, nextState, prevResult, result)
		state = nextState
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}

	fmt.Printf("\nfinal result = %v", result)
}

// get the remainder of a / b, always as a positive number
// https://labex.io/tutorials/go-how-to-perform-modulo-operations-in-go-418324
func positiveMod(a int, b int) int {
	return ((a % b) + b) % b
}
