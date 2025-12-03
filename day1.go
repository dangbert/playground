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
	//fname := "example.txt"

	const method2 = false

	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if (err != nil) {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	const DIAL_POSITIONS = 100

	// set true for running day1 part 1
	var result int = 0
	var state int = 50  // position of dial


	// iterate lines
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fmt.Println(line)

		chars := strings.Split(line, "")
		direction := chars[0]
		amount, err := strconv.Atoi(line[1:])

		if (err != nil) {
			log.Fatalf("failed to parse line '%s'", line)
		}

		// fmt.Printf("type of direction = %T\n", direction)
		// fmt.Printf("direction=%v, amount=%d\n", direction, amount)

		sign := 1
		if (direction == "L") {
			sign = -1
		}

		// adding DIAL_POSITIONS below as modulo can be negative in go!
		nextState := (state + sign * amount + DIAL_POSITIONS) % DIAL_POSITIONS

		if (!method2) {
			if (nextState == 0) {
				result += 1
			}
			state = nextState
			fmt.Println("flag1")
			continue
		}

		prevResult := result
		fullRotations := amount / DIAL_POSITIONS

		// nextState := (state + sign * amount) % DIAL_POSITIONS
		//fmt.Printf("state=%v, nextState=%v\n", state, nextState)

		if (fullRotations > 0) {
			result += fullRotations
		}


		// we passed 0 if we go into the negatives or >= DIAL_POSITIONS
		remainder := state + sign * (amount % DIAL_POSITIONS)


		// starting at 0 doesn't count as a click
		if (state != 0 && (remainder <= 0 || remainder >= DIAL_POSITIONS)) {
			result += 1
		}

		// landing on 0 (avoiding double count from above)
		/*
		if (nextState == 0 && state != 0) {
			result += 1
			fmt.Println("flag2")
			state = nextState
			continue
		}

		if (sign == -1 && nextState > state) {
			result += 1
			fmt.Println("flag3")
		} else if (sign == 1 && nextState < state) {
			result += 1
			fmt.Println("flag4")
		}
		*/

		//else if ((nextState > 0) != (state > 0)) {
		//	// signs differ so we passed 0
		//	fmt.Println("flag3")
		//	result += 1
		//}

		fmt.Printf("state=%v -> %v (result = %v -> %v)\n\n", state, nextState, prevResult, result)
		state = nextState

		//os.Exit(0)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}

	fmt.Printf("\nfinal result = %v", result)
}
