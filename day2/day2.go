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
	fname := "input.txt"
	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if err != nil {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	// parse file
	var input string = ""
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		input = scanner.Text()
		break
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}
	fmt.Println(input)

	// analyze ranges in file
	var sum int = 0
	var ranges []string = strings.Split(input, ",")
	fmt.Printf("found %v ID ranges", len(ranges))
	for _, v := range strings.Split(input, ",") {
		var r []string = strings.Split(v, "-")
		if (len(r) != 2) {
			panic(fmt.Sprintf("expected split of length 2: %v", r))
		}

		var stop int
		start, err := strconv.Atoi(r[0])
		if err != nil {
			panic(fmt.Sprintf("failed to parse int '%v', %v", start, err))
		}
		stop, err = strconv.Atoi(r[0])
		if err != nil {
			panic(fmt.Sprintf("failed to parse int '%v', %v", start, err))
		}

		invalids := findInvalids(start, stop) 
		if len(invalids) > 0 {
			fmt.Printf("\ninvalids: %v", invalids)
			for _, invalidId := range invalids {
				sum += invalidId
			}
		}
	}

	fmt.Printf("\nfinal result = %v", sum)
}

// returns the "invalid" IDs in the given range (end inclusive).
func findInvalids(start int, end int) []int {
	var invalids = []int{} // slice
	for i := start; i <= end; i++ {
		if isInvalid(i) {
			invalids = append(invalids, i)
		}
	}

	return invalids
}

func isInvalid(num int) bool {
	var digits = strconv.FormatInt(int64(num), 10)

	if len(digits)%2 == 1 {
		return false // odd number of digits -> no possible duplication
	}
	return digits[:(len(digits)/2)] == digits[(len(digits)/2):]
}
