package main

import (
	"bufio"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strconv"
	"strings"
)

func main() {
	log.SetPrefix("[day2]: ")
	log.SetFlags(0)

	fname := "example.txt"
	slog.Info(fmt.Sprintf("reading '%v'\n", fname))
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
	slog.Info(fmt.Sprintf("found %v ID ranges", len(ranges)))
	for _, v := range strings.Split(input, ",") {
		var r []string = strings.Split(v, "-")
		if len(r) != 2 {
			panic(fmt.Sprintf("expected split of length 2: %v", r))
		}

		var stop int
		start, err := strconv.Atoi(r[0])
		if err != nil {
			log.Fatal(fmt.Sprintf("failed to parse int '%v', %v", start, err))
		}
		stop, err = strconv.Atoi(r[0])
		if err != nil {
			log.Fatal(fmt.Sprintf("failed to parse int '%v', %v", start, err))
		}

		// sum results
		invalids := findInvalids(start, stop)
		if len(invalids) == 0 {
			continue
		}
		slog.Info(fmt.Sprintf("invalids: %v", invalids))
		for _, invalidId := range invalids {
			sum += invalidId
		}
	}

	slog.Info(fmt.Sprintf("final result = %v", sum))
}

// returns the "invalid" IDs in the given range (end inclusive).
func findInvalids(start int, end int) []int {
	slog.Debug(fmt.Sprintf("s\nearching range %v-%v", start, end))
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
