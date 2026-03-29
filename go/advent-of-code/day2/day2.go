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

	var logLevel = new(slog.LevelVar)
	// logLevel.Set(slog.LevelDebug)
	logLevel.Set(slog.LevelInfo)
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})
	slog.SetDefault(slog.New(handler))

	//fname := "example.txt"
	fname := "input.txt"
	const part2 = true

	slog.Info(fmt.Sprintf("reading '%v'", fname))
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
	slog.Debug(input)

	// analyze ranges in file
	var sum int = 0
	var ranges []string = strings.Split(input, ",")
	slog.Info(fmt.Sprintf("found %v ID ranges", len(ranges)))
	slog.Info(fmt.Sprintf("ranges = %v", ranges))
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
		stop, err = strconv.Atoi(r[1])
		if err != nil {
			log.Fatal(fmt.Sprintf("failed to parse int '%v', %v", start, err))
		}

		// sum results
		invalids := findInvalids(start, stop, part2)
		if len(invalids) == 0 {
			continue
		}
		slog.Debug(fmt.Sprintf("invalids: %v", invalids))
		for _, invalidId := range invalids {
			sum += invalidId
		}
	}

	slog.Info(fmt.Sprintf("final result = %v", sum))
}

// returns the "invalid" IDs in the given range (end inclusive).
func findInvalids(start int, end int, part2 bool) []int {
	slog.Debug(fmt.Sprintf("searching range %v-%v", start, end))
	var invalids = []int{} // slice

	isInvalid := isInvalidPart1
	if part2 {
		isInvalid = isInvalidPart2
	}

	for i := start; i <= end; i++ {
		if isInvalid(i) {
			invalids = append(invalids, i)
		}
	}

	slog.Debug(fmt.Sprintf("%v", invalids))
	return invalids
}

func isInvalidPart1(num int) bool {
	var digits = strconv.FormatInt(int64(num), 10)

	if len(digits)%2 == 1 {
		return false // odd number of digits -> no possible duplication
	}
	return digits[:(len(digits)/2)] == digits[(len(digits)/2):]
}

func isInvalidPart2(num int) bool {
	// slog.Debug(fmt.Sprintf("checking %v", num))
	var digits = strconv.FormatInt(int64(num), 10)

	for groupSize := 1; groupSize <= len(digits)/2; groupSize++ {
		if len(digits)%groupSize != 0 {
			continue
		}

		target := digits[0:groupSize] // candidate group for repeating
		invalid := true

		if groupSize == 1 {
			if digits == strings.Repeat(string(digits[0]), len(digits)) {
				return true
			} else {
				continue
			}
		}

		for groupNum := 0; groupNum < len(digits)/groupSize; groupNum++ {
			if digits[(groupNum*groupSize):((groupNum+1)*groupSize)] != target {
				invalid = false
				break
			}
		}
		if invalid {
			return true
		}
	}
	return false
}
