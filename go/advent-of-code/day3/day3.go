package main

import (
	"bufio"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strconv"
)

func main() {
	log.SetPrefix("[day3]: ")
	log.SetFlags(0)

	var logLevel = new(slog.LevelVar)
	// logLevel.Set(slog.LevelDebug)
	logLevel.Set(slog.LevelInfo)
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})
	slog.SetDefault(slog.New(handler))

	fname := "example.txt"
	//fname := "input.txt"

	slog.Info(fmt.Sprintf("reading '%v'", fname))
	file, err := os.Open(fname)
	if err != nil {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	var result int = 0

	// parse file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		var joltage = maxJoltage(line)
		slog.Info(fmt.Sprintf("%v => %v", line, joltage))
		result += joltage
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}

	slog.Info(fmt.Sprintf("final result = %v", result))
}

func maxJoltage(bank string) int {
	var err error
	var jolt1, jolt2 int // left, right-most max digit

	// note this is overly verbose / complicated
	jolt1, err = strconv.Atoi(string(bank[0]))
	assertNotNil(err)

	jolt2, err = strconv.Atoi(string(bank[1]))
	assertNotNil(err)

	var cur int
	for _, digit := range bank[2:] {
		cur, err = strconv.Atoi(string(digit))
		assertNotNil(err)

		if cur > jolt1 || cur > jolt2 {
			jolt1 = max(jolt1, jolt2)
			jolt2 = cur
		}
	}
	return jolt1*10 + jolt2
}

func assertNotNil(err error) {
	if err != nil {
		panic(err)
	}
}
