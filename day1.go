package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

func main() {
	fmt.Println("hi")
	fname := "input1.txt"

	file, err := os.Open(fname)
	if (err != nil) {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// iterate lines
	for scanner.Scan() {
		line := scanner.Text()
		fmt.Println(line)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}
}
