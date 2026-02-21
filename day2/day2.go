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
	fname := "day2.txt"

	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if (err != nil) {
		log.Fatalf("failed to open '%v', %s", fname, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var input string = ""
	for scanner.Scan() {
		input = scanner.Text()
		break
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading file: %s", err)
	}



}

func findInvalids(start int, end int) []int {
	var invalids []int = []

}
