package day2

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	//"strings"
)

func main() {
	fname := "day2.txt"

	fmt.Printf("reading '%v'\n", fname)
	file, err := os.Open(fname)
	if err != nil {
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
	fmt.Println(input)
}

func findInvalids(start int, end int) []int {
	//var invalids = []int{} // slice

	return []int{0, 1}
}

func isInvalid(num int) bool {
	var digits = strconv.FormatInt(int64(num), 10)

	if len(digits)%2 == 1 {
		return false // odd number of digits -> no possible duplication
	}

	return digits[:(len(digits)/2)] == digits[(len(digits)/2):]

	//for i := 0; i<len(digits); i++ {
}
