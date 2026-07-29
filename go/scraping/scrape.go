//usr/bin/env go run "$0" "$@"; exit
// USAGE: ./scrape.go -baseUrl=example.com/blog/ -start=1 -end=5 -j10

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

func main() {
	baseUrlPtr := flag.String("baseUrl", "", "url to scrape")
	startPtr := flag.Int("start", 1, "start num to append to url")
	endPtr := flag.Int("end", -1, "final num to append to url (incremented sequentially)")
	jPtr := flag.Int("j", 4, "max concurrent threads")
	flag.Parse()

	if *baseUrlPtr == "" {
		fmt.Printf("missing arg: -baseUrl=example.com")
		os.Exit(1)
	}

	if *startPtr < 0 {
		fmt.Printf("start=%v must be >= 0\n", *startPtr)
		os.Exit(1)
	}
	if *endPtr < *startPtr {
		fmt.Printf("end=%v must be >= 0\n", *endPtr)
		os.Exit(1)
	}

	if !strings.HasPrefix(*baseUrlPtr, "http") {
		*baseUrlPtr = "https://" + *baseUrlPtr
	}
	if !strings.HasSuffix(*baseUrlPtr, "/") {
		*baseUrlPtr = *baseUrlPtr + "/"
	}

	fmt.Printf("scraping '%v%v' -> '%v%v' (%v threads)\n", *baseUrlPtr, *startPtr, *baseUrlPtr, *startPtr, *jPtr)

	var nextNum uint64 = uint64(*startPtr) // next number to use for scraping
	// results := make(chan string)
	go func() {
		// construct url for this job
		assignedNum := atomic.LoadUint64(&nextNum)
		atomic.AddUint64(&nextNum, 1)
		url := *baseUrlPtr + strconv.Itoa(int(assignedNum))

		fmt.Printf("at '%v'", url)
	}()

	fmt.Printf("nextNum='%v'", nextNum)
}
