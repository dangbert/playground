//usr/bin/env go run "$0" "$@"; exit
// USAGE: ./scrape.go -baseUrl=example.com/blog/ -start=1 -end=5

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	baseUrlPtr := flag.String("baseUrl", "", "url to scrape")
	startPtr := flag.Int("start", 1, "start num to append to url")
	endPtr := flag.Int("end", -1, "final num to append to url (incremented sequentially)")
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

	fmt.Printf("scraping '%v%v' -> '%v%v'\n", *baseUrlPtr, *startPtr, *baseUrlPtr, *startPtr)
}
