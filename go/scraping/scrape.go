//usr/bin/env go run "$0" "$@"; exit
// USAGE: ./scrape.go -baseUrl=example.com/blog/ -start=1 -end=5 -j10

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

type payload struct {
	url      string
	finalUrl string // after possible redirection

	title    string
	httpCode int
}

func main() {
	baseUrlPtr := flag.String("baseUrl", "", "url to scrape")
	startPtr := flag.Int("start", 1, "start num to append to url")
	endPtr := flag.Int("end", -1, "final num to append to url (incremented sequentially)")
	jPtr := flag.Int("j", 4, "max concurrent threads")
	flag.Parse()

	// parse args
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

	fmt.Printf("scraping '%v%v' -> '%v%v' (%v threads)\n", *baseUrlPtr, *startPtr, *baseUrlPtr, *endPtr, *jPtr)

	// https://medium.com/hprog99/concurrency-in-go-a-deep-dive-2abbb4838984

	// channel for storing tasks (to be updated in place with results)
	tasks := make(chan payload, *endPtr-*startPtr+1)
	// create desired workforce
	var wg sync.WaitGroup
	for j := 0; j < *jPtr; j++ {
		wg.Add(1)
		go worker(j, tasks, &wg)
	}

	// send tasks
	for curNum := *startPtr; curNum <= *endPtr; curNum++ {
		url := *baseUrlPtr + strconv.Itoa(curNum)
		tasks <- payload{url: url}
	}

	fmt.Printf("awaiting results...\n")
	wg.Wait()
	close(tasks)

	fmt.Printf("\nresults:\n")
	for t := range tasks {
		fmt.Printf("%v\n", t)
	}
}

// scrape a set of pages?
// results are sent back into the channel (annotated as bidirectional here)
func worker(id int, item chan payload, wg *sync.WaitGroup) {
	defer wg.Done()
	url := (<-item).url
	fmt.Printf("worker %d at %v\n", id, url)
	item <- scrapePage(url) // update Payload in place with result
}

func scrapePage(url string) payload {
	fmt.Printf("\tscraping '%v'\n", url)

	return payload{
		url:      url,
		finalUrl: url,
		title:    "dummy result",
		httpCode: -1,
	}
}
