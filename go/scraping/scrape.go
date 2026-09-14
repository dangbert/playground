//usr/bin/env go run "$0" "$@"; exit
// USAGE: ./scrape.go -baseUrl=example.com/blog/ -start=1 -end=5 -j10

package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

type payload struct {
	url      string
	finalUrl string // after possible redirection

	title    string
	httpCode int
}

var (
	lastScrapeMu sync.Mutex
	lastScrape   time.Time
)

func rateLimit(rpm int) {
	interval := time.Duration(float64(time.Minute) / float64(rpm))
	lastScrapeMu.Lock()
	defer lastScrapeMu.Unlock()

	now := time.Now()
	elapsed := now.Sub(lastScrape)
	if elapsed < interval {
		time.Sleep(interval - elapsed)
		lastScrape = time.Now()
	} else {
		lastScrape = time.Now()
	}
}

func main() {
	baseUrlPtr := flag.String("baseUrl", "", "url to scrape")
	startPtr := flag.Int("start", 1, "start num to append to url")
	endPtr := flag.Int("end", -1, "final num to append to url (incremented sequentially)")
	//sleepPtr := flag.Float("end", 0.25, "time to sleep between scrapes")
	jPtr := flag.Int("j", 4, "max concurrent threads")
	rpmPtr := flag.Int("rpm", 120, "requests per minute")
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
	//if !strings.HasSuffix(*baseUrlPtr, "/") {
	//	*baseUrlPtr = *baseUrlPtr + "/"
	//}

	//fmt.Printf("scraping '%v%v' -> '%v%v' (%v threads)\n", *baseUrlPtr, *startPtr, *baseUrlPtr, *endPtr, *jPtr)

	// https://medium.com/hprog99/concurrency-in-go-a-deep-dive-2abbb4838984

	// channel for storing tasks
	tasks := make(chan payload, *jPtr)
	results := make(chan payload, *jPtr)
	// create desired workforce
	var wg sync.WaitGroup
	for j := 0; j < *jPtr; j++ {
		wg.Add(1)
		go worker(j, tasks, results, &wg, *rpmPtr)
	}

	// send tasks in a goroutine so we can start processing results immediately
	go func() {
		for curNum := *startPtr; curNum <= *endPtr; curNum++ {
			url := *baseUrlPtr + strconv.Itoa(curNum)
			tasks <- payload{url: url}
		}
		close(tasks)
	}()

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	fmt.Printf("awaiting results...\n")
	for x := range results {
		fmt.Printf("%v\n", x)
	}
}

// scrape a set of assigned urls
func worker(id int, tasks chan payload, results chan payload, wg *sync.WaitGroup, rpm int) {
	defer wg.Done()
	for item := range tasks {
		url := item.url
		fmt.Printf("worker %d at %v\n", id, url)
		results <- scrapePage(url, rpm)
	}
}

func scrapePage(url string, rpm int) payload {
	//fmt.Printf("\tscraping '%v'\n", url)

	bad := payload{
		url:      url,
		finalUrl: url,
		title:    "",
		httpCode: -1,
	}

	rateLimit(rpm)

	res, err := http.Get(url)
	time.Sleep(250 * time.Millisecond)

	if err != nil {
		fmt.Printf("error1: %s\n", err)
		return bad
	}
	defer res.Body.Close()

	body, err := io.ReadAll(res.Body)
	if err != nil {
		fmt.Printf("error2: %s\n", err)
		return bad
	}
	re := regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	match := re.FindStringSubmatch(string(body))
	title := ""
	if len(match) > 1 {
		title = strings.TrimSpace(match[1])
	}

	return payload{
		url:      url,
		finalUrl: res.Request.URL.String(),
		title:    title,
		httpCode: res.StatusCode,
	}
}
