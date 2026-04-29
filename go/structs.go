//usr/bin/env go run "$0" "$@"; exit
/********************************************************************************
* practicing go structs
* next steps:
* https://www.geeksforgeeks.org/go-language/class-and-object-in-golang/
********************************************************************************/

package main

import (
	"fmt"
	"time"
)

type route struct {
	name  string
	grade int
}

type history struct {
	routes []route
	date   time.Time
}

func main() {
	fmt.Println("what's up")

	x := route{name: "El guincho liso", grade: 6}
	fmt.Println(x)

	y := route{}
	fmt.Println(y.grade) // 0

	journal := history{routes: []route{route{name: "pindakaas"}}, date: time.Now()}
	fmt.Println("journal:")
	fmt.Println(journal.routes)
	fmt.Println(journal.date)
}
