
package main

import (
	"os"
	"log"
	"fmt"
)

func main() {
	fname := "MUSIC_converted_to_list.txt"

	file, err := os.Open(fname)
	if (err != nil) {
		log.Fatal(fmt.Sprintf("file not found '%v'", fname))
	}

	// TODO split on newline...
	data := make([]byte, 100)
	count, err := file.Read(data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("read %d bytes: %q\n", count, data[:count])
}
