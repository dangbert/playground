## Go

Here we follow the intro tutorials for the Go programming language and then do some free styling to combine the two.

* [docs: Get started with Go](https://go.dev/doc/tutorial/getting-started) [./hello](./hello)

* [docs: create module](https://go.dev/doc/tutorial/create-module) [./greetings](./greetings)


Next steps:
* [Go Tutorials list](https://go.dev/doc/tutorial/)
* [A Tour of Go (interactive)](https://go.dev/tour)

### How To

````bash
# run the code
cd hello/
go run hello.go
# or equivalently
go run .

# install any packages referenced (in go.mod)
go mody tidy

# verify hashes of dependencies vs go.sum
go mod verify

# note that go.mod was created initially with
go mod init example/hello


# compile the code
go build
# run the executable
./hello

# optionally you can install the executable
go list -f '{{.Target}}' # print install dir
go install               # install
````


````bash
# run tests
cd greetings/

go test

# OR
go test -v
````

