package main

import (
	"context"
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func main() {
	log.Printf("peparing to run()")
	if err := run(); err != nil {
		log.Fatalln(err)
	}
}

func run() (err error) {
	// gracefully handle SIGINT
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// setup OpenTelemetry
	otelShutdown, err := setupOtelSDK(ctx)
	if err != nil {
		return err
	}
	// proper shutdown
	defer func() {
		log.Printf("ready to shutdown")
		err = errors.Join(err, otelShutdown(context.Background()))
	}()

	port := 8080
	srv := &http.Server{
		Addr:         ":" + strconv.Itoa(port),
		BaseContext:  func(net.Listener) context.Context { return ctx },
		ReadTimeout:  time.Second,
		WriteTimeout: 10 * time.Second,
		Handler:      newHTTPHandler(),
	}
	srvErr := make(chan error, 1)
	go func() {
		log.Printf("Running HTTP server on port %v\n", port)
		srvErr <- srv.ListenAndServe()
	}()

	// Wait for interruption.
	select {
	case err = <-srvErr:
		// Error when starting HTTP server.
		return err
	case <-ctx.Done():
		// Wait for first CTRL+C.
		// Stop receiving signal notifications as soon as possible.
		stop()
	}

	// When Shutdown is called, ListenAndServe immediately returns ErrServerClosed.
	err = srv.Shutdown(context.Background())
	return err
}

func newHTTPHandler() http.Handler {
	mux := http.NewServeMux()

	// Register handlers.
	mux.HandleFunc("/rolldice/", rolldice)
	mux.HandleFunc("/rolldice/{player}", rolldice)

	// add http instrumentation
	handler := otelhttp.NewHandler(mux, "/")
	return handler
}
