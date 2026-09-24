package main

import (
	"io"
	"log"
	"math/rand"
	"net/http"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"

	"go.opentelemetry.io/contrib/bridges/otelslog"
)

const name = "go.opentelemetry.io/contrib/examples/dice"
const enabled = true // toggle to distinguiosh automatical telemetry from custrom

var (
	tracer    = otel.Tracer(name)
	meter     = otel.Meter(name)
	logger    = otelslog.NewLogger(name)
	rollCount metric.Int64Counter
)

// called automatically by Go before main()
func init() {
	log.Printf("initalizing dice")
	var err error
	rollCount, err = meter.Int64Counter("dice.rolls",
		metric.WithDescription("num rolls by roll value"),
		metric.WithUnit("{roll}"))
	if err != nil {
		panic(err)
	}
}

// example function to do some logged work
func rolldice(w http.ResponseWriter, r *http.Request) {
	var msg string
	player := r.PathValue("player")

	if player != "" {
		msg = player + " rolling the dice"
	} else {
		msg = "anon player rolling the dice"
	}

	roll := 1 + rand.Intn(6) // rand int in range [1,6]

	if enabled {
		// register a span
		ctx, span := tracer.Start(r.Context(), "roll")
		defer span.End() // indicate Span has completed

		logger.InfoContext(ctx, msg, "result", roll)
		rollValueAttr := attribute.Int("roll.value", roll)
		span.SetAttributes(rollValueAttr)
		// increment count of this particular dice result (e.g. we've rolled 3 ten times so far)
		rollCount.Add(ctx, 1, metric.WithAttributes(rollValueAttr))
	} else {
		log.Println(msg)
	}

	resp := strconv.Itoa(roll) + "\n"
	if _, err := io.WriteString(w, resp); err != nil {
		log.Printf("Write failed %v", err)
	}
}
