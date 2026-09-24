// https://opentelemetry.io/docs/languages/go/instrumentation/

package main

import (
	"context"
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutlog"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutmetric"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/propagation"
	olog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/trace"
)

// where to write telemetry. set to "" to use stdout instead
const outputDir = "./output"

// bootstrap OpenTelemetry pipeline
func setupOtelSDK(ctx context.Context) (func(context.Context) error, error) {
	var shutdownFuncs []func(context.Context) error
	var err error

	if outputDir != "" {
		if err = os.MkdirAll(outputDir, 0755); err != nil {
			return nil, err
		}
	}

	// caller of all registered cleanup funcs
	shutdown := func(ctx context.Context) error {
		log.Printf("Shutting down otel")
		var err error
		for _, fn := range shutdownFuncs {
			err = errors.Join(err, fn(ctx))
		}
		shutdownFuncs = nil
		return err
	}

	handleErr := func(inErr error) {
		err = errors.Join(inErr, shutdown(ctx))
	}

	// propagator
	prop := newPropagator()
	otel.SetTextMapPropagator(prop)

	// tracer
	tracerProvider, fns, err := newSignal("traces.json", newTracerProvider)
	if err != nil {
		handleErr(err)
		return shutdown, err
	}
	shutdownFuncs = append(shutdownFuncs, fns...)
	otel.SetTracerProvider(tracerProvider)

	// meter
	meterProvider, fns, err := newSignal("metrics.json", newMeterProvider)
	if err != nil {
		handleErr(err)
		return shutdown, err
	}
	shutdownFuncs = append(shutdownFuncs, fns...)
	otel.SetMeterProvider(meterProvider)

	// logger provider
	loggerProvider, fns, err := newSignal("logs.json", newLoggerProvider)
	if err != nil {
		handleErr(err)
		return shutdown, err
	}
	shutdownFuncs = append(shutdownFuncs, fns...)
	global.SetLoggerProvider(loggerProvider)

	return shutdown, err
}

type provider interface {
	Shutdown(context.Context) error
}

// pairs a provider with the sink it writes to, returning cleanups ordered so
// the provider flushes before the sink closes
func newSignal[P provider](name string, newProvider func(io.Writer) (P, error)) (P, []func(context.Context) error, error) {
	var zero P

	if outputDir == "" {
		p, err := newProvider(os.Stdout)
		if err != nil {
			return zero, nil, err
		}
		return p, []func(context.Context) error{p.Shutdown}, nil
	}

	f, err := os.OpenFile(filepath.Join(outputDir, name), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return zero, nil, err
	}

	p, err := newProvider(f)
	if err != nil {
		return zero, nil, errors.Join(err, f.Close())
	}

	return p, []func(context.Context) error{
		p.Shutdown,
		func(context.Context) error { return f.Close() },
	}, nil
}

func newPropagator() propagation.TextMapPropagator {
	return propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	)
}

func newTracerProvider(w io.Writer) (*trace.TracerProvider, error) {
	traceExporter, err := stdouttrace.New(
		stdouttrace.WithWriter(w),
		stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, err
	}

	tracerProvider := trace.NewTracerProvider(
		trace.WithBatcher(traceExporter,
			// Default is 5s. Set to 1s for demonstrative purposes.
			trace.WithBatchTimeout(time.Second)),
	)
	return tracerProvider, nil
}

func newMeterProvider(w io.Writer) (*metric.MeterProvider, error) {
	metricExporter, err := stdoutmetric.New(
		stdoutmetric.WithWriter(w),
		stdoutmetric.WithPrettyPrint())
	if err != nil {
		return nil, err
	}

	// regularly export regisered metrics
	meterProvider := metric.NewMeterProvider(
		metric.WithReader(metric.NewPeriodicReader(metricExporter,
			// Default is 1m.
			metric.WithInterval(15*time.Second))),
	)
	return meterProvider, nil
}

func newLoggerProvider(w io.Writer) (*olog.LoggerProvider, error) {
	logExporter, err := stdoutlog.New(
		stdoutlog.WithWriter(w),
		stdoutlog.WithPrettyPrint())
	if err != nil {
		return nil, err
	}

	loggerProvider := olog.NewLoggerProvider(
		olog.WithProcessor(olog.NewBatchProcessor(logExporter)),
	)
	return loggerProvider, nil
}
