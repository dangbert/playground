## OpenTelemetry (OTEL) Demo

* https://opentelemetry.io/docs/languages/go/getting-started/

````bash
go mod tidy

# https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/#general-sdk-configuration
export OTEL_RESOURCE_ATTRIBUTES="service.name=dice,service.version=0.1.0"
#rm -rf output/ # optionally wipe
make run

curl http://localhost:8080/rolldice
curl http://localhost:8080/rolldice/dan
````


see also:
* [./example_output/](./example_output/)
* https://opentelemetry.io/docs/demo/ (more extensive demo)
