#!/bin/bash
# validate the codebase with unit tests, linting etc

set -e

SCRIPT_DIR="$(realpath "$(dirname "$0")")"

function main() {
  cd "$SCRIPT_DIR"

  echo -e "\nformatting code"
  ruff format .

  echo -e "\nrunning ruff --fix"
  ruff check --fix .

  echo -e "\nrunning unit tests..."
  pytest -v

  echo -e "\n\nrunning mypy..."
  mypy .
}

main "$@"
