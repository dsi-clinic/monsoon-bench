# MonsoonBench Docker workflow helper.
#
# This Makefile assumes:
# - docker-compose.yaml defines a service named "monsoonbench"
# - Raw data is mounted via docker-compose volumes (not via `docker run -v ...`)
# - You can optionally set HOST_RAW_DATA to point to an external raw data root.
#
# Examples:
#   make build
#   make shell
#   make test
#   make lab
#
# External data:
#   make lab HOST_RAW_DATA=/absolute/path/to/monsoonbench_raw

# --- Docker config ---
SERVICE := monsoonbench

# Allow overriding HOST_RAW_DATA on the command line:
#   make lab HOST_RAW_DATA=/path/to/raw
HOST_RAW_DATA ?=

.PHONY: build shell test lab up down clean

## build: Build the Docker image for the monsoonbench service.
build:
	@echo "Building Docker image..."
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose build

## shell: Open an interactive shell inside the container.
shell: build
	@echo "Starting interactive shell..."
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose run --rm $(SERVICE) bash

## test: Run pytest inside the container.
test: build
	@echo "Running tests..."
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose run --rm $(SERVICE) pytest -q

## lab: Run Jupyter Lab inside the container on port 8888.
lab: build
	@echo "Starting Jupyter Lab at http://localhost:8888 ..."
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose run --rm -p 8888:8888 $(SERVICE) \
		bash -lc "uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --allow-root"

## up: Bring up the service (useful if you later add a long-running command in compose).
up: build
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose up

## down: Stop services started with `make up`.
down:
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose down

## clean: Remove containers, images, volumes created by compose.
clean:
	@HOST_RAW_DATA="$(HOST_RAW_DATA)" docker compose down --rmi all --volumes --remove-orphans
