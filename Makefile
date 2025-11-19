# --- Basic paths ---
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
project_dir := $(subst Makefile,, $(mkfile_path))

# --- Docker config ---
service := monsoonbench    # name of service from docker-compose.yaml
mount_data := -v $(project_dir)/monsoonbench/data:/project/monsoonbench/data

.PHONY: build shell test run-notebooks clean

# Build the MonsoonBench Docker image
build:
	docker compose build

# Open an interactive shell inside the container
shell: build
	docker compose run --rm $(mount_data) $(service) bash

# Run tests (pytest is inside the repo)
test: build
	docker compose run --rm $(mount_data) $(service) pytest -q

# Run demo notebooks
run-notebooks: build
	docker compose run --rm -p 8888:8888 $(mount_data) $(service) \
		uv run jupyter lab --port=8888 --ip='*' --NotebookApp.token='' --no-browser

clean:
	docker compose down --rmi all --volumes --remove-orphans