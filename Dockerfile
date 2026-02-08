# MonsoonBench base image (code + dependencies only).
#
# Key principle:
# - Raw data MUST NOT be baked into the image.
#   Raw data should be mounted at runtime via docker-compose volumes.
#
# IMPORTANT:
# - You should also add a .dockerignore to exclude large raw data folders from the build context,
#   otherwise builds can be extremely slow because Docker must send the context to the daemon.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# --- System dependencies ---
# Keep this minimal; add OS packages only when required by your Python deps or tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    cdo \
  && rm -rf /var/lib/apt/lists/*

# --- Workspace ---
WORKDIR /project

# --- Copy only what is needed to resolve dependencies + run the package ---
# Copy pyproject first to maximize layer caching.
COPY pyproject.toml .
# README is optional but harmless; also helps some tooling.
COPY README.md .

# Copy package code.
# NOTE: If your repo contains large raw data under monsoonbench/data/*, you MUST exclude it via .dockerignore.
COPY monsoonbench/ ./monsoonbench/

# --- Python environment via uv ---
# Create venv OUTSIDE /project so bind-mount won't overwrite it
RUN /usr/local/bin/uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/project

# Install deps into /opt/venv
RUN uv sync --extra netcdf4 --active
RUN uv pip install --no-cache-dir jupyterlab ipykernel

# Default to an interactive shell (compose may override).
CMD ["/bin/bash"]
