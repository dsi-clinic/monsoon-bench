# This is a basic docker image for use in the clinic
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Switch to root to update and install tools
RUN apt-get update && apt-get install -y curl git cdo
# Create working directory
WORKDIR /project

COPY pyproject.toml .
COPY README.md .
COPY monsoonbench/ ./monsoonbench/

# Resolve and install Python packages from pyproject/uv.lock
RUN /usr/local/bin/uv venv
ENV VIRTUAL_ENV=/project/.venv
ENV PATH="/project/.venv/bin:$PATH"
ENV PYTHONPATH=/project
RUN uv sync --extra netcdf4

CMD ["/bin/bash"]