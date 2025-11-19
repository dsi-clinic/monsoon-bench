# Test Suite Overview

The MonsoonBench test suite is split into unit and integration layers to cover
both isolated helpers and the end-to-end CLI workflow. Pytest discovers tests
under `tests/unit` and `tests/integration`.

## Unit Tests

Unit coverage focuses on public APIs and utility modules without hitting the
filesystem or network:

- `test_cli_main.py` validates imports, method signatures, and class structure
  for the CLI and metrics modules.
- `test_config_loader.py` exercises `load_config` and `get_config`, checking
  YAML parsing, CLI overrides, and default/download flag handling.
- `test_visualization_data_downloader.py` covers the downloader API by checking
  metadata inference, selective exports, and multi-format persistence logic.

These tests rely on light-weight fixtures with synthetic data to keep runtime
low while still verifying behavior.

## Integration Tests

`tests/integration/test_cli_integration.py` mocks the heavy dependencies
(`DataLoader`, metrics classes, plotting) but executes the `monsoonbench.cli.main`
entry point across the key scenarios:

- Deterministic, probabilistic, and climatology workflows.
- Tolerance-day calculations for different forecast windows.
- NetCDF output attributes, plot generation, and the new visualization
  downloader (including default vs. user-specified formats/metrics).

`test_probabilistic_climatology.py` adds additional probabilistic/climatology
logic checks for other subsystems.

Together these tests ensure the CLI operates correctly when users vary config
parameters without requiring the full dataset.
