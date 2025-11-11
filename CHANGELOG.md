# Changelog

All notable changes to MonsoonBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure with `src/monsoonbench/` layout
- Module organization: data, metrics, onset, visualization, config, spatial, cli, utils
- Test framework structure (unit, integration, fixtures)
- Examples directory with configs, scripts, and notebooks
- Documentation structure

### Changed
- Migrating from standalone scripts in `reference_scripts/` to organized package
- Consolidating code from `consolidate_onset_metrics/` into modular package structure

## [0.1.0] - TBD

### Added
- First release (in development)
- Basic onset metrics computation (MAE, FAR, Miss Rate)
- Support for deterministic, probabilistic, and climatology forecasts
- YAML-based configuration system
- CLI interface via `monsoonbench` command
