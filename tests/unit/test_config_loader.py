"""Unit tests for configuration loading helpers."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from monsoonbench.config.loader import get_config, load_config


def _write_yaml(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).strip() + "\n")


def test_load_config_reads_yaml(tmp_path: Path):
    """load_config should read YAML files into dictionaries."""
    config_file = tmp_path / "config.yaml"
    _write_yaml(
        config_file,
        """
        model_type: climatology
        years: [2020, 2021]
        """,
    )

    cfg = load_config(config_file)
    assert cfg["model_type"] == "climatology"
    assert cfg["years"] == [2020, 2021]


def test_get_config_with_minimal_cli_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """get_config should parse CLI arguments and apply defaults."""
    imd = tmp_path / "imd"
    imd.mkdir()
    thres = tmp_path / "threshold.nc"
    thres.write_text("data")
    shp = tmp_path / "india.shp"
    shp.write_text("data")

    argv = [
        "monsoonbench",
        "--model_type",
        "climatology",
        "--years",
        "2020",
        "2021",
        "--imd_folder",
        str(imd),
        "--thres_file",
        str(thres),
        "--shpfile_path",
        str(shp),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cfg = get_config()

    assert cfg.model_type == "climatology"
    assert cfg.years == [2020, 2021]
    assert cfg.download_dir is None
    assert cfg.download_formats is None
    assert cfg.download_keep_nans is False


def test_get_config_download_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """YAML values should be merged with CLI download overrides."""
    imd = tmp_path / "imd"
    imd.mkdir()
    thres = tmp_path / "threshold.nc"
    thres.write_text("data")
    shp = tmp_path / "india.shp"
    shp.write_text("data")
    forecasts = tmp_path / "forecasts"
    forecasts.mkdir()

    config_file = tmp_path / "config.yaml"
    _write_yaml(
        config_file,
        f"""
        model_type: deterministic
        years: [2019]
        imd_folder: {imd}
        thres_file: {thres}
        shpfile_path: {shp}
        model_forecast_dir: {forecasts}
        download_dir: default_artifacts
        download_formats: ["csv"]
        """,
    )

    argv = [
        "monsoonbench",
        "--config",
        str(config_file),
        "--download_dir",
        "cli_artifacts",
        "--download_formats",
        "csv",
        "json",
        "--download_metrics",
        "mean_mae",
        "miss_rate",
        "--download_keep_nans",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cfg = get_config()

    assert cfg.model_type == "deterministic"
    assert cfg.download_dir == "cli_artifacts"
    assert cfg.download_formats == ["csv", "json"]
    assert cfg.download_metrics == ["mean_mae", "miss_rate"]
    assert cfg.download_keep_nans is True
