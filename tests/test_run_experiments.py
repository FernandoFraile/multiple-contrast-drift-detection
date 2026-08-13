"""Tests for the experiment runner command-line distribution selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_experiments_script", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load experiment runner from {SCRIPT_PATH}.")
RUN_EXPERIMENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_EXPERIMENTS)


def _selected_distributions(monkeypatch, *arguments: str) -> tuple[str, ...]:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiments.py", *arguments],
    )
    parsed = RUN_EXPERIMENTS.parse_arguments()
    return RUN_EXPERIMENTS._selected_values(
        parsed.distribution,
        RUN_EXPERIMENTS.DEFAULT_DISTRIBUTIONS,
    )


def test_normal_distribution_is_used_by_default(monkeypatch) -> None:
    assert _selected_distributions(monkeypatch) == ("normal",)


def test_exponential_distribution_can_be_selected_explicitly(monkeypatch) -> None:
    assert _selected_distributions(
        monkeypatch,
        "--distribution",
        "exponential",
    ) == ("exponential",)


def test_gamma_distribution_can_be_selected_explicitly(monkeypatch) -> None:
    assert _selected_distributions(
        monkeypatch,
        "--distribution",
        "gamma",
    ) == ("gamma",)


def test_multiple_distributions_can_be_selected_explicitly(monkeypatch) -> None:
    assert _selected_distributions(
        monkeypatch,
        "--distribution",
        "normal",
        "--distribution",
        "gamma",
    ) == ("normal", "gamma")
