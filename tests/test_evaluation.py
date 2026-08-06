"""Tests for experiment scoring, HDF5 reading, and CSV output."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from mcdd.experiments import (
    ExperimentConfiguration,
    get_configuration,
    read_stream,
    run_archive_experiment,
    summarize_archive_results,
)
from mcdd.experiments.evaluation import _score_first_alarm


@pytest.mark.parametrize(
    ("alarm_index", "expected_outcome"),
    [
        (None, "missed"),
        (100, "false_alarm"),
        (101, "detected"),
        (119, "detected"),
        (120, "false_alarm"),
        (90, "false_alarm"),
        (130, "false_alarm"),
    ],
)
def test_alarm_scoring_uses_strict_interval_limits(
    alarm_index: int | None,
    expected_outcome: str,
) -> None:
    """Only start < detection < end should count as a valid detection."""
    outcome = _score_first_alarm(
        alarm_index=alarm_index,
        drift_start=100,
        valid_detection_end=120,
    )

    assert outcome == expected_outcome


def test_get_configuration_uses_the_public_configuration_name() -> None:
    """A configuration should be selected by its exact published abbreviation."""
    configuration = get_configuration("MCDD-G20kT")

    assert configuration.name == "MCDD-G20kT"
    assert configuration.method == "mcdd"
    assert configuration.window_mode == "growing_fixed"
    assert configuration.max_window_size == 20_000

    with pytest.raises(ValueError, match="Unknown configuration"):
        get_configuration("UNKNOWN")


def _create_test_archive(path: Path, number_of_streams: int = 3) -> None:
    """Create a small abrupt-normal HDF5 archive for integration tests."""
    base_stream = np.concatenate(
        (
            np.zeros(20, dtype=np.float64),
            np.ones(20, dtype=np.float64),
        )
    )
    values = np.stack(
        [base_stream + index * 0.01 for index in range(number_of_streams)]
    )

    with h5py.File(path, "w") as archive:
        archive.attrs["drift_type"] = "abrupt"
        archive.attrs["distribution"] = "normal"
        archive.create_dataset("values", data=values)
        archive.create_dataset(
            "seeds",
            data=np.arange(42, 42 + number_of_streams, dtype=np.int64),
        )
        archive.create_dataset(
            "drift_start",
            data=np.full(number_of_streams, 20, dtype=np.int64),
        )
        archive.create_dataset(
            "drift_end",
            data=np.full(number_of_streams, 20, dtype=np.int64),
        )


def test_read_stream_returns_values_and_metadata(tmp_path: Path) -> None:
    """One stored row should be recoverable without loading the full archive."""
    archive_path = tmp_path / "abrupt_normal.h5"
    _create_test_archive(archive_path, number_of_streams=2)

    values, metadata = read_stream(archive_path, row_index=1)

    assert values.shape == (40,)
    assert metadata == {
        "drift_type": "abrupt",
        "distribution": "normal",
        "row_index": 1,
        "seed": 43,
        "drift_start": 20,
        "drift_end": 20,
    }


def test_selected_archive_experiment_creates_per_run_and_summary_csv(
    tmp_path: Path,
) -> None:
    """A reduced experiment should produce auditable per-run and summary files."""
    archive_path = tmp_path / "abrupt_normal.h5"
    per_run_path = tmp_path / "per_run.csv"
    summary_path = tmp_path / "summary.csv"
    _create_test_archive(archive_path, number_of_streams=3)

    configuration = ExperimentConfiguration(
        name="TSH-TEST",
        method="tsh",
        window_mode="sliding",
        window_size=20,
        n_subwindows=2,
        alpha=0.01,
        description="Small integration-test configuration.",
    )

    run_archive_experiment(
        archive_path=archive_path,
        configuration=configuration,
        output_file=per_run_path,
        max_streams=2,
        overwrite=False,
        progress_every=0,
    )
    summary = summarize_archive_results(
        per_run_file=per_run_path,
        output_file=summary_path,
    )

    per_run = pd.read_csv(per_run_path)

    assert per_run_path.is_file()
    assert summary_path.is_file()
    assert len(per_run) == 2
    assert per_run["configuration"].tolist() == ["TSH-TEST", "TSH-TEST"]
    assert per_run["row_index"].tolist() == [0, 1]

    assert len(summary) == 1
    assert summary.loc[0, "configuration"] == "TSH-TEST"
    assert summary.loc[0, "drift_type"] == "abrupt"
    assert summary.loc[0, "distribution"] == "normal"
    assert int(summary.loc[0, "replications"]) == 2
    assert (
        int(summary.loc[0, "TP"])
        + int(summary.loc[0, "FP"])
        + int(summary.loc[0, "FN"])
        == 2
    )


def test_archive_experiment_does_not_overwrite_by_default(
    tmp_path: Path,
) -> None:
    """Existing result files should be protected unless overwrite is explicit."""
    archive_path = tmp_path / "abrupt_normal.h5"
    output_path = tmp_path / "per_run.csv"
    _create_test_archive(archive_path, number_of_streams=1)
    output_path.write_text("existing", encoding="utf-8")

    configuration = ExperimentConfiguration(
        name="TSH-TEST",
        method="tsh",
        window_size=20,
        n_subwindows=2,
    )

    with pytest.raises(FileExistsError):
        run_archive_experiment(
            archive_path=archive_path,
            configuration=configuration,
            output_file=output_path,
            max_streams=1,
        )
