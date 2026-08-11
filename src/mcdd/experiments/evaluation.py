"""Reusable experiment evaluation for the MCDD benchmark."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from mcdd.detectors import LORDLocalDependence, MCDD

Method: TypeAlias = Literal["mcdd", "tsh", "kswin", "lord"]
WindowMode: TypeAlias = Literal[
    "sliding",
    "growing_dynamic",
    "growing_fixed",
]
Outcome: TypeAlias = Literal[
    "detected",
    "false_alarm",
    "late_detection",
    "missed",
]


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    """One detector configuration evaluated in the paper experiments."""

    name: str
    method: Method
    window_mode: WindowMode = "sliding"
    window_size: int = 6_000
    min_window_size: int | None = None
    max_window_size: int | None = None
    n_subwindows: int = 10
    alpha: float = 0.01
    correction: str | None = None
    min_rejections: int = 1
    description: str = ""


PAPER_CONFIGURATIONS: tuple[ExperimentConfiguration, ...] = (
    ExperimentConfiguration(
        name="MCDD-S",
        method="mcdd",
        window_mode="sliding",
        correction="fdr_by",
        description="MCDD with a fixed 6,000-sample sliding window.",
    ),
    ExperimentConfiguration(
        name="MCDD-G20k",
        method="mcdd",
        window_mode="growing_dynamic",
        min_window_size=6_000,
        max_window_size=20_000,
        correction="fdr_by",
        description=(
            "MCDD growing to 20,000 samples while keeping ten subwindows."
        ),
    ),
    ExperimentConfiguration(
        name="MCDD-G30k",
        method="mcdd",
        window_mode="growing_dynamic",
        min_window_size=6_000,
        max_window_size=30_000,
        correction="fdr_by",
        description=(
            "MCDD growing to 30,000 samples while keeping ten subwindows."
        ),
    ),
    ExperimentConfiguration(
        name="MCDD-G20kT",
        method="mcdd",
        window_mode="growing_fixed",
        min_window_size=6_000,
        max_window_size=20_000,
        correction="fdr_by",
        description=(
            "MCDD growing to 20,000 samples with fixed 600-sample "
            "subwindows and an increasing number of contrasts."
        ),
    ),
    ExperimentConfiguration(
        name="MCDD-G30kT",
        method="mcdd",
        window_mode="growing_fixed",
        min_window_size=6_000,
        max_window_size=30_000,
        correction="fdr_by",
        description=(
            "MCDD growing to 30,000 samples with fixed 600-sample "
            "subwindows and an increasing number of contrasts."
        ),
    ),
    ExperimentConfiguration(
        name="TSH-S",
        method="tsh",
        window_mode="sliding",
        description=(
            "Traditional single KS test with a 6,000-sample sliding window."
        ),
    ),
    ExperimentConfiguration(
        name="TSH-G20k",
        method="tsh",
        window_mode="growing_dynamic",
        min_window_size=6_000,
        max_window_size=20_000,
        description=(
            "Traditional single KS test with a window growing to 20,000."
        ),
    ),
    ExperimentConfiguration(
        name="TSH-G30k",
        method="tsh",
        window_mode="growing_dynamic",
        min_window_size=6_000,
        max_window_size=30_000,
        description=(
            "Traditional single KS test with a window growing to 30,000."
        ),
    ),
    ExperimentConfiguration(
        name="KSWIN",
        method="kswin",
        window_mode="sliding",
        description=(
            "River KSWIN with window_size=6,000 and stat_size=600."
        ),
    ),
    ExperimentConfiguration(
        name="LORD-LD",
        method="lord",
        window_mode="sliding",
        description=(
            "LORD under local dependence using 6,000-sample windows."
        ),
    ),
)

DRIFT_TYPES = ("abrupt", "gradual", "incremental")
DISTRIBUTIONS = ("normal", "exponential", "gamma")

PER_RUN_FIELDS = (
    "configuration",
    "method",
    "window_mode",
    "window_size",
    "min_window_size",
    "max_window_size",
    "n_subwindows",
    "alpha",
    "correction",
    "min_rejections",
    "drift_type",
    "distribution",
    "row_index",
    "seed",
    "alarm_index",
    "outcome",
    "drift_start",
    "valid_detection_end",
    "delay",
    "late_delay",
    "TP",
    "FP",
    "FN",
)


def configuration_table(
    configurations: Sequence[ExperimentConfiguration] = PAPER_CONFIGURATIONS,
) -> pd.DataFrame:
    """Return the experiment configurations as a DataFrame."""
    return pd.DataFrame([asdict(configuration) for configuration in configurations])


def expected_dataset_paths(data_directory: str | Path) -> list[Path]:
    """Return and validate the nine HDF5 paths used in the experiments."""
    directory = Path(data_directory)
    paths = [
        directory / f"{drift_type}_{distribution}.h5"
        for drift_type in DRIFT_TYPES
        for distribution in DISTRIBUTIONS
    ]
    missing = [path for path in paths if not path.is_file()]

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The following dataset archives are missing:\n" + missing_text
        )

    return paths


def read_stream(
    archive_path: str | Path,
    row_index: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Read one stream and its drift metadata from an HDF5 archive."""
    path = Path(archive_path)

    with h5py.File(path, "r") as archive:
        number_of_streams = int(archive["values"].shape[0])
        if not 0 <= row_index < number_of_streams:
            raise IndexError(
                f"row_index must be in [0, {number_of_streams - 1}]."
            )

        values = np.asarray(
            archive["values"][row_index],
            dtype=np.float64,
        )
        metadata = {
            "drift_type": _decode_attribute(archive.attrs["drift_type"]),
            "distribution": _decode_attribute(archive.attrs["distribution"]),
            "row_index": int(row_index),
            "seed": int(archive["seeds"][row_index]),
            "drift_start": int(archive["drift_start"][row_index]),
            "drift_end": int(archive["drift_end"][row_index]),
        }

    return values, metadata


def evaluate_single_stream(
    values: Sequence[float],
    *,
    drift_type: str,
    distribution: str,
    row_index: int,
    seed: int,
    drift_start: int,
    drift_end: int,
    configuration: ExperimentConfiguration,
) -> dict[str, Any]:
    """Evaluate one detector configuration on one stream."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)

    if configuration.method == "mcdd":
        alarm_index = _first_mcdd_alarm(data, configuration)
    elif configuration.method == "tsh":
        alarm_index = _first_tsh_alarm(data, configuration)
    elif configuration.method == "kswin":
        alarm_index = _first_kswin_alarm(data, configuration)
    elif configuration.method == "lord":
        alarm_index = _first_lord_alarm(data, configuration)
    else:
        raise ValueError(f"Unsupported method: {configuration.method}.")

    valid_detection_end = (
        drift_start + 2_000 if drift_type == "abrupt" else drift_end
    )
    outcome = _score_first_alarm(
        alarm_index=alarm_index,
        drift_start=drift_start,
        valid_detection_end=valid_detection_end,
    )

    delay = (
        alarm_index - drift_start
        if outcome == "detected" and alarm_index is not None
        else None
    )
    late_delay = (
        alarm_index - valid_detection_end
        if outcome == "late_detection" and alarm_index is not None
        else None
    )

    true_positive = int(outcome == "detected")
    false_positive = int(outcome == "false_alarm")
    false_negative = int(outcome in {"late_detection", "missed"})

    if true_positive + false_positive + false_negative != 1:
        raise RuntimeError("Each run must be classified as exactly one TP, FP, or FN.")

    return {
        "configuration": configuration.name,
        "method": configuration.method,
        "window_mode": configuration.window_mode,
        "window_size": configuration.window_size,
        "min_window_size": configuration.min_window_size,
        "max_window_size": configuration.max_window_size,
        "n_subwindows": configuration.n_subwindows,
        "alpha": configuration.alpha,
        "correction": configuration.correction,
        "min_rejections": configuration.min_rejections,
        "drift_type": drift_type,
        "distribution": distribution,
        "row_index": row_index,
        "seed": seed,
        "alarm_index": alarm_index,
        "outcome": outcome,
        "drift_start": drift_start,
        "valid_detection_end": valid_detection_end,
        "delay": delay,
        "late_delay": late_delay,
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
    }


def _score_first_alarm(
    *,
    alarm_index: int | None,
    drift_start: int,
    valid_detection_end: int,
) -> Outcome:
    """Classify the first alarm relative to the valid detection interval."""
    if valid_detection_end < drift_start:
        raise ValueError("valid_detection_end must be greater than or equal to drift_start.")
    if alarm_index is None:
        return "missed"
    if alarm_index < drift_start:
        return "false_alarm"
    if alarm_index <= valid_detection_end:
        return "detected"
    return "late_detection"


def _first_mcdd_alarm(
    values: NDArray[np.float64],
    configuration: ExperimentConfiguration,
) -> int | None:
    detector = MCDD(
        ks_2samp,
        window_size=configuration.window_size,
        n_subwindows=configuration.n_subwindows,
        alpha=configuration.alpha,
        window_mode=configuration.window_mode,
        min_window_size=configuration.min_window_size,
        max_window_size=configuration.max_window_size,
        correction=configuration.correction or "fdr_by",
        min_rejections=configuration.min_rejections,
    )
    batch_size = detector.batch_size

    for end_index in range(batch_size, len(values), batch_size):
        detector.update(values[end_index - batch_size : end_index])
        if detector.drift_detected:
            return int(end_index)

    return None


def _first_tsh_alarm(
    values: NDArray[np.float64],
    configuration: ExperimentConfiguration,
) -> int | None:
    step_size = configuration.window_size // configuration.n_subwindows
    start_index = (
        configuration.window_size
        if configuration.window_mode == "sliding"
        else configuration.min_window_size or configuration.window_size
    )

    for end_index in range(start_index, len(values), step_size):
        if configuration.window_mode == "sliding":
            current_window_size = configuration.window_size
        else:
            maximum = configuration.max_window_size or configuration.window_size
            current_window_size = min(end_index, maximum)

        window_start = end_index - current_window_size
        window_midpoint = end_index - current_window_size // 2
        pvalue = _call_test(
            ks_2samp,
            values[window_start:window_midpoint],
            values[window_midpoint:end_index],
        )
        if pvalue < configuration.alpha:
            return int(end_index)

    return None


def _first_kswin_alarm(
    values: NDArray[np.float64],
    configuration: ExperimentConfiguration,
) -> int | None:
    from river import drift

    detector = drift.KSWIN(
        alpha=configuration.alpha,
        window_size=configuration.window_size,
        stat_size=configuration.window_size // configuration.n_subwindows,
        seed=42,
    )

    for index, value in enumerate(values):
        detector.update(float(value))
        if detector.drift_detected:
            return int(index)

    return None


def _first_lord_alarm(
    values: NDArray[np.float64],
    configuration: ExperimentConfiguration,
) -> int | None:
    step_size = configuration.window_size // configuration.n_subwindows
    indices = list(range(configuration.window_size, len(values), step_size))
    pvalues = np.empty(len(indices), dtype=np.float64)

    for position, end_index in enumerate(indices):
        window_start = end_index - configuration.window_size
        window_midpoint = end_index - configuration.window_size // 2
        pvalues[position] = _call_test(
            ks_2samp,
            values[window_start:window_midpoint],
            values[window_midpoint:end_index],
        )

    lag = max(0, math.ceil(configuration.window_size / step_size) - 1)
    procedure = LORDLocalDependence(
        alpha=configuration.alpha,
        number_of_hypotheses=len(pvalues),
        lag=lag,
        start_fraction=0.1,
        gamma_exponent=1.6,
    )
    rejection_positions = np.flatnonzero(procedure.run(pvalues))
    if rejection_positions.size == 0:
        return None

    first_position = int(rejection_positions[0])
    return int(configuration.window_size + first_position * step_size)


def _call_test(
    test: Any,
    first_sample: Sequence[float],
    second_sample: Sequence[float],
) -> float:
    result = test(first_sample, second_sample)

    if hasattr(result, "pvalue"):
        pvalue = float(result.pvalue)
    elif isinstance(result, (float, int, np.floating, np.integer)):
        pvalue = float(result)
    elif isinstance(result, tuple) and len(result) >= 2:
        pvalue = float(result[1])
    elif isinstance(result, dict):
        for key in ("pvalue", "p_val", "p"):
            if key in result:
                pvalue = float(result[key])
                break
        else:
            raise ValueError("The test mapping does not contain a p-value.")
    else:
        raise ValueError("Unsupported hypothesis-test output.")

    if not 0.0 <= pvalue <= 1.0:
        raise ValueError(f"Invalid p-value: {pvalue}.")
    return pvalue


def run_experiment_suite(
    *,
    data_directory: str | Path,
    output_file: str | Path,
    configurations: Sequence[ExperimentConfiguration] = PAPER_CONFIGURATIONS,
    max_streams_per_archive: int | None = None,
    overwrite: bool = False,
    progress_every: int = 25,
) -> Path:
    """Evaluate all configurations and write one CSV row per run."""
    archive_paths = expected_dataset_paths(data_directory)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace it."
        )

    total_streams = 0
    archive_limits: dict[Path, int] = {}
    for archive_path in archive_paths:
        with h5py.File(archive_path, "r") as archive:
            available = int(archive["values"].shape[0])
        limit = (
            available
            if max_streams_per_archive is None
            else min(available, max_streams_per_archive)
        )
        archive_limits[archive_path] = limit
        total_streams += limit

    total_runs = total_streams * len(configurations)
    completed_runs = 0

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RUN_FIELDS)
        writer.writeheader()

        for archive_path in archive_paths:
            with h5py.File(archive_path, "r") as archive:
                drift_type = _decode_attribute(archive.attrs["drift_type"])
                distribution = _decode_attribute(archive.attrs["distribution"])
                limit = archive_limits[archive_path]

                for row_index in range(limit):
                    values = np.asarray(archive["values"][row_index], dtype=np.float64)
                    seed = int(archive["seeds"][row_index])
                    drift_start = int(archive["drift_start"][row_index])
                    drift_end = int(archive["drift_end"][row_index])

                    for configuration in configurations:
                        writer.writerow(
                            evaluate_single_stream(
                                values,
                                drift_type=drift_type,
                                distribution=distribution,
                                row_index=row_index,
                                seed=seed,
                                drift_start=drift_start,
                                drift_end=drift_end,
                                configuration=configuration,
                            )
                        )
                        completed_runs += 1

                    handle.flush()
                    if progress_every > 0 and (row_index + 1) % progress_every == 0:
                        print(
                            f"{archive_path.name}: {row_index + 1}/{limit} streams; "
                            f"{completed_runs}/{total_runs} runs completed.",
                            flush=True,
                        )

    print(f"Per-run results written to {output_path}.")
    return output_path


def get_configuration(
    name: str,
    configurations: Sequence[ExperimentConfiguration] = PAPER_CONFIGURATIONS,
) -> ExperimentConfiguration:
    """Return one experiment configuration by its exact public name."""
    matches = [configuration for configuration in configurations if configuration.name == name]
    if not matches:
        available = ", ".join(configuration.name for configuration in configurations)
        raise ValueError(
            f"Unknown configuration '{name}'. Available configurations: {available}."
        )
    if len(matches) > 1:
        raise ValueError(f"Configuration name '{name}' is not unique.")
    return matches[0]


def run_archive_experiment(
    *,
    archive_path: str | Path,
    configuration: ExperimentConfiguration,
    output_file: str | Path,
    max_streams: int | None = None,
    overwrite: bool = False,
    progress_every: int = 25,
) -> Path:
    """Evaluate one detector configuration on one HDF5 dataset archive."""
    source_path = Path(archive_path)
    output_path = Path(output_file)

    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {source_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace it."
        )

    with h5py.File(source_path, "r") as archive:
        available_streams = int(archive["values"].shape[0])
        if max_streams is None:
            number_of_streams = available_streams
        else:
            if max_streams <= 0:
                raise ValueError("max_streams must be positive or None.")
            number_of_streams = min(int(max_streams), available_streams)

        drift_type = _decode_attribute(archive.attrs["drift_type"])
        distribution = _decode_attribute(archive.attrs["distribution"])
        temporary_path = output_path.with_suffix(output_path.suffix + ".part")

        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PER_RUN_FIELDS)
                writer.writeheader()

                for row_index in range(number_of_streams):
                    values = np.asarray(archive["values"][row_index], dtype=np.float64)
                    result = evaluate_single_stream(
                        values,
                        drift_type=drift_type,
                        distribution=distribution,
                        row_index=row_index,
                        seed=int(archive["seeds"][row_index]),
                        drift_start=int(archive["drift_start"][row_index]),
                        drift_end=int(archive["drift_end"][row_index]),
                        configuration=configuration,
                    )
                    writer.writerow(result)
                    handle.flush()

                    completed = row_index + 1
                    if progress_every > 0 and (
                        completed % progress_every == 0
                        or completed == number_of_streams
                    ):
                        print(
                            f"{configuration.name} on {source_path.name}: "
                            f"{completed}/{number_of_streams} streams completed.",
                            flush=True,
                        )

            temporary_path.replace(output_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    print(f"Per-run results written to {output_path}.")
    return output_path


def summarize_archive_results(
    *,
    per_run_file: str | Path,
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """Create one summary row for a single archive/configuration experiment."""
    per_run_path = Path(per_run_file)
    if not per_run_path.is_file():
        raise FileNotFoundError(f"Per-run result file not found: {per_run_path}.")

    results = pd.read_csv(per_run_path)
    if results.empty:
        raise ValueError("The per-run result file is empty.")

    identifying_columns = (
        "configuration",
        "method",
        "drift_type",
        "distribution",
    )
    for column in identifying_columns:
        if results[column].nunique(dropna=False) != 1:
            raise ValueError(
                "summarize_archive_results expects results from exactly one "
                f"configuration and one dataset archive; column '{column}' "
                "contains multiple values."
            )

    summary = _aggregate(results, group_columns=list(identifying_columns))
    ordered_columns = [
        "configuration",
        "method",
        "drift_type",
        "distribution",
        "replications",
        "TP",
        "FP",
        "FN",
        "FDR",
        "MDR",
        "IR",
        "mean_delay",
    ]
    summary = summary[ordered_columns]

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Summary results written to {output_path}.")
    return summary


def summarize_results(
    *,
    per_run_file: str | Path,
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate FDR, MDR, IR, and mean delay from per-run results."""
    per_run_path = Path(per_run_file)
    results = pd.read_csv(per_run_path)

    exact = _aggregate(
        results,
        group_columns=["configuration", "method", "drift_type", "distribution"],
    )
    by_drift = _aggregate(
        results,
        group_columns=["configuration", "method", "drift_type"],
    )
    by_drift["distribution"] = "all"
    overall = _aggregate(results, group_columns=["configuration", "method"])
    overall["drift_type"] = "all"
    overall["distribution"] = "all"

    summary = pd.concat([exact, by_drift, overall], ignore_index=True, sort=False)
    ordered_columns = [
        "configuration",
        "method",
        "drift_type",
        "distribution",
        "replications",
        "TP",
        "FP",
        "FN",
        "FDR",
        "MDR",
        "IR",
        "mean_delay",
    ]
    summary = summary[ordered_columns].sort_values(
        ["configuration", "drift_type", "distribution"],
        ignore_index=True,
    )

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Summary results written to {output_path}.")
    return summary


def _aggregate(
    results: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    required = {"TP", "FP", "FN", "delay", *group_columns}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(
            "The per-run results are missing required columns: " + ", ".join(missing)
        )

    classifications = results[["TP", "FP", "FN"]].sum(axis=1)
    if not classifications.eq(1).all():
        raise ValueError("Every per-run row must satisfy TP + FP + FN = 1.")

    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys, strict=True))
        true_positives = int(group["TP"].sum())
        false_positives = int(group["FP"].sum())
        false_negatives = int(group["FN"].sum())
        detections = true_positives + false_positives
        true_drifts = true_positives + false_negatives

        row.update(
            {
                "replications": int(len(group)),
                "TP": true_positives,
                "FP": false_positives,
                "FN": false_negatives,
                "FDR": (
                    false_positives / detections if detections > 0 else np.nan
                ),
                "MDR": (
                    false_negatives / true_drifts if true_drifts > 0 else np.nan
                ),
                "IR": (
                    true_positives / detections if detections > 0 else np.nan
                ),
                "mean_delay": (
                    float(group["delay"].dropna().mean())
                    if group["delay"].notna().any()
                    else np.nan
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _decode_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
