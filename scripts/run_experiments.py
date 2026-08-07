#!/usr/bin/env python3
"""Run the MCDD benchmark experiment from the command line.

The runner supports both the complete 90,000-run benchmark and partial
executions filtered by drift type, distribution, or detector configuration.
Partial executions can be appended safely to the same master result file,
which makes it possible to run abrupt, gradual, and incremental drift in
separate sessions without repeating completed work.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
import sys

import pandas as pd

# Allow direct execution from the repository root without installing the
# package first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from mcdd.experiments import (  # noqa: E402
    PAPER_CONFIGURATIONS,
    get_configuration,
    run_archive_experiment,
    run_experiment_suite,
    summarize_results,
)

DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "datasets"
DEFAULT_RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
DRIFT_TYPES = ("abrupt", "gradual", "incremental")
DISTRIBUTIONS = ("normal", "exponential", "gamma")
RESULT_KEY_COLUMNS = (
    "configuration",
    "drift_type",
    "distribution",
    "row_index",
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the benchmark execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all or part of the MCDD paper benchmark and generate "
            "per-run and summary CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=(
            "Directory containing the HDF5 benchmark archives. "
            "Defaults to data/datasets."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help=(
            "Directory in which result CSV files are written. "
            "Defaults to results."
        ),
    )
    parser.add_argument(
        "--drift",
        action="append",
        choices=DRIFT_TYPES,
        default=None,
        help=(
            "Run only the selected drift type. Repeat the option to select "
            "multiple drift types. Omit it to use all drift types."
        ),
    )
    parser.add_argument(
        "--distribution",
        action="append",
        choices=DISTRIBUTIONS,
        default=None,
        help=(
            "Run only the selected distribution. Repeat the option to select "
            "multiple distributions. Omit it to use all distributions."
        ),
    )
    parser.add_argument(
        "--configuration",
        "--config",
        dest="configuration",
        action="append",
        default=None,
        help=(
            "Run only one detector configuration, for example MCDD-S or "
            "KSWIN. Repeat the option to select multiple configurations. "
            "Omit it to use all paper configurations."
        ),
    )
    parser.add_argument(
        "--max-streams",
        "--max-streams-per-archive",
        dest="max_streams_per_archive",
        type=int,
        default=None,
        help=(
            "Maximum number of streams evaluated from each selected HDF5 "
            "archive. Omit this option to evaluate all 1,000 streams."
        ),
    )

    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the existing master per_run_results.csv with only the "
            "current selection. Useful after a reduced validation run."
        ),
    )
    write_mode.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append the current selection to an existing master result file "
            "and regenerate summary_results.csv. Duplicate runs are rejected."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "Print progress after this many streams within each archive. "
            "Use 0 to disable progress messages. Defaults to 25."
        ),
    )
    return parser.parse_args()


def _selected_configurations(names: list[str] | None):
    if not names:
        return PAPER_CONFIGURATIONS

    unique_names = list(dict.fromkeys(names))
    return tuple(get_configuration(name) for name in unique_names)


def _selected_values(
    requested: list[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested:
        return available
    return tuple(dict.fromkeys(requested))


def _is_complete_selection(
    drift_types: tuple[str, ...],
    distributions: tuple[str, ...],
    configuration_count: int,
) -> bool:
    return (
        set(drift_types) == set(DRIFT_TYPES)
        and set(distributions) == set(DISTRIBUTIONS)
        and configuration_count == len(PAPER_CONFIGURATIONS)
    )


def _selected_archive_paths(
    data_directory: Path,
    drift_types: tuple[str, ...],
    distributions: tuple[str, ...],
) -> list[Path]:
    paths = [
        data_directory / f"{drift_type}_{distribution}.h5"
        for drift_type in drift_types
        for distribution in distributions
    ]
    missing = [path for path in paths if not path.is_file()]

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The following selected dataset archives are missing:\n"
            + missing_text
        )

    return paths


def _run_filtered_selection(
    *,
    archive_paths: list[Path],
    configurations,
    max_streams_per_archive: int | None,
    progress_every: int,
) -> pd.DataFrame:
    """Run a filtered selection and return its per-run rows."""
    frames: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory(prefix="mcdd_partial_") as temporary:
        temporary_directory = Path(temporary)

        for archive_path in archive_paths:
            for configuration in configurations:
                safe_configuration = (
                    configuration.name.lower().replace("-", "_")
                )
                temporary_file = temporary_directory / (
                    f"{archive_path.stem}__{safe_configuration}.csv"
                )

                run_archive_experiment(
                    archive_path=archive_path,
                    configuration=configuration,
                    output_file=temporary_file,
                    max_streams=max_streams_per_archive,
                    overwrite=True,
                    progress_every=progress_every,
                )
                frames.append(pd.read_csv(temporary_file))

    if not frames:
        raise RuntimeError("The selected experiment did not produce any rows.")

    return pd.concat(frames, ignore_index=True)


def _duplicate_keys(
    existing: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return run keys present in both existing and new result sets."""
    existing_keys = existing.loc[:, RESULT_KEY_COLUMNS].drop_duplicates()
    new_keys = new_results.loc[:, RESULT_KEY_COLUMNS].drop_duplicates()
    return existing_keys.merge(
        new_keys,
        on=list(RESULT_KEY_COLUMNS),
        how="inner",
    )


def _write_results_atomically(
    results: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".part")

    try:
        results.to_csv(temporary_file, index=False)
        temporary_file.replace(output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


def _merge_partial_results(
    *,
    new_results: pd.DataFrame,
    per_run_file: Path,
    append: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Write or append a filtered result selection safely."""
    if append:
        if per_run_file.is_file():
            existing = pd.read_csv(per_run_file)
            duplicates = _duplicate_keys(existing, new_results)

            if not duplicates.empty:
                examples = duplicates.head(5).to_string(index=False)
                raise ValueError(
                    f"Refusing to append {len(duplicates)} duplicate runs. "
                    "The same configuration/drift/distribution/row_index "
                    "combination is already present. Use --overwrite to "
                    "start again, or choose a selection that has not yet "
                    f"been executed. Example duplicates:\n{examples}"
                )

            combined = pd.concat(
                [existing, new_results],
                ignore_index=True,
            )
        else:
            combined = new_results

        _write_results_atomically(combined, per_run_file)
        return combined

    if per_run_file.exists() and not overwrite:
        raise FileExistsError(
            f"{per_run_file} already exists. Use --append to add a new "
            "selection or --overwrite to replace it."
        )

    _write_results_atomically(new_results, per_run_file)
    return new_results


def main() -> int:
    """Execute the requested benchmark selection and generate its summary."""
    arguments = parse_arguments()

    if (
        arguments.max_streams_per_archive is not None
        and arguments.max_streams_per_archive <= 0
    ):
        raise ValueError("--max-streams must be a positive integer.")
    if arguments.progress_every < 0:
        raise ValueError("--progress-every must be zero or a positive integer.")

    data_directory = arguments.data_dir.resolve()
    results_directory = arguments.results_dir.resolve()
    results_directory.mkdir(parents=True, exist_ok=True)

    per_run_file = results_directory / "per_run_results.csv"
    summary_file = results_directory / "summary_results.csv"

    drift_types = _selected_values(arguments.drift, DRIFT_TYPES)
    distributions = _selected_values(
        arguments.distribution,
        DISTRIBUTIONS,
    )
    configurations = _selected_configurations(arguments.configuration)

    streams_per_archive = (
        "all available streams"
        if arguments.max_streams_per_archive is None
        else str(arguments.max_streams_per_archive)
    )

    print("MCDD benchmark execution")
    print(f"Data directory: {data_directory}")
    print(f"Results directory: {results_directory}")
    print(f"Drift types: {', '.join(drift_types)}")
    print(f"Distributions: {', '.join(distributions)}")
    print(
        "Detector configurations: "
        + ", ".join(configuration.name for configuration in configurations)
    )
    print(f"Streams per selected archive: {streams_per_archive}")
    print(
        "Write mode: "
        + (
            "append"
            if arguments.append
            else "overwrite"
            if arguments.overwrite
            else "protect existing results"
        )
    )

    started_at = time.monotonic()

    complete_selection = _is_complete_selection(
        drift_types,
        distributions,
        len(configurations),
    )

    # Keep the original efficient full-suite path when the complete benchmark
    # is requested in one execution. Filtered or append runs use the archive
    # runner so only the requested work is performed.
    if complete_selection and not arguments.append:
        run_experiment_suite(
            data_directory=data_directory,
            output_file=per_run_file,
            configurations=configurations,
            max_streams_per_archive=arguments.max_streams_per_archive,
            overwrite=arguments.overwrite,
            progress_every=arguments.progress_every,
        )
    else:
        archive_paths = _selected_archive_paths(
            data_directory,
            drift_types,
            distributions,
        )
        new_results = _run_filtered_selection(
            archive_paths=archive_paths,
            configurations=configurations,
            max_streams_per_archive=arguments.max_streams_per_archive,
            progress_every=arguments.progress_every,
        )
        combined = _merge_partial_results(
            new_results=new_results,
            per_run_file=per_run_file,
            append=arguments.append,
            overwrite=arguments.overwrite,
        )
        print(
            f"Master per-run file now contains {len(combined):,} rows."
        )

    summary = summarize_results(
        per_run_file=per_run_file,
        output_file=summary_file,
    )

    elapsed = time.monotonic() - started_at
    print(f"Experiment completed in {elapsed / 60.0:.1f} minutes.")
    print(f"Per-run results: {per_run_file}")
    print(f"Summary results: {summary_file}")
    print(f"Summary rows: {len(summary)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
