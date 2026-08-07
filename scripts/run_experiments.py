#!/usr/bin/env python3
"""Run the complete MCDD benchmark experiment from the command line."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow direct execution from the repository root without installing the
# package first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from mcdd.experiments import (  # noqa: E402
    PAPER_CONFIGURATIONS,
    run_experiment_suite,
    summarize_results,
)

DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "datasets"
DEFAULT_RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the benchmark execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all paper detector configurations on the nine HDF5 "
            "benchmark archives and generate per-run and summary CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=(
            "Directory containing the nine HDF5 benchmark archives. "
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
        "--max-streams",
        "--max-streams-per-archive",
        dest="max_streams_per_archive",
        type=int,
        default=None,
        help=(
            "Maximum number of streams evaluated from each HDF5 archive. "
            "Omit this option to evaluate all 1,000 streams per archive."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per_run_results.csv file.",
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


def main() -> int:
    """Execute the benchmark and generate its summary."""
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

    streams_per_archive = (
        "all available streams"
        if arguments.max_streams_per_archive is None
        else str(arguments.max_streams_per_archive)
    )

    print("MCDD benchmark execution")
    print(f"Data directory: {data_directory}")
    print(f"Results directory: {results_directory}")
    print(f"Detector configurations: {len(PAPER_CONFIGURATIONS)}")
    print(f"Streams per archive: {streams_per_archive}")
    print(
        "Existing per-run results will be "
        + ("overwritten." if arguments.overwrite else "protected.")
    )

    started_at = time.monotonic()

    run_experiment_suite(
        data_directory=data_directory,
        output_file=per_run_file,
        configurations=PAPER_CONFIGURATIONS,
        max_streams_per_archive=arguments.max_streams_per_archive,
        overwrite=arguments.overwrite,
        progress_every=arguments.progress_every,
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
