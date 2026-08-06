#!/usr/bin/env python3
"""Generate the HDF5 datasets used in the MCDD experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Allow the script to be executed directly from the repository root without
# installing the package first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from mcdd.datasets.generators import (  # noqa: E402
    SUPPORTED_DISTRIBUTIONS,
    SUPPORTED_DRIFT_TYPES,
    generate_stream,
)

N_SAMPLES = 70_000
DRIFT_POINT = 40_000
NOISE_STANDARD_DEVIATION = 0.1
REPLICATIONS = 1_000
FIRST_SEED = 42
TRANSITION_SIZES = (1_000, 2_000, 3_000)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate nine HDF5 files: one for every combination of three "
            "drift types and three distributions."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "datasets",
        help="Directory in which the HDF5 files will be created.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=REPLICATIONS,
        help=(
            "Number of streams per drift/distribution combination. The paper "
            "uses 1000."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing HDF5 files.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(0, 10),
        metavar="0-9",
        help="Gzip compression level used by HDF5.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.replications <= 0:
        raise ValueError("--replications must be greater than zero.")

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    total_streams = (
        arguments.replications
        * len(SUPPORTED_DRIFT_TYPES)
        * len(SUPPORTED_DISTRIBUTIONS)
    )
    print(f"Generating {total_streams:,} streams in {output_directory}")

    started_at = time.monotonic()
    completed = 0

    for drift_type in SUPPORTED_DRIFT_TYPES:
        for distribution in SUPPORTED_DISTRIBUTIONS:
            output_path = output_directory / f"{drift_type}_{distribution}.h5"
            _write_archive(
                output_path=output_path,
                drift_type=drift_type,
                distribution=distribution,
                replications=arguments.replications,
                overwrite=arguments.overwrite,
                compression_level=arguments.compression_level,
            )
            completed += arguments.replications
            elapsed = time.monotonic() - started_at
            print(
                f"[{completed:,}/{total_streams:,}] {output_path.name} "
                f"({elapsed / 60.0:.1f} min)",
                flush=True,
            )

    elapsed = time.monotonic() - started_at
    print(f"Generation completed in {elapsed / 60.0:.1f} minutes.")
    return 0


def _write_archive(
    *,
    output_path: Path,
    drift_type: str,
    distribution: str,
    replications: int,
    overwrite: bool,
    compression_level: int,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    temporary_path = output_path.with_suffix(".h5.part")
    temporary_path.unlink(missing_ok=True)

    parameter_count = 1 if distribution == "exponential" else 2
    compression = "gzip" if compression_level > 0 else None
    compression_options = compression_level if compression else None

    try:
        with h5py.File(temporary_path, "w") as archive:
            archive.attrs["drift_type"] = drift_type
            archive.attrs["distribution"] = distribution
            archive.attrs["n_samples"] = N_SAMPLES
            archive.attrs["replications"] = replications
            archive.attrs["first_seed"] = FIRST_SEED
            archive.attrs["noise_standard_deviation"] = (
                NOISE_STANDARD_DEVIATION
            )
            archive.attrs["transition_sizes"] = TRANSITION_SIZES

            values_dataset = archive.create_dataset(
                "values",
                shape=(replications, N_SAMPLES),
                dtype=np.float64,
                chunks=(1, N_SAMPLES),
                compression=compression,
                compression_opts=compression_options,
                shuffle=True,
            )
            seeds_dataset = archive.create_dataset(
                "seeds", shape=(replications,), dtype=np.int64
            )
            drift_start_dataset = archive.create_dataset(
                "drift_start", shape=(replications,), dtype=np.int64
            )
            drift_end_dataset = archive.create_dataset(
                "drift_end", shape=(replications,), dtype=np.int64
            )
            transition_size_dataset = archive.create_dataset(
                "transition_size", shape=(replications,), dtype=np.int64
            )
            before_parameters_dataset = archive.create_dataset(
                "before_parameters",
                shape=(replications, parameter_count),
                dtype=np.float64,
            )
            after_parameters_dataset = archive.create_dataset(
                "after_parameters",
                shape=(replications, parameter_count),
                dtype=np.float64,
            )

            parameter_names: tuple[str, ...] | None = None
            for replication in range(replications):
                seed = FIRST_SEED + replication
                stream = generate_stream(
                    drift_type=drift_type,
                    distribution=distribution,
                    seed=seed,
                    n_samples=N_SAMPLES,
                    drift_point=DRIFT_POINT,
                    noise_standard_deviation=NOISE_STANDARD_DEVIATION,
                    transition_sizes=TRANSITION_SIZES,
                )

                values_dataset[replication] = stream.values
                seeds_dataset[replication] = seed
                drift_start_dataset[replication] = stream.drift_start
                drift_end_dataset[replication] = stream.drift_end
                transition_size_dataset[replication] = stream.transition_size
                before_parameters_dataset[replication] = (
                    stream.before_parameters
                )
                after_parameters_dataset[replication] = stream.after_parameters
                parameter_names = stream.parameter_names

            if parameter_names is not None:
                archive.attrs["parameter_names"] = parameter_names

        if output_path.exists():
            output_path.unlink()
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
