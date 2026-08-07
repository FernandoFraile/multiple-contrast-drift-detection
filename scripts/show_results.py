#!/usr/bin/env python3
"""Display available article-style MCDD result tables from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pandas import read_csv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from mcdd.experiments import (  # noqa: E402
    article_table_to_latex,
    build_article_tables,
    format_article_table,
)

DEFAULT_SUMMARY_FILE = REPOSITORY_ROOT / "results" / "summary_results.csv"
DRIFT_TYPES = ("abrupt", "gradual", "incremental")
DISTRIBUTIONS = ("normal", "exponential", "gamma")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display article-style result tables from the experiment results "
            "currently available in summary_results.csv."
        )
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY_FILE,
        help=(
            "Path to summary_results.csv. Defaults to "
            "results/summary_results.csv."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places displayed. Defaults to 4.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX tables instead of plain-text tables.",
    )
    parser.add_argument(
        "--no-article-na",
        action="store_true",
        help=(
            "Disable the article-style NA convention for scenarios with no "
            "valid detections."
        ),
    )
    return parser.parse_args()


def _available_scenarios(summary_file: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Return available drift types and distributions for each drift type."""
    if not summary_file.is_file():
        raise FileNotFoundError(f"Summary result file not found: {summary_file}.")

    results = read_csv(summary_file)
    exact = results[
        results["drift_type"].isin(DRIFT_TYPES)
        & results["distribution"].isin(DISTRIBUTIONS)
    ]

    drift_types = set(exact["drift_type"])
    distributions = {
        drift_type: set(
            exact.loc[
                exact["drift_type"] == drift_type,
                "distribution",
            ]
        )
        for drift_type in drift_types
    }
    return drift_types, distributions


def main() -> int:
    arguments = parse_arguments()

    if arguments.decimals < 0:
        raise ValueError("--decimals must be zero or greater.")

    summary_file = arguments.summary_file.resolve()
    available_drifts, available_distributions = _available_scenarios(summary_file)

    tables = build_article_tables(
        summary_file,
        article_na=not arguments.no_article_na,
        require_all_distributions=False,
        require_all_drifts=False,
    )

    titles = {
        "abrupt": "Abrupt drift",
        "gradual": "Gradual drift",
        "incremental": "Incremental drift",
        "overall": "Overall comparison",
    }

    for key in DRIFT_TYPES:
        if key not in available_drifts:
            continue

        table = tables[key]
        title = titles[key]
        present_distributions = available_distributions[key]
        complete_distributions = present_distributions == set(DISTRIBUTIONS)

        print()
        print(f"=== {title} ===")
        if complete_distributions:
            print("Averaged across: normal, exponential, gamma")
        else:
            ordered = [
                distribution
                for distribution in DISTRIBUTIONS
                if distribution in present_distributions
            ]
            missing = [
                distribution
                for distribution in DISTRIBUTIONS
                if distribution not in present_distributions
            ]
            print(
                "PARTIAL RESULT — averaged across available distributions: "
                + ", ".join(ordered)
            )
            print("Missing distributions: " + ", ".join(missing))

        if arguments.latex:
            print(
                article_table_to_latex(
                    table,
                    decimals=arguments.decimals,
                    caption=title,
                    label=f"tab:{key}_results",
                )
            )
        else:
            formatted = format_article_table(
                table,
                decimals=arguments.decimals,
            )
            print(formatted.to_string(index=False))

    if available_drifts:
        print()
        print("=== Overall comparison ===")
        if available_drifts == set(DRIFT_TYPES):
            print("Averaged across: abrupt, gradual, incremental")
        else:
            ordered_drifts = [
                drift_type
                for drift_type in DRIFT_TYPES
                if drift_type in available_drifts
            ]
            missing_drifts = [
                drift_type
                for drift_type in DRIFT_TYPES
                if drift_type not in available_drifts
            ]
            print(
                "PARTIAL RESULT — averaged across available drift types: "
                + ", ".join(ordered_drifts)
            )
            print("Missing drift types: " + ", ".join(missing_drifts))

        table = tables["overall"]
        if arguments.latex:
            print(
                article_table_to_latex(
                    table,
                    decimals=arguments.decimals,
                    caption=titles["overall"],
                    label="tab:overall_results",
                )
            )
        else:
            formatted = format_article_table(
                table,
                decimals=arguments.decimals,
            )
            print(formatted.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
