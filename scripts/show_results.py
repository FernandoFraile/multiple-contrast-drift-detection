#!/usr/bin/env python3
"""Display article-style MCDD result tables from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display the article-style abrupt, gradual, incremental, and "
            "overall result tables from summary_results.csv."
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


def main() -> int:
    arguments = parse_arguments()

    if arguments.decimals < 0:
        raise ValueError("--decimals must be zero or greater.")

    tables = build_article_tables(
        arguments.summary_file.resolve(),
        article_na=not arguments.no_article_na,
    )

    titles = {
        "abrupt": "Abrupt drift",
        "gradual": "Gradual drift",
        "incremental": "Incremental drift",
        "overall": "Overall comparison",
    }

    for key in ("abrupt", "gradual", "incremental", "overall"):
        table = tables[key]
        title = titles[key]

        print()
        print(f"=== {title} ===")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
