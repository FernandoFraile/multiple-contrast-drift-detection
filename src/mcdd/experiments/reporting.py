"""Article-style result reporting for the MCDD experiments.

For each detector configuration and drift type, the reported table uses the
arithmetic mean of the distribution-specific metrics for normal, exponential,
and gamma streams. The overall table then averages the drift-level values.
Pooled ``distribution="all"`` rows are not used for these tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    DISTRIBUTIONS,
    DRIFT_TYPES,
    PAPER_CONFIGURATIONS,
)

ARTICLE_METRICS: tuple[str, ...] = (
    "FDR",
    "MDR",
    "IR",
    "mean_delay",
)
ARTICLE_COLUMNS: tuple[str, ...] = (
    "Config",
    "FDR",
    "MDR",
    "IR",
    "Mean Delay",
)


def _configuration_order() -> list[str]:
    return [configuration.name for configuration in PAPER_CONFIGURATIONS]


def _load_distribution_rows(
    summary_file: str | Path,
    *,
    require_all_distributions: bool = True,
) -> pd.DataFrame:
    """Read and validate the distribution-specific summary rows."""
    summary_path = Path(summary_file)

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Summary result file not found: {summary_path}."
        )

    results = pd.read_csv(summary_path)

    required_columns = {
        "configuration",
        "method",
        "drift_type",
        "distribution",
        "TP",
        "FP",
        "FN",
        *ARTICLE_METRICS,
    }
    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        raise ValueError(
            "The summary file is missing required columns: "
            + ", ".join(missing_columns)
        )

    exact = results[
        results["drift_type"].isin(DRIFT_TYPES)
        & results["distribution"].isin(DISTRIBUTIONS)
    ].copy()

    if exact.empty:
        raise ValueError(
            "No distribution-specific drift rows were found in the summary file."
        )

    duplicate_mask = exact.duplicated(
        subset=["configuration", "drift_type", "distribution"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = exact.loc[
            duplicate_mask,
            ["configuration", "drift_type", "distribution"],
        ]
        raise ValueError(
            "The summary file contains duplicate distribution rows:\n"
            + duplicates.to_string(index=False)
        )

    if require_all_distributions:
        expected_distributions = set(DISTRIBUTIONS)
        for (configuration, drift_type), group in exact.groupby(
            ["configuration", "drift_type"],
            dropna=False,
        ):
            present = set(group["distribution"])
            if present != expected_distributions:
                missing = sorted(expected_distributions.difference(present))
                raise ValueError(
                    "Expected normal, exponential, and gamma rows for "
                    f"{configuration}/{drift_type}; missing="
                    + ", ".join(missing)
                )

    return exact


def average_metrics_by_drift(
    summary_file: str | Path,
    *,
    article_na: bool = True,
    require_all_distributions: bool = True,
) -> pd.DataFrame:
    """Average distribution-specific metrics within each available drift type.

    With ``article_na=True``, an undefined metric in any included distribution
    remains undefined in the drift-level table rather than being omitted from
    the arithmetic mean.
    """
    exact = _load_distribution_rows(
        summary_file,
        require_all_distributions=require_all_distributions,
    )
    rows: list[dict[str, object]] = []
    distribution_rank = {
        name: index for index, name in enumerate(DISTRIBUTIONS)
    }

    for (configuration, method, drift_type), group in exact.groupby(
        ["configuration", "method", "drift_type"],
        dropna=False,
        sort=False,
    ):
        true_positives = int(group["TP"].sum())
        false_positives = int(group["FP"].sum())
        false_negatives = int(group["FN"].sum())
        present_distributions = sorted(
            set(group["distribution"]),
            key=lambda name: distribution_rank.get(name, len(distribution_rank)),
        )

        row: dict[str, object] = {
            "configuration": configuration,
            "method": method,
            "drift_type": drift_type,
            "distributions": int(len(group)),
            "distribution_names": ",".join(present_distributions),
            "TP": true_positives,
            "FP": false_positives,
            "FN": false_negatives,
        }

        for metric in ARTICLE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            if article_na and values.isna().any():
                row[metric] = np.nan
            else:
                row[metric] = float(values.mean())

        rows.append(row)

    averaged = pd.DataFrame(rows)

    configuration_rank = {
        name: index for index, name in enumerate(_configuration_order())
    }
    drift_rank = {
        name: index for index, name in enumerate(DRIFT_TYPES)
    }

    averaged["_configuration_order"] = averaged["configuration"].map(
        configuration_rank
    ).fillna(len(configuration_rank))
    averaged["_drift_order"] = averaged["drift_type"].map(
        drift_rank
    ).fillna(len(drift_rank))

    averaged = averaged.sort_values(
        ["_drift_order", "_configuration_order", "configuration"],
        ignore_index=True,
    ).drop(columns=["_configuration_order", "_drift_order"])

    return averaged


def average_metrics_overall(
    by_drift: pd.DataFrame,
    *,
    require_all_drifts: bool = True,
) -> pd.DataFrame:
    """Average drift-level metric values for each detector."""
    required_columns = {
        "configuration",
        "method",
        "drift_type",
        *ARTICLE_METRICS,
    }
    missing = sorted(required_columns.difference(by_drift.columns))
    if missing:
        raise ValueError(
            "The drift-level table is missing required columns: "
            + ", ".join(missing)
        )

    expected_drifts = set(DRIFT_TYPES)
    rows: list[dict[str, object]] = []

    for (configuration, method), group in by_drift.groupby(
        ["configuration", "method"],
        dropna=False,
        sort=False,
    ):
        present_drifts = set(group["drift_type"])

        if require_all_drifts and present_drifts != expected_drifts:
            missing_drifts = sorted(expected_drifts.difference(present_drifts))
            raise ValueError(
                f"{configuration} is missing drift rows: "
                + ", ".join(missing_drifts)
            )

        row: dict[str, object] = {
            "configuration": configuration,
            "method": method,
            "drifts": int(len(group)),
            "drift_names": ",".join(
                drift for drift in DRIFT_TYPES if drift in present_drifts
            ),
        }

        for metric in ARTICLE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.isna().any():
                row[metric] = np.nan
            else:
                row[metric] = float(values.mean())

        rows.append(row)

    overall = pd.DataFrame(rows)

    configuration_rank = {
        name: index for index, name in enumerate(_configuration_order())
    }
    overall["_configuration_order"] = overall["configuration"].map(
        configuration_rank
    ).fillna(len(configuration_rank))

    overall = overall.sort_values(
        ["_configuration_order", "configuration"],
        ignore_index=True,
    ).drop(columns=["_configuration_order"])

    return overall


def article_table_for_drift(
    by_drift: pd.DataFrame,
    drift_type: str,
) -> pd.DataFrame:
    """Return one drift table with the metric columns used in the paper."""
    if drift_type not in DRIFT_TYPES:
        raise ValueError(
            f"Unknown drift type {drift_type!r}. Choose from {DRIFT_TYPES}."
        )

    table = by_drift.loc[
        by_drift["drift_type"] == drift_type,
        ["configuration", *ARTICLE_METRICS],
    ].copy()

    return _rename_article_columns(table)


def overall_article_table(
    overall: pd.DataFrame,
) -> pd.DataFrame:
    """Return the overall detector table with article-style column names."""
    table = overall[["configuration", *ARTICLE_METRICS]].copy()
    return _rename_article_columns(table)


def build_article_tables(
    summary_file: str | Path,
    *,
    article_na: bool = True,
    require_all_distributions: bool = True,
    require_all_drifts: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build available drift tables and an overall comparison."""
    by_drift = average_metrics_by_drift(
        summary_file,
        article_na=article_na,
        require_all_distributions=require_all_distributions,
    )
    overall = average_metrics_overall(
        by_drift,
        require_all_drifts=require_all_drifts,
    )

    tables = {
        drift_type: article_table_for_drift(by_drift, drift_type)
        for drift_type in DRIFT_TYPES
    }
    tables["overall"] = overall_article_table(overall)
    return tables


def format_article_table(
    table: pd.DataFrame,
    *,
    decimals: int = 4,
    na_rep: str = "NA",
) -> pd.DataFrame:
    """Return a display copy with fixed decimals and explicit ``NA`` values."""
    if decimals < 0:
        raise ValueError("decimals must be zero or greater.")

    formatted = table.copy()
    numeric_columns = [
        column
        for column in ("FDR", "MDR", "IR", "Mean Delay")
        if column in formatted.columns
    ]

    for column in numeric_columns:
        formatted[column] = formatted[column].map(
            lambda value: (
                na_rep
                if pd.isna(value)
                else f"{float(value):.{decimals}f}"
            )
        )

    return formatted


def article_table_to_latex(
    table: pd.DataFrame,
    *,
    decimals: int = 4,
    na_rep: str = "NA",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Convert an article table to LaTeX with fixed decimal precision."""
    formatted = format_article_table(
        table,
        decimals=decimals,
        na_rep=na_rep,
    )
    return formatted.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )


def _rename_article_columns(table: pd.DataFrame) -> pd.DataFrame:
    renamed = table.rename(
        columns={
            "configuration": "Config",
            "mean_delay": "Mean Delay",
        }
    )
    return renamed.loc[:, list(ARTICLE_COLUMNS)]
