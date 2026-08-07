"""Tests for article-style result aggregation and formatting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mcdd.experiments import (
    article_table_for_drift,
    average_metrics_by_drift,
    average_metrics_overall,
    format_article_table,
    overall_article_table,
)


def _write_summary(path: Path) -> None:
    rows: list[dict[str, object]] = []

    values = {
        "abrupt": [
            (0.10, 0.30, 0.90, 100.0),
            (0.20, 0.20, 0.80, 200.0),
            (0.30, 0.10, 0.70, 300.0),
        ],
        "gradual": [
            (0.20, 0.20, 0.80, 200.0),
            (0.30, 0.10, 0.70, 300.0),
            (0.40, 0.00, 0.60, 400.0),
        ],
        "incremental": [
            (0.30, 0.10, 0.70, 300.0),
            (0.40, 0.00, 0.60, 400.0),
            (0.50, 0.20, 0.50, 500.0),
        ],
    }

    distributions = ("normal", "exponential", "gamma")

    for drift_type, metric_rows in values.items():
        for distribution, metrics in zip(distributions, metric_rows, strict=True):
            fdr, mdr, ir, mean_delay = metrics
            rows.append(
                {
                    "configuration": "MCDD-S",
                    "method": "mcdd",
                    "drift_type": drift_type,
                    "distribution": distribution,
                    "replications": 1000,
                    "TP": 900,
                    "FP": 100,
                    "FN": 100,
                    "FDR": fdr,
                    "MDR": mdr,
                    "IR": ir,
                    "mean_delay": mean_delay,
                }
            )

    # KSWIN reproduces the article NA convention: only false alarms and no
    # valid detections for each drift type.
    for drift_type in ("abrupt", "gradual", "incremental"):
        for distribution in distributions:
            rows.append(
                {
                    "configuration": "KSWIN",
                    "method": "kswin",
                    "drift_type": drift_type,
                    "distribution": distribution,
                    "replications": 1000,
                    "TP": 0,
                    "FP": 1000,
                    "FN": 0,
                    "FDR": 1.0,
                    "MDR": 0.0,
                    "IR": 0.0,
                    "mean_delay": 0.0,
                }
            )

    # Pooled rows must be ignored by article reporting.
    rows.append(
        {
            "configuration": "MCDD-S",
            "method": "mcdd",
            "drift_type": "abrupt",
            "distribution": "all",
            "replications": 3000,
            "TP": 1,
            "FP": 999,
            "FN": 0,
            "FDR": 0.999,
            "MDR": 0.999,
            "IR": 0.001,
            "mean_delay": 9999.0,
        }
    )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_distribution_means_are_used_for_each_drift(tmp_path: Path) -> None:
    summary_file = tmp_path / "summary_results.csv"
    _write_summary(summary_file)

    by_drift = average_metrics_by_drift(summary_file)
    abrupt = article_table_for_drift(by_drift, "abrupt")
    mcdd = abrupt.loc[abrupt["Config"] == "MCDD-S"].iloc[0]

    assert np.isclose(mcdd["FDR"], 0.20)
    assert np.isclose(mcdd["MDR"], 0.20)
    assert np.isclose(mcdd["IR"], 0.80)
    assert np.isclose(mcdd["Mean Delay"], 200.0)


def test_overall_table_averages_the_three_drift_tables(tmp_path: Path) -> None:
    summary_file = tmp_path / "summary_results.csv"
    _write_summary(summary_file)

    by_drift = average_metrics_by_drift(summary_file)
    overall = overall_article_table(average_metrics_overall(by_drift))
    mcdd = overall.loc[overall["Config"] == "MCDD-S"].iloc[0]

    assert np.isclose(mcdd["FDR"], 0.30)
    assert np.isclose(mcdd["MDR"], 0.10)
    assert np.isclose(mcdd["IR"], 0.70)
    assert np.isclose(mcdd["Mean Delay"], 300.0)


def test_kswin_article_na_convention_is_preserved(tmp_path: Path) -> None:
    summary_file = tmp_path / "summary_results.csv"
    _write_summary(summary_file)

    by_drift = average_metrics_by_drift(summary_file, article_na=True)
    abrupt = article_table_for_drift(by_drift, "abrupt")
    kswin = abrupt.loc[abrupt["Config"] == "KSWIN"].iloc[0]

    assert np.isclose(kswin["FDR"], 1.0)
    assert pd.isna(kswin["MDR"])
    assert pd.isna(kswin["IR"])
    assert pd.isna(kswin["Mean Delay"])

    formatted = format_article_table(abrupt, decimals=4)
    formatted_kswin = formatted.loc[
        formatted["Config"] == "KSWIN"
    ].iloc[0]
    assert formatted_kswin["FDR"] == "1.0000"
    assert formatted_kswin["MDR"] == "NA"
    assert formatted_kswin["IR"] == "NA"
    assert formatted_kswin["Mean Delay"] == "NA"
