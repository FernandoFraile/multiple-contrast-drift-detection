"""Tests for article-style result aggregation and formatting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mcdd.experiments import (
    article_table_for_drift,
    average_metrics_by_drift,
    average_metrics_overall,
    build_article_tables,
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
                    "MDR": np.nan,
                    "IR": 0.0,
                    "mean_delay": np.nan,
                }
            )

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

    expected_fdr = np.mean([0.20, 0.30, 0.40])
    expected_mdr = np.mean([0.20, 0.10, 0.10])
    expected_ir = np.mean([0.80, 0.70, 0.60])
    expected_mean_delay = np.mean([200.0, 300.0, 400.0])

    assert np.isclose(mcdd["FDR"], expected_fdr)
    assert np.isclose(mcdd["MDR"], expected_mdr)
    assert np.isclose(mcdd["IR"], expected_ir)
    assert np.isclose(mcdd["Mean Delay"], expected_mean_delay)


def test_undefined_metrics_are_preserved_in_article_reporting(
    tmp_path: Path,
) -> None:
    summary_file = tmp_path / "summary_results.csv"
    _write_summary(summary_file)

    by_drift = average_metrics_by_drift(summary_file, article_na=True)
    abrupt = article_table_for_drift(by_drift, "abrupt")
    kswin = abrupt.loc[abrupt["Config"] == "KSWIN"].iloc[0]

    assert np.isclose(kswin["FDR"], 1.0)
    assert pd.isna(kswin["MDR"])
    assert np.isclose(kswin["IR"], 0.0)
    assert pd.isna(kswin["Mean Delay"])

    formatted = format_article_table(abrupt, decimals=4)
    formatted_kswin = formatted.loc[
        formatted["Config"] == "KSWIN"
    ].iloc[0]
    assert formatted_kswin["FDR"] == "1.0000"
    assert formatted_kswin["MDR"] == "NA"
    assert formatted_kswin["IR"] == "0.0000"
    assert formatted_kswin["Mean Delay"] == "NA"


def test_nan_in_one_distribution_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    summary_file = tmp_path / "summary_results.csv"
    _write_summary(summary_file)
    results = pd.read_csv(summary_file)

    mask = (
        (results["configuration"] == "MCDD-S")
        & (results["drift_type"] == "abrupt")
        & (results["distribution"] == "gamma")
    )
    results.loc[mask, "MDR"] = np.nan
    results.to_csv(summary_file, index=False)

    by_drift = average_metrics_by_drift(summary_file, article_na=True)
    abrupt = article_table_for_drift(by_drift, "abrupt")
    mcdd = abrupt.loc[abrupt["Config"] == "MCDD-S"].iloc[0]

    assert pd.isna(mcdd["MDR"])


def test_partial_abrupt_results_can_be_reported(tmp_path: Path) -> None:
    full_summary = tmp_path / "full_summary.csv"
    partial_summary = tmp_path / "summary_results.csv"
    _write_summary(full_summary)

    results = pd.read_csv(full_summary)
    abrupt_only = results[
        (results["drift_type"] == "abrupt")
        | (results["drift_type"] == "all")
    ]
    abrupt_only.to_csv(partial_summary, index=False)

    tables = build_article_tables(
        partial_summary,
        require_all_distributions=False,
        require_all_drifts=False,
    )

    assert not tables["abrupt"].empty
    assert tables["gradual"].empty
    assert tables["incremental"].empty
    assert not tables["overall"].empty

    abrupt_mcdd = tables["abrupt"].loc[
        tables["abrupt"]["Config"] == "MCDD-S"
    ].iloc[0]
    overall_mcdd = tables["overall"].loc[
        tables["overall"]["Config"] == "MCDD-S"
    ].iloc[0]

    assert np.isclose(overall_mcdd["FDR"], abrupt_mcdd["FDR"])
    assert np.isclose(overall_mcdd["MDR"], abrupt_mcdd["MDR"])
    assert np.isclose(overall_mcdd["IR"], abrupt_mcdd["IR"])
    assert np.isclose(overall_mcdd["Mean Delay"], abrupt_mcdd["Mean Delay"])


def test_partial_distribution_results_can_be_reported(tmp_path: Path) -> None:
    full_summary = tmp_path / "full_summary.csv"
    partial_summary = tmp_path / "summary_results.csv"
    _write_summary(full_summary)

    results = pd.read_csv(full_summary)
    partial = results[
        (results["drift_type"] == "abrupt")
        & (results["distribution"] == "normal")
    ]
    partial.to_csv(partial_summary, index=False)

    by_drift = average_metrics_by_drift(
        partial_summary,
        require_all_distributions=False,
    )
    abrupt = article_table_for_drift(by_drift, "abrupt")
    mcdd = abrupt.loc[abrupt["Config"] == "MCDD-S"].iloc[0]

    assert np.isclose(mcdd["FDR"], 0.10)
    assert np.isclose(mcdd["MDR"], 0.30)
    assert np.isclose(mcdd["IR"], 0.90)
    assert np.isclose(mcdd["Mean Delay"], 100.0)
