# -*- coding: utf-8 -*-
"""The validation page's data layer, on v4 artifacts.

The page used to read model_metrics.json and pair_scores_2026.csv — the v1
model, whose validation set we proved was contaminated (reports/tuning-log.md).
It now reads reports/rolling_metrics.json and data/pair_scores_rolling.csv.

Those two files have a different shape from v1's, so the reshaping is pinned
here rather than left to the display layer.
"""
import json

import pandas as pd
import pytest

import validation as v


@pytest.fixture
def metrics_file(scratch_dir, monkeypatch):
    """A rolling_metrics.json with the exact keys train_rolling.py writes."""
    reports = scratch_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "rolling_metrics.json").write_text(json.dumps({
        "lookback_months": 12, "horizon_months": 3,
        "splits": {
            "train": ["2025-01", "2025-02"],
            "validation": ["2025-09", "2025-10"],
            "embargo_train_val": ["2025-07", "2025-08"],
            "embargo_val_test": ["2025-11", "2025-12"],
            "test": ["2026-01", "2026-02"],
        },
        "rows": {"train": 2996823, "validation": 1022035, "test": 2051182},
        "iterations_cap": 6000, "early_stopping_rounds": 200,
        "best_iteration": 1345, "converged": True,
        "base_rate_test": 0.1137,
        "auc": {"train": 0.8110, "validation": 0.7709, "test": 0.7743},
        "auc_per_snapshot": {"2026-01": 0.7749, "2026-02": 0.7727},
        "precision_top30_model": 0.6186, "precision_top30_revenue": 0.3186,
        "feature_importance": {"client_days_since_last": 17.99},
    }), encoding="utf-8")
    monkeypatch.setattr(v, "REPORTS", reports)
    return reports


def test_metrics_expose_auc_for_each_split(metrics_file):
    m = v.load_metrics()
    assert m["train"]["auc"] == 0.8110
    assert m["validation"]["auc"] == 0.7709
    assert m["test"]["auc"] == 0.7743


def test_each_split_reports_its_window_and_size(metrics_file):
    m = v.load_metrics()
    assert m["train"]["n_pairs"] == 2996823
    assert "2025-01" in m["train"]["window"] and "2025-02" in m["train"]["window"]


def test_both_embargoes_are_exposed(metrics_file):
    """The train->val embargo is the whole point of v4. A validation page that
    hides it would be describing the design that failed."""
    m = v.load_metrics()
    assert m["embargo_train_val"]["window"]
    assert m["embargo_val_test"]["window"]


def test_the_validation_minus_test_gap_is_available(metrics_file):
    """The number that proves the leak is gone: validation is now slightly
    PESSIMISTIC about test, where the contaminated run was +0.0163 optimistic."""
    m = v.load_metrics()
    assert m["gap_validation_test"] == pytest.approx(0.7709 - 0.7743, abs=1e-9)


def test_convergence_is_reported_not_assumed(metrics_file):
    m = v.load_metrics()
    assert m["converged"] is True
    assert m["best_iteration"] == 1345
    assert m["iterations_cap"] == 6000


def test_scores_normalise_the_probability_column(scratch_dir, monkeypatch):
    """train_rolling writes `p`; the page has always spoken `p_model`. The
    rename belongs in the loader, not in eight display call sites."""
    data = scratch_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"client_id": ["0002"], "article_id": ["A1"],
                  "vendor_code": ["011"], "snapshot": ["2026-01-01"],
                  "amount_sum": [166.7], "p": [0.42], "label": [1]}).to_csv(
        data / "pair_scores_rolling.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(v, "SCORE_FILES", {"test": data / "pair_scores_rolling.csv"})

    out = v.load_scores("test")
    assert "p_model" in out.columns
    assert out.loc[0, "p_model"] == 0.42
    # queue_replay computes p_model * ca_sum; v4 calls that column amount_sum.
    assert out.loc[0, "ca_sum"] == 166.7
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "client_id"] == "0002"   # leading zero survives
