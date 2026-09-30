# -*- coding: utf-8 -*-
"""Unit tests for src/validation.py. Synthetic frames only — no dependency on data/."""
import numpy as np
import pandas as pd
import pytest

import validation as v


def test_calibration_bins_preserve_row_count():
    p = pd.Series(np.linspace(0.01, 0.99, 500))
    y = pd.Series((np.linspace(0.01, 0.99, 500) > 0.5).astype(int))
    tab = v.calibration_table(p, y, n_bins=10)
    assert tab["n"].sum() == 500
    assert len(tab) <= 10


def test_calibration_is_perfect_when_predictions_match_reality():
    """500 pairs at p=0.2 of which exactly 20% are positive -> gap ~0."""
    p = pd.Series([0.2] * 500)
    y = pd.Series([1] * 100 + [0] * 400)
    tab = v.calibration_table(p, y, n_bins=10)
    row = tab.iloc[0]
    assert row["observed"] == pytest.approx(0.2)
    assert row["gap"] == pytest.approx(0.0, abs=1e-9)


def test_decile_table_is_monotonic_on_a_perfect_ranking():
    p = pd.Series(np.linspace(0, 1, 1000))
    y = (p > 0.5).astype(int)
    tab = v.decile_table(p, y)
    assert len(tab) == 10
    assert tab["rate"].is_monotonic_increasing
    assert tab.iloc[-1]["rate"] == 1.0
    assert tab.iloc[0]["rate"] == 0.0


def test_confusion_at_threshold_zero_flags_everything():
    p = pd.Series([0.1, 0.4, 0.9])
    y = pd.Series([0, 1, 1])
    c = v.confusion_at(p, y, 0.0)
    assert c["n_flagged"] == 3
    assert c["tp"] == 2 and c["fp"] == 1
    assert c["tn"] == 0 and c["fn"] == 0


def test_confusion_at_threshold_above_max_flags_nothing():
    p = pd.Series([0.1, 0.4, 0.9])
    y = pd.Series([0, 1, 1])
    c = v.confusion_at(p, y, 1.0)
    assert c["n_flagged"] == 0
    assert c["tp"] == 0 and c["fp"] == 0
    assert c["fn"] == 2 and c["tn"] == 1
    assert c["precision"] == 0.0


def test_confusion_precision_and_recall_are_correct():
    p = pd.Series([0.9, 0.8, 0.7, 0.2])
    y = pd.Series([1, 0, 1, 1])
    c = v.confusion_at(p, y, 0.5)
    assert c["tp"] == 2 and c["fp"] == 1 and c["fn"] == 1 and c["tn"] == 0
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)


def _fact(rows):
    """Build a minimal ca_monthly-shaped frame. rows = (client, article, year, month, ca)."""
    return pd.DataFrame(
        [{"client_id": c, "article_id": a,
          "month": pd.Timestamp(year=y, month=m, day=1), "ca": ca,
          "vendor_code": "007"} for c, a, y, m, ca in rows])


def test_eligible_pairs_requires_both_years():
    fact = _fact([
        ("C1", "A1", 2024, 3, 100.0),   # both years   -> eligible
        ("C1", "A1", 2025, 3, 120.0),
        ("C2", "A2", 2025, 2, 300.0),   # 2025 only    -> not eligible
        ("C3", "A3", 2024, 1, 50.0),    # 2024 only    -> not eligible
        ("C4", "A4", 2024, 1, -10.0),   # return only  -> not eligible
        ("C4", "A4", 2025, 1, 80.0),
    ])
    el = v.eligible_pairs(fact, (2024, 2025))
    got = set(zip(el["client_id"], el["article_id"]))
    assert got == {("C1", "A1")}


def test_queue_replay_ranks_by_expected_value_and_caps_at_top_k():
    scores = pd.DataFrame({
        "client_id": ["C1"] * 4,
        "article_id": ["A1", "A2", "A3", "A4"],
        "vendor_code": ["007"] * 4,
        "ca_sum": [100.0, 1000.0, 10.0, 500.0],
        "p_model": [0.9, 0.5, 0.99, 0.2],   # EV: 90, 500, 9.9, 100
        "label": [1, 1, 0, 0],
    })
    eligible = scores[["client_id", "article_id"]]
    out = v.queue_replay(scores, eligible, top_k=2)
    assert list(out["article_id"]) == ["A2", "A4"]
    assert list(out["rank_in_client"]) == [1, 2]
    assert out.iloc[0]["valeur_attendue"] == pytest.approx(500.0)


def test_queue_replay_drops_pairs_outside_the_eligible_pool():
    scores = pd.DataFrame({
        "client_id": ["C1", "C1"],
        "article_id": ["A1", "A2"],
        "vendor_code": ["007", "007"],
        "ca_sum": [100.0, 1000.0],
        "p_model": [0.9, 0.9],
        "label": [1, 1],
    })
    eligible = pd.DataFrame({"client_id": ["C1"], "article_id": ["A1"]})
    out = v.queue_replay(scores, eligible, top_k=3)
    assert list(out["article_id"]) == ["A1"]


def test_precision_at_n_counts_hits_in_the_top_n_per_group():
    df = pd.DataFrame({
        "vendor_code": ["V1"] * 4 + ["V2"] * 4,
        "p_model": [0.9, 0.8, 0.2, 0.1] + [0.9, 0.8, 0.2, 0.1],
        "label":   [1,   0,   1,   1]  + [1,   1,   0,   0],
    })
    r = v.precision_at_n(df, "p_model", n=2)
    assert r["n_pairs"] == 4          # 2 per vendor
    assert r["n_hits"] == 3           # V1: 1 hit, V2: 2 hits
    assert r["precision"] == pytest.approx(0.75)
    assert r["n_groups"] == 2


def test_precision_at_n_handles_groups_smaller_than_n():
    df = pd.DataFrame({
        "vendor_code": ["V1", "V2"],
        "p_model": [0.9, 0.4],
        "label": [1, 0],
    })
    r = v.precision_at_n(df, "p_model", n=30)
    assert r["n_pairs"] == 2
    assert r["precision"] == pytest.approx(0.5)


def test_realized_ca_sums_only_the_requested_year_and_positive_months():
    fact = _fact([
        ("C1", "A1", 2026, 1, 100.0),
        ("C1", "A1", 2026, 2, 50.0),
        ("C1", "A1", 2025, 1, 999.0),   # wrong year -> excluded
        ("C2", "A2", 2026, 1, -20.0),   # net return -> excluded
    ])
    r = v.realized_ca(fact, 2026)
    assert len(r) == 1
    assert r.iloc[0]["ca_realized"] == pytest.approx(150.0)


def test_apply_calibration_returns_input_unchanged_when_absent():
    p = pd.Series([0.1, 0.5, 0.9])
    out = v.apply_calibration(p, None)
    assert list(out) == pytest.approx([0.1, 0.5, 0.9])
    assert list(v.apply_calibration(p, {})) == pytest.approx([0.1, 0.5, 0.9])


def test_apply_calibration_maps_through_the_curve():
    """A curve that halves every probability must halve them."""
    cal = {"x": [0.0, 1.0], "y": [0.0, 0.5]}
    out = v.apply_calibration([0.0, 0.4, 1.0], cal)
    assert list(out) == pytest.approx([0.0, 0.2, 0.5])


def test_apply_calibration_clips_outside_the_fitted_range():
    cal = {"x": [0.2, 0.8], "y": [0.1, 0.6]}
    out = v.apply_calibration([0.0, 1.0], cal)
    assert out[0] == pytest.approx(0.1)   # below range -> first value
    assert out[1] == pytest.approx(0.6)   # above range -> last value


def test_apply_calibration_preserves_order():
    """Isotonic is monotonic: calibration must never reorder pairs."""
    cal = {"x": [0.0, 0.3, 0.7, 1.0], "y": [0.0, 0.1, 0.45, 0.8]}
    raw = [0.05, 0.9, 0.4, 0.65, 0.2]
    out = list(v.apply_calibration(raw, cal))
    assert [i for i, _ in sorted(enumerate(raw), key=lambda t: t[1])] == \
           [i for i, _ in sorted(enumerate(out), key=lambda t: t[1])]
