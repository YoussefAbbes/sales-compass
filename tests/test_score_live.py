# -*- coding: utf-8 -*-
"""Live scoring contract. Synthetic frames only.

A live snapshot predicts a window that has not happened yet. The two things
that can go wrong silently are scoring the wrong date and shipping the
all-zero placeholder labels as if they were outcomes; both are pinned here.
"""
import numpy as np
import pandas as pd
import pytest

from score_live import live_snapshot_date, score_live


def frame(*dates):
    return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates]})


def test_live_date_is_the_month_after_the_last_transaction():
    """Data ends 2026-06-30, so the first snapshot with a complete feature
    window strictly before T is 2026-07-01."""
    assert live_snapshot_date(frame("2026-01-05", "2026-06-30")) \
        == pd.Timestamp("2026-07-01")


def test_a_last_date_mid_month_still_rolls_to_the_next_month():
    assert live_snapshot_date(frame("2026-07-15")) == pd.Timestamp("2026-08-01")


def test_a_last_date_on_the_first_of_a_month_rolls_forward_not_sideways():
    """2026-07-01 means July has data, so T must be August. Returning July
    would put the last day of data inside the feature window's own future."""
    assert live_snapshot_date(frame("2026-07-01")) == pd.Timestamp("2026-08-01")


class StubModel:
    """Stands in for CatBoostClassifier. score_live only needs predict_proba,
    so the test does not pay for training a real model."""

    def __init__(self, p=0.42):
        self.p = p
        self.seen_columns = None

    def predict_proba(self, X):
        self.seen_columns = list(X.columns)
        return np.column_stack([np.full(len(X), 1 - self.p),
                                np.full(len(X), self.p)])


def ledger(*rows):
    """Synthetic ledger lines, same shape as tests/test_snapshots.py::tx."""
    out = []
    for date, client, article in rows:
        out.append({"date": pd.Timestamp(date), "client_id": client,
                    "article_id": article, "vendor_code": "V1",
                    "enseigne": "FEBI", "amount": 100.0, "qty": 1.0,
                    "cost": 70.0, "is_return": False,
                    "doc_no": f"D{pd.Timestamp(date):%Y%m%d}{client}"})
    return pd.DataFrame(out)


LIVE_T = pd.Timestamp("2026-07-01")


def test_live_scores_carry_no_label_column():
    """build_snapshot_at derives labels from the horizon window. For a future T
    that window is empty, so every label would be 0 — the absence of an
    observation, not an observation of absence. Shipping it invites someone to
    compute an accuracy of zero against it."""
    out = score_live(ledger(("2026-03-10", "C1", "A1")), StubModel(), LIVE_T)
    assert "label" not in out.columns


def test_live_scores_have_exactly_the_expected_columns():
    """The queue needs more than the probability. amount_sum is the pair's own
    revenue ON THE LEDGER, which is what lets expected value be computed without
    borrowing ca_2025 from the older extract basis; the recency and cadence
    columns back the rule-based reason text shown to vendors."""
    out = score_live(ledger(("2026-03-10", "C1", "A1")), StubModel(), LIVE_T)
    assert list(out.columns) == [
        "client_id", "article_id", "vendor_code", "p",
        "amount_sum", "days_since_last", "median_interval", "n_orders",
    ]


def test_context_columns_come_from_the_snapshot_not_the_model():
    """A regression guard: these must be the snapshot's own values, so a pair
    with a single 100 TND purchase reports amount_sum 100, not something the
    model invented."""
    out = score_live(ledger(("2026-03-10", "C1", "A1")), StubModel(), LIVE_T)
    assert out.loc[0, "amount_sum"] == 100.0
    assert out.loc[0, "n_orders"] == 1


def test_one_row_per_candidate_pair_with_p_in_range():
    out = score_live(
        ledger(("2026-03-10", "C1", "A1"), ("2026-04-10", "C1", "A2")),
        StubModel(p=0.42), LIVE_T)
    assert len(out) == 2
    assert set(zip(out["client_id"], out["article_id"])) == {("C1", "A1"), ("C1", "A2")}
    assert out["p"].between(0, 1).all()


def test_the_model_is_given_the_canonical_feature_set():
    """Guards against scoring on a different column order or set than training
    used — the failure mode that produces confident nonsense."""
    from snapshots import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    model = StubModel()
    score_live(ledger(("2026-03-10", "C1", "A1")), model, LIVE_T)
    assert model.seen_columns == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_a_snapshot_missing_model_features_is_rejected_by_name():
    class Broken:
        def predict_proba(self, X):
            raise AssertionError("must not be reached")

    import score_live as mod
    original = mod.build_snapshot_at
    mod.build_snapshot_at = lambda tx, T: pd.DataFrame(
        {"client_id": ["C1"], "article_id": ["A1"], "vendor_code": ["V1"]})
    try:
        with pytest.raises(ValueError, match="n_orders"):
            score_live(ledger(("2026-03-10", "C1", "A1")), Broken(), LIVE_T)
    finally:
        mod.build_snapshot_at = original


# --- the serving-window record -------------------------------------------------
# The score file alone cannot say which window it answers for: T is derived at
# write time and nothing in the CSV records it. Without this the API has to guess,
# and it guessed from the TRAINING metrics — which is how it came to describe live
# recommendations with the backtest's snapshot span.


def test_live_window_record_ends_on_the_last_day_inside_the_horizon():
    """The label window is [T, T + HORIZON), half-open. A vendor reads the end
    date as "act before this", so it must be the last day the prediction still
    covers, not the first day it no longer does."""
    from score_live import live_window_record

    rec = live_window_record(LIVE_T, n_pairs=1, generated_at="2026-08-24T16:00:00")

    assert rec["snapshot_date"] == "2026-07-01"
    assert rec["window_end"] == "2026-09-30"


def test_live_window_record_reports_the_shared_horizon_constant():
    """score_live.main() had its own literal months=3 alongside snapshots'
    HORIZON_MONTHS. Two copies of one number drift; the record must report the
    constant the snapshot was actually built with."""
    from snapshots import HORIZON_MONTHS
    from score_live import live_window_record

    rec = live_window_record(LIVE_T, n_pairs=1, generated_at="2026-08-24T16:00:00")

    assert rec["horizon_months"] == HORIZON_MONTHS


def test_live_window_record_carries_the_pair_count_and_write_time():
    """Both are provenance, not decoration: a pair count that disagrees with the
    CSV means the two artifacts were shipped from different runs."""
    from score_live import live_window_record

    rec = live_window_record(LIVE_T, n_pairs=518278, generated_at="2026-08-24T16:00:00")

    assert rec["n_pairs"] == 518278
    assert rec["generated_at"] == "2026-08-24T16:00:00"
