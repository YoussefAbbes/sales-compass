# -*- coding: utf-8 -*-
"""Window boundaries and leakage. Synthetic frames only.

These tests are the reason the rolling design can be trusted: the v1 model's two
documented dead-ends both came from a validation split that shared a year with
training, and the fix is only real if the boundaries are enforced by tests.
"""
import pandas as pd
import pytest

from snapshots import (
    EMBARGO_TV, EMBARGO_VT, HORIZON_MONTHS, LOOKBACK_MONTHS, TEST_T, TRAIN_T,
    VAL_T,
    build_snapshot_at,
)


def tx(date, client="C1", article="A1", amount=100.0, qty=1.0, vendor="V1",
       is_return=False, doc=None):
    """One synthetic ledger line. doc_no defaults to one document per date, so
    n_orders counts distinct documents exactly as it does on the real table."""
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "vendor_code": vendor, "enseigne": "FEBI", "amount": amount, "qty": qty,
            "cost": amount * 0.7, "is_return": is_return,
            "doc_no": doc or f"D{pd.Timestamp(date):%Y%m%d}"}


T = pd.Timestamp("2025-06-01")


def test_candidate_requires_a_purchase_inside_the_lookback_window():
    frame = pd.DataFrame([
        tx("2025-01-10", client="C1"),                 # inside  [2024-06, 2025-06)
        tx("2024-03-10", client="C2"),                 # older than the lookback
    ])
    out = build_snapshot_at(frame, T)
    assert out["client_id"].tolist() == ["C1"]


def test_label_comes_only_from_the_horizon_window():
    frame = pd.DataFrame([
        tx("2025-01-10"),                              # feature window
        tx("2025-07-10"),                              # inside  [2025-06, 2025-09)
    ])
    assert build_snapshot_at(frame, T).loc[0, "label"] == 1


def test_a_purchase_after_the_horizon_is_not_a_positive_label():
    frame = pd.DataFrame([tx("2025-01-10"), tx("2025-11-10")])
    assert build_snapshot_at(frame, T).loc[0, "label"] == 0


def test_a_purchase_on_the_snapshot_date_itself_is_a_label_not_a_feature():
    """T is the boundary: [T-lookback, T) for features, [T, T+horizon) for labels.
    A transaction exactly at T must land on the label side."""
    frame = pd.DataFrame([tx("2025-01-10"), tx(T)])
    out = build_snapshot_at(frame, T)
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "n_orders"] == 1, "the T transaction must not enter the features"


def test_returns_do_not_create_a_candidate_or_a_positive_label():
    frame = pd.DataFrame([
        tx("2025-01-10", client="C1", amount=-50.0, is_return=True),
    ])
    assert build_snapshot_at(frame, T).empty


def test_no_feature_column_reflects_data_at_or_after_T():
    """The leakage guard. Adding future transactions must not change any feature —
    only the label."""
    past = pd.DataFrame([tx("2025-01-10"), tx("2025-03-10")])
    future = pd.concat([past, pd.DataFrame([tx("2025-07-10"), tx("2025-08-10")])],
                       ignore_index=True)

    a = build_snapshot_at(past, T)
    b = build_snapshot_at(future, T)

    feature_cols = [c for c in a.columns if c not in ("label", "snapshot")]
    pd.testing.assert_frame_equal(a[feature_cols], b[feature_cols])
    assert a.loc[0, "label"] == 0 and b.loc[0, "label"] == 1


def test_snapshot_column_records_T():
    out = build_snapshot_at(pd.DataFrame([tx("2025-01-10")]), T)
    assert out.loc[0, "snapshot"] == T


def test_days_since_last_is_measured_against_T():
    frame = pd.DataFrame([tx("2025-05-02")])
    assert build_snapshot_at(frame, T).loc[0, "days_since_last"] == 30


def test_splits_are_chronological_and_do_not_overlap():
    assert max(TRAIN_T) < min(EMBARGO_TV)
    assert max(EMBARGO_TV) < min(VAL_T) < max(VAL_T) < min(EMBARGO_VT)
    assert max(EMBARGO_VT) < min(TEST_T)
    assert len(TRAIN_T) == 6 and len(VAL_T) == 2
    assert len(EMBARGO_TV) == 2 and len(EMBARGO_VT) == 2 and len(TEST_T) == 4


def test_the_sixteen_available_snapshots_are_all_accounted_for():
    """Data allows T = 2025-01 .. 2026-04. Every one is either used or knowingly
    dropped as embargo; none may be silently lost."""
    assert len(TRAIN_T) + len(EMBARGO_TV) + len(VAL_T)            + len(EMBARGO_VT) + len(TEST_T) == 16


def test_validation_to_test_gap_is_at_least_the_horizon():
    """Without it, a training snapshot's label window overlaps a test snapshot's
    feature window — the same leak the project has avoided everywhere else."""
    gap_months = (min(TEST_T).year - max(VAL_T).year) * 12 + \
                 (min(TEST_T).month - max(VAL_T).month)
    assert gap_months >= HORIZON_MONTHS


def test_lookback_and_horizon_match_the_agreed_design():
    assert LOOKBACK_MONTHS == 12
    assert HORIZON_MONTHS == 3


def test_train_to_validation_gap_is_at_least_the_horizon():
    """The val->test embargo has a twin, and it was missing.

    A training snapshot's label window runs HORIZON_MONTHS forward. Without the
    same gap in front of VAL_T, the last training labels and the validation
    labels are built from the SAME purchase events, so validation rewards
    memorisation instead of generalisation. That is what let a 6000-iteration
    run raise validation AUC to 0.788 while test AUC fell to 0.771 — the exact
    failure this module's docstring blames for the v1 dead-ends.
    """
    gap_months = (min(VAL_T).year - max(TRAIN_T).year) * 12 + \
                 (min(VAL_T).month - max(TRAIN_T).month)
    assert gap_months >= HORIZON_MONTHS
