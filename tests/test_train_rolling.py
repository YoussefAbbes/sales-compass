# -*- coding: utf-8 -*-
"""Guard the training contract, not the metric values."""
import pandas as pd

from snapshots import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_rolling import (
    FORBIDDEN_FEATURES, assert_no_current_state_features, early_stopping_fired,
)


def test_current_state_columns_are_never_trainable():
    """Stocks/Bloqué/Solde describe 2026-08-10 only. A model trained on them would
    learn the outcome it is meant to predict."""
    assert_no_current_state_features(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_the_guard_actually_fires():
    try:
        assert_no_current_state_features(NUMERIC_FEATURES + ["blocked"])
    except ValueError as exc:
        assert "blocked" in str(exc)
    else:
        raise AssertionError("the guard did not reject a current-state feature")


def test_forbidden_list_covers_every_master_snapshot_column():
    for col in ("blocked", "blocked_reason", "stock", "stock_class",
                "balance", "credit_limit"):
        assert col in FORBIDDEN_FEATURES


def test_early_stopping_did_not_fire_when_the_best_sits_near_the_cap():
    """The 2026-08-12 run: best iteration 5970 of a 6000 cap, patience 200. Only
    29 iterations followed the best, so training ended by exhausting the cap. The
    old predicate (best < cap - 1) called that converged and printed a claim of
    convergence into a jury-facing report."""
    assert early_stopping_fired(5970, 6000, 200) is False


def test_early_stopping_fired_when_patience_was_fully_spent():
    assert early_stopping_fired(5000, 6000, 200) is True


def test_the_boundary_where_patience_is_exactly_spent():
    # best 5799 -> iterations 5800..5999 are a full 200 non-improving rounds.
    assert early_stopping_fired(5799, 6000, 200) is True
    # best 5800 -> only 199 rounds observed before the cap cuts training off.
    assert early_stopping_fired(5800, 6000, 200) is False


# --- the ranking the delivery surfaces actually use ------------------------------
# precision_top30_model ranks by raw p, which no product surface does: block A and
# the call queue both damp for recency (article_prospects.recency_damping). The
# model card reported 61.9% for a ranking nobody serves, so a reader benchmarking
# the deliverable on it would see an unexplained regression.

import pytest

from train_rolling import damped_score


def test_damped_score_demotes_a_row_served_early_in_its_cycle():
    """The whole point: a near-certain repurchase that needs no call ranks below a
    less certain one that does."""
    frame = pd.DataFrame([
        {"p": 0.90, "days_since_last": 1,  "median_interval": 12.0},
        {"p": 0.50, "days_since_last": 80, "median_interval": 40.0},
    ])

    out = damped_score(frame)

    assert out.iloc[0] < out.iloc[1]


def test_damped_score_leaves_a_due_row_untouched():
    frame = pd.DataFrame([{"p": 0.9, "days_since_last": 40, "median_interval": 12.0}])

    assert damped_score(frame).iloc[0] == pytest.approx(0.9)


def test_damped_score_is_the_raw_score_when_the_frame_has_no_cadence():
    """The metric must not become a precondition on columns the trainer may not
    carry -- it degrades to the old number rather than raising."""
    frame = pd.DataFrame([{"p": 0.7}])

    assert damped_score(frame).iloc[0] == pytest.approx(0.7)


def test_damped_score_keeps_the_frame_s_index_so_it_can_be_assigned_back():
    frame = pd.DataFrame([{"p": 0.9, "days_since_last": 1, "median_interval": 12.0},
                          {"p": 0.4, "days_since_last": 90, "median_interval": 30.0}],
                         index=[7, 9])

    assert list(damped_score(frame).index) == [7, 9]
