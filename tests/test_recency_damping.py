# -*- coding: utf-8 -*-
"""A client served yesterday must not be ranked as if they were due.

The repurchase label asks "does this client buy this article within the next 3
months?". A client with 27 purchases on a 12-day cadence answers yes with ~91%
certainty -- and the ranking then placed them at #2 one day after their order,
under a panel headed "why this ranks up".

The model is not wrong; it was asked a 3-month question and the call list is a
today question. Nothing in the pipeline converted between the two: priority_score
and expected_value both multiply a probability by money, and no term anywhere
carried "already served".

Damping closes that gap without touching the model: the score still says who is
worth calling, the damping says whether it is worth calling THEM TODAY.
Synthetic frames only -- no dependency on data/.
"""
import math

import pandas as pd
import pytest

import article_prospects as ap


# --- the pure factor -------------------------------------------------------------

def test_a_client_exactly_at_their_rhythm_is_not_damped():
    """At ratio 1.0 the client is due. Due is the baseline, not a penalty."""
    assert ap.recency_damping(22, 22.0) == pytest.approx(1.0)


def test_a_client_long_past_their_rhythm_is_not_damped():
    """Overdue must never be boosted here -- lateness is already the model's
    strongest feature. Damping only ever removes priority it never adds."""
    assert ap.recency_damping(90, 22.0) == pytest.approx(1.0)


def test_a_client_halfway_through_their_cycle_is_damped_by_the_square():
    """Linear damping with a floor high enough to matter could not tell "served
    yesterday" from "a fifth of the way through": the floor bound first and
    flattened both to the same number. Squaring separates them."""
    assert ap.recency_damping(11, 22.0) == pytest.approx(0.25)


def test_a_client_served_yesterday_is_damped_to_the_floor():
    """The screenshot case: 1 day into a 12-day cadence, ratio 0.083."""
    assert ap.recency_damping(1, 12.0) == pytest.approx(0.05)


def test_a_client_served_today_is_damped_to_the_floor_not_to_zero():
    """A floor, not zero: a large account stays reachable further down the list
    rather than vanishing from a vendor's screen the day they order."""
    assert ap.recency_damping(0, 12.0) == pytest.approx(0.05)


def test_a_client_with_no_cadence_is_not_damped():
    """median_interval is NaN for a single purchase and snapshots.py keeps that
    distinct from zero on purpose. With no cadence there is no 'too soon' to
    measure, so the ranking must not invent one."""
    assert ap.recency_damping(1, float("nan")) == pytest.approx(1.0)
    assert ap.recency_damping(1, None) == pytest.approx(1.0)
    assert ap.recency_damping(1, 0.0) == pytest.approx(1.0)


def test_damping_never_leaves_the_closed_unit_interval():
    for days in (0, 1, 5, 22, 400):
        for interval in (1.0, 12.0, 22.0, 365.0):
            assert 0.05 <= ap.recency_damping(days, interval) <= 1.0


# --- block A, the article page ---------------------------------------------------

def make_fact():
    """V1 trades with RECENT and DUE; both bought article X."""
    return pd.DataFrame([
        {"client_id": "RECENT", "article_id": "X", "vendor_code": "V1", "ca": 500.0},
        {"client_id": "DUE", "article_id": "X", "vendor_code": "V1", "ca": 300.0},
    ])


def make_scores(recent_days, recent_interval, due_days, due_interval):
    """RECENT outranks DUE on raw expected value: higher score AND higher CA.
    Only recency can reorder them."""
    return pd.DataFrame([
        {"client_id": "RECENT", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 1000.0, "p_model": 0.91, "n_orders": 27,
         "days_since_last": recent_days, "median_interval": recent_interval},
        {"client_id": "DUE", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 600.0, "p_model": 0.60, "n_orders": 6,
         "days_since_last": due_days, "median_interval": due_interval},
    ])


def test_block1_ranks_the_recently_served_client_first_without_damping():
    """The defect, stated as the arithmetic that produces it: 0.91 x 1000 beats
    0.60 x 600, so the client who ordered yesterday leads the page."""
    scores = make_scores(1, 12.0, 80, 74.0)
    raw = (scores.set_index("client_id")["p_model"]
           * scores.set_index("client_id")["ca_sum"])
    assert raw["RECENT"] > raw["DUE"]


def test_block1_demotes_a_client_served_yesterday_below_a_client_who_is_due():
    """RECENT: 1 day into 12 -> x0.2 -> 910 becomes 182.
    DUE: 80 days into 74 -> x1.0 -> 360 unchanged. DUE now leads."""
    out = ap.existing_customers(make_scores(1, 12.0, 80, 74.0), make_fact(), "X", "V1")

    assert list(out["client_id"]) == ["DUE", "RECENT"]


def test_block1_keeps_the_recently_served_client_on_the_page():
    """Demoted, never dropped. The account is still the vendor's largest."""
    out = ap.existing_customers(make_scores(1, 12.0, 80, 74.0), make_fact(), "X", "V1")

    assert "RECENT" in set(out["client_id"])


def test_block1_leaves_the_order_alone_when_both_clients_are_due():
    """Damping must not reshuffle a list where nobody was served early."""
    out = ap.existing_customers(make_scores(20, 12.0, 80, 74.0), make_fact(), "X", "V1")

    assert list(out["client_id"]) == ["RECENT", "DUE"]


def test_block1_ranks_unchanged_when_the_scores_carry_no_cadence_columns():
    """A score table without evidence columns must still rank. Damping needs
    days_since_last and median_interval; absent them it is a no-op, not a crash."""
    scores = make_scores(1, 12.0, 80, 74.0).drop(
        columns=["days_since_last", "median_interval", "n_orders"])

    out = ap.existing_customers(scores, make_fact(), "X", "V1")

    assert list(out["client_id"]) == ["RECENT", "DUE"]


def test_block1_keeps_expected_value_as_real_money_not_a_blended_figure():
    """Rank by the damped figure, display the honest one.

    contact_queue.py already draws this line for priority_score: a vendor must be
    able to say the number out loud. 0.91 x 1000 x 0.2 = 182 is neither what the
    account is worth nor what it is expected to spend -- it corresponds to nothing
    physical. The demotion is explained by overdue_ratio, which the page already
    prints as "0,1x de son rythme"; it must not be smuggled into a money column.
    """
    out = ap.existing_customers(make_scores(1, 12.0, 80, 74.0), make_fact(), "X", "V1")
    recent = out.set_index("client_id").loc["RECENT"]

    assert recent["expected_value"] == pytest.approx(0.91 * 1000.0)


def test_a_nearly_due_client_is_barely_damped_while_a_just_served_one_is_not():
    """The property that picked the curve, stated as the comparison that decides it.

    A client at 90% of their cycle is a day or two early -- ringing them is fine and
    the ranking should hardly notice. A client at 8% was served yesterday. Linear
    damping treats these as 0.9 and 0.2: a 4.5x gap. Squared it is 0.81 and 0.05,
    a 16x gap, which is the difference the call list actually cares about.
    """
    nearly_due = ap.recency_damping(9, 10.0)
    just_served = ap.recency_damping(1, 12.0)

    assert nearly_due > 0.75
    assert just_served < 0.10
    assert nearly_due / just_served > 10


def test_block1_does_not_leak_its_internal_ranking_column():
    """The damped figure orders the list and stays out of the contract. Adding a
    column here would change every caller's schema for a number none of them show."""
    out = ap.existing_customers(make_scores(1, 12.0, 80, 74.0), make_fact(), "X", "V1")

    assert list(out.columns)[:4] == ["client_id", "p_model", "ca_sum", "expected_value"]
    assert not [c for c in out.columns if c.startswith("_")]
