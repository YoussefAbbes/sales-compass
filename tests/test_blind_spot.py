# -*- coding: utf-8 -*-
"""The blind-spot measurement is arithmetic on labels, so test it as arithmetic."""
import pandas as pd

from blind_spot import repurchase_rates


def tx(date, client, article, amount=100.0):
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "amount": amount, "is_return": False}


def test_a_september_repurchase_is_invisible_to_a_jan_may_label():
    """The whole claim in one test: the same pair counts as 0 under the v1 label
    and 1 under a full-year label."""
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),     # active in the feature window
        tx("2025-09-15", "C1", "A1"),     # repurchased outside Jan-May
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["n_pairs"] == 1
    assert r["rate_jan_may"] == 0.0
    assert r["rate_full_year"] == 1.0


def test_a_may_repurchase_is_visible_to_both():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),
        tx("2025-05-15", "C1", "A1"),
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["rate_jan_may"] == 1.0
    assert r["rate_full_year"] == 1.0


def test_understatement_is_the_ratio_of_the_two_rates():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"), tx("2025-03-01", "C1", "A1"),   # visible
        tx("2024-02-01", "C2", "A1"), tx("2025-10-01", "C2", "A1"),   # hidden
        tx("2024-02-01", "C3", "A1"),                                  # never rebought
        tx("2024-02-01", "C4", "A1"),                                  # never rebought
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["n_pairs"] == 4
    assert r["rate_jan_may"] == 0.25
    assert r["rate_full_year"] == 0.50
    assert r["understatement"] == 2.0


def test_returns_do_not_count_as_a_repurchase():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),
        {"date": pd.Timestamp("2025-09-01"), "client_id": "C1", "article_id": "A1",
         "amount": -100.0, "is_return": True},
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["rate_full_year"] == 0.0
