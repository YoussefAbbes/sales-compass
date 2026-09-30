# -*- coding: utf-8 -*-
"""Aggregating the ledger to the v1 monthly grain must be exact and lossless."""
import pandas as pd

from reconcile import reconcile_monthly


def tx_row(date, amount, client="0289", article="01381", vendor="007"):
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "vendor_code": vendor, "amount": amount, "is_return": amount < 0}


def test_lines_in_the_same_month_are_summed():
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-01-20", 50.0)])
    out = reconcile_monthly(tx)
    assert len(out) == 1
    assert out.loc[0, "ca"] == 150.0
    assert out.loc[0, "month"] == pd.Timestamp("2024-01-01")


def test_returns_are_netted_in_matching_v1_semantics():
    """v1 kept negative CA in the fact table because the source system netted
    returns into revenue. Reconciliation must reproduce that, not improve on it."""
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-01-15", -30.0)])
    out = reconcile_monthly(tx)
    assert out.loc[0, "ca"] == 70.0


def test_distinct_months_stay_separate():
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-02-02", 40.0)])
    out = reconcile_monthly(tx).sort_values("month").reset_index(drop=True)
    assert out["ca"].tolist() == [100.0, 40.0]
