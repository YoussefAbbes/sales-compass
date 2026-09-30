"""Year-on-year brand evolution must compare equal month spans.

The v1 extract held January-May of every year, so summing two calendar years
compared like with like. The NAV ledger is continuous and stops in June 2026, so
the same sum now weighs twelve months of 2025 against six of 2026 and reports a
~50% collapse for every brand on the dashboard.
"""
import pandas as pd
import pytest

from model_results_api.app import services


def _flat_brand_ledger() -> pd.DataFrame:
    """One brand billing exactly 100 TND a month, all of 2025 and half of 2026.

    Like-for-like evolution is therefore 0%: the months that exist in both years
    are identical. Any other answer is a span artefact.
    """
    months = list(pd.date_range("2025-01-01", "2025-12-01", freq="MS"))
    months += list(pd.date_range("2026-01-01", "2026-06-01", freq="MS"))
    return pd.DataFrame([
        {"brand": "ACME", "month": month, "ca": 100.0,
         "client_id": "C1", "article_id": "A1", "designation": "PIECE"}
        for month in months
    ])


def test_brand_evolution_compares_equal_month_spans():
    summary = services._brand_summaries(_flat_brand_ledger())[0]
    assert summary.evolution_pct == pytest.approx(0.0), (
        "a flat 100 TND/month brand must show 0% evolution; a non-zero value means "
        "twelve months of 2025 are being compared against six of 2026"
    )
