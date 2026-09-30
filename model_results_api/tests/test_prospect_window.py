"""The adoption context must be a complete half-year, not a calendar year.

`prospect_model.cbm` is trained on six-month windows. Serving it a twelve-month
context is train/serve skew: the client-level counts it reads (articles held,
basket size, similarity) all scale with window length, so a twelve-month context
inflates every one of them relative to what the model learned.

The bug is invisible on today's data — the ledger stops at June 2026, so calendar
2026 and 2026-H1 are the same six months. It appears the moment the ledger extends
past June, which is why this test extends it rather than trusting the fixture.
"""
import pandas as pd

from model_results_api.app import repository as repo


def _ledger_extended_into_h2() -> pd.DataFrame:
    """The fixture ledger plus a July-December echo of its first half."""
    facts = repo.fact()
    h1 = facts[(facts["month"].dt.year == 2026) & (facts["month"].dt.month <= 6)].copy()
    h2 = h1.copy()
    h2["month"] = h2["month"] + pd.DateOffset(months=6)
    return pd.concat([facts, h2], ignore_index=True)


def test_prospect_context_window_is_one_complete_half_year(monkeypatch):
    extended = _ledger_extended_into_h2()
    assert extended[extended["month"].dt.year == 2026]["month"].dt.month.nunique() == 12, (
        "the fixture setup must produce a full calendar 2026 for this test to mean anything"
    )

    monkeypatch.setattr(repo, "fact", lambda: extended)
    repo.prospect_context.cache_clear()
    try:
        window = repo.prospect_context().win
        months = sorted(window["month"].dt.month.unique())
        assert len(months) == 6, f"expected a six-month window, got {len(months)}: {months}"
        assert months == [7, 8, 9, 10, 11, 12], (
            f"expected the latest complete half (2026-H2), got months {months}"
        )
    finally:
        repo.prospect_context.cache_clear()
