# -*- coding: utf-8 -*-
"""The observation window must be derived from the data, never assumed.

baseline_lists.py was written when the extract only ever contained January-May,
so "5" and "10" were safe to hardcode: the score normaliser (ca x months / 10),
and the vendor-facing sentence "Commande 3 mois sur 5 en 2024".

The NAV ledger has 30 months. Regenerating ca_monthly.parquet from it without
these fixed would print "Commande 8 mois sur 5" to a vendor and silently rescale
every score. These tests pin the window to the data so that regeneration is safe.
"""
import pandas as pd

from baseline_lists import months_observed


def frame(*pairs):
    """(year, month) pairs -> a ca_monthly-shaped frame."""
    rows = [{"vendor_code": "V1", "client_id": "C1", "article_id": "A1",
             "month": pd.Timestamp(year=y, month=m, day=1), "ca": 100.0}
            for y, m in pairs]
    df = pd.DataFrame(rows)
    df["year"] = df["month"].dt.year
    df["mois"] = df["month"].dt.month
    return df


def test_the_january_to_may_extract_reports_five():
    """Today's data. The refactor must not change any existing output."""
    df = frame((2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5))
    assert months_observed(df, 2024) == 5


def test_a_full_year_reports_twelve():
    df = frame(*[(2025, m) for m in range(1, 13)])
    assert months_observed(df, 2025) == 12


def test_a_part_year_reports_what_is_actually_there():
    """2026 ends 30 June on the ledger, so its denominator is 6, not 12."""
    df = frame(*[(2026, m) for m in range(1, 7)])
    assert months_observed(df, 2026) == 6


def test_gaps_do_not_inflate_the_denominator():
    """A month with no trading at all is not an observed month."""
    df = frame((2024, 1), (2024, 3), (2024, 5))
    assert months_observed(df, 2024) == 3


def test_each_year_is_measured_separately():
    """Mixing years is what produced the fixed /10 normaliser."""
    df = frame((2024, 1), (2024, 2), *[(2025, m) for m in range(1, 13)])
    assert months_observed(df, 2024) == 2
    assert months_observed(df, 2025) == 12


def test_a_year_absent_from_the_data_reports_zero():
    df = frame((2024, 1))
    assert months_observed(df, 2030) == 0
