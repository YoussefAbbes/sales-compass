# -*- coding: utf-8 -*-
"""Unit tests for src/dataset_profile.py — synthetic frames only."""
import pandas as pd
import pytest

import dataset_profile as dp


def _fact(rows):
    return pd.DataFrame(
        [{"vendor_code": v, "client_id": c, "article_id": a,
          "month": pd.Timestamp(year=y, month=m, day=1), "ca": ca}
         for v, c, a, y, m, ca in rows])


def _empty(col):
    return pd.DataFrame({col: pd.Series(dtype=str)})


def test_coverage_lists_every_observed_month_once():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 100.0),
                  ("V1", "C1", "A1", 2024, 2, 50.0),
                  ("V1", "C2", "A2", 2024, 2, 25.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": ["A1"], "designation": ["x"]}),
                   pd.DataFrame({"client_id": ["C1"]}),
                   pd.DataFrame({"vendor_code": ["V1"]}))
    months = [c["month"] for c in p["coverage"]]
    assert months == ["2024-01", "2024-02"]
    assert p["coverage"][1]["ca"] == pytest.approx(75.0)


def test_naming_gap_counts_unnamed_entities():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 10.0),
                  ("V1", "C2", "A2", 2024, 1, 10.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": ["A1"], "designation": ["x"]}),
                   pd.DataFrame({"client_id": ["C1"]}),
                   pd.DataFrame({"vendor_code": ["V1"]}))
    assert p["naming"]["articles"] == {"total": 2, "named": 1, "unnamed": 1}
    assert p["naming"]["clients"] == {"total": 2, "named": 1, "unnamed": 1}
    assert p["naming"]["vendors"] == {"total": 1, "named": 1, "unnamed": 0}


def test_sparsity_reports_single_month_pairs():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 10.0),
                  ("V1", "C1", "A1", 2024, 2, 10.0),
                  ("V1", "C2", "A2", 2024, 1, 10.0)])
    p = dp.profile(fact, _empty("article_id"), _empty("client_id"), _empty("vendor_code"))
    assert p["sparsity"]["n_pairs"] == 2
    assert p["sparsity"]["single_month_pairs"] == 1
    assert p["sparsity"]["dense_grid"] == 4  # 2 clients x 2 articles


def test_returns_are_counted_separately():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 100.0),
                  ("V1", "C1", "A2", 2024, 1, -20.0)])
    p = dp.profile(fact, _empty("article_id"), _empty("client_id"), _empty("vendor_code"))
    assert p["returns"]["n_rows"] == 1
    assert p["returns"]["total_ca"] == pytest.approx(-20.0)
    assert p["returns"]["share"] == pytest.approx(0.5)


def test_entities_count_distinct_values():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 10.0),
                  ("V2", "C1", "A2", 2024, 2, 10.0),
                  ("V1", "C2", "A1", 2025, 1, 10.0)])
    p = dp.profile(fact, _empty("article_id"), _empty("client_id"), _empty("vendor_code"))
    e = p["entities"]
    assert (e["rows"], e["vendors"], e["clients"], e["articles"], e["months"]) == (3, 2, 2, 2, 3)
    assert e["total_ca"] == pytest.approx(30.0)
