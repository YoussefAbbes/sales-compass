# -*- coding: utf-8 -*-
"""Unit tests for src/prospect_features.py. Synthetic frames only — no dependency on data/."""
import numpy as np
import pandas as pd
import pytest

import prospect_features as pf


def make_articles():
    return pd.DataFrame([
        {"article_id": "X", "enseigne": "MRQ01", "designation": "PLAQUETTE FREIN AV"},
        {"article_id": "S1", "enseigne": "MRQ01", "designation": "DISQUE FREIN AV"},
        {"article_id": "S2", "enseigne": "MRQ02", "designation": "PLAQUETTE FREIN AR"},
        {"article_id": "S3", "enseigne": "MRQ02", "designation": "  filtre  huile  "},
        {"article_id": "N", "enseigne": "MRQ03", "designation": None},
    ])


def test_family_is_the_upper_cased_first_token():
    assert pf.family_of("PLAQUETTE FREIN AV") == "PLAQUETTE"
    assert pf.family_of("  filtre  huile  ") == "FILTRE"


def test_family_of_missing_designation_is_unknown():
    assert pf.family_of(None) == pf.FAMILY_UNKNOWN
    assert pf.family_of(np.nan) == pf.FAMILY_UNKNOWN
    assert pf.family_of("   ") == pf.FAMILY_UNKNOWN


def test_family_map_covers_every_reference_row():
    fam = pf.family_map(make_articles())
    assert fam["X"] == "PLAQUETTE"
    assert fam["N"] == pf.FAMILY_UNKNOWN
    assert len(fam) == 5


def test_window_keeps_only_positive_rows_of_that_year():
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "C1", "article_id": "X",
         "month": pd.Timestamp("2024-03"), "ca": 100.0},
        {"vendor_code": "V1", "client_id": "C1", "article_id": "X",
         "month": pd.Timestamp("2025-03"), "ca": 100.0},
        {"vendor_code": "V1", "client_id": "C2", "article_id": "X",
         "month": pd.Timestamp("2024-04"), "ca": -50.0},
    ])
    out = pf.window(fact, 2024)
    assert list(out["client_id"]) == ["C1"]


def make_fact():
    """Synthetic two-window fact table: 2024 is the feature window, 2025 the label window.

    Articles : X (target), S1 S2 S3 (its signature), N (popular but unrelated)
    Clients  : B1..B4 bought X in 2024        -> already customers, not prospects
               P1 P2 P3 are V1 clients who never bought X in 2024
               P1 ADOPTS X in 2025 (the positive), P2 and P3 do not
               O1 is a V2-only client         -> out of V1's scope
               F1..F6 buy only N              -> the trap: popular article, no relation to X
    """
    rows = []

    def add(vendor, client, article, year, ca=100.0):
        rows.append({"vendor_code": vendor, "client_id": client, "article_id": article,
                     "month": pd.Timestamp(f"{year}-03"), "ca": ca})

    for c in ["B1", "B2", "B3", "B4"]:
        add("V1", c, "X", 2024)
    for c in ["B1", "B2", "B3", "B4", "P1", "P4"]:
        add("V1", c, "S1", 2024)
    add("V2", "O1", "S1", 2024)
    for c in ["B1", "B2", "B3", "P1", "P4"]:
        add("V1", c, "S2", 2024)
    for c in ["B1", "B2", "P1", "P2", "P4"]:
        add("V1", c, "S3", 2024)
    for c in ["B1", "F1", "F2", "F3", "F4", "F5", "F6"]:
        add("V1", c, "N", 2024)
    add("V1", "P3", "Z", 2024)
    add("V2", "P4", "X", 2024)              # P4 buys X through another vendor
    add("V1", "P1", "X", 2025)              # the adoption we want to predict
    add("V1", "B1", "X", 2025)              # a repeat, not an adoption
    return pd.DataFrame(rows)


def test_basket_matrix_marks_bought_pairs_only():
    win = pf.window(make_fact(), 2024)
    m, clients, articles = pf.basket_matrix(win)
    assert m.shape == (len(clients), len(articles))
    i, j = clients.get_loc("B1"), articles.get_loc("X")
    assert m[i, j] == 1
    assert m[clients.get_loc("P1"), j] == 0


def test_basket_matrix_is_binary_not_a_count():
    """Two months of the same pair must still read as one purchase."""
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "C1", "article_id": "A",
         "month": pd.Timestamp("2024-03"), "ca": 10.0},
        {"vendor_code": "V1", "client_id": "C1", "article_id": "A",
         "month": pd.Timestamp("2024-04"), "ca": 10.0},
    ])
    m, _, _ = pf.basket_matrix(pf.window(fact, 2024))
    assert m[0, 0] == 1


def test_client_similarity_is_symmetric_with_zero_diagonal():
    win = pf.window(make_fact(), 2024)
    m, clients, _ = pf.basket_matrix(win)
    sim = pf.client_similarity(m)
    assert sim.shape == (len(clients), len(clients))
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 0.0)


def test_similar_baskets_score_above_unrelated_ones():
    win = pf.window(make_fact(), 2024)
    m, clients, _ = pf.basket_matrix(win)
    sim = pf.client_similarity(m)
    p1, b1, f1 = clients.get_loc("P1"), clients.get_loc("B1"), clients.get_loc("F1")
    assert sim[p1, b1] > sim[p1, f1]


def test_client_features_are_all_behavioural():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.client_features(win, fam)
    assert list(out.columns) == pf.CLIENT_COLUMNS
    b1 = out.loc["B1"]
    assert b1["client_n_articles"] == 5           # X S1 S2 S3 N
    assert b1["client_n_vendors"] == 1
    assert b1["client_ca"] == pytest.approx(500.0)
    assert b1["client_mean_ca_per_article"] == pytest.approx(100.0)


def test_client_features_count_distinct_families():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.client_features(win, fam)
    # B1 holds PLAQUETTE(X), DISQUE(S1), PLAQUETTE(S2), FILTRE(S3), INCONNU(N) -> 4 distinct
    assert out.loc["B1"]["client_n_families"] == 4


def test_article_features_carry_the_categoricals():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.article_features(win, make_articles(), fam)
    assert list(out.columns) == pf.ARTICLE_COLUMNS
    assert out.loc["X"]["article_n_buyers"] == 5      # B1..B4 under V1, P4 under V2
    assert out.loc["X"]["enseigne"] == "MRQ01"
    assert out.loc["X"]["family"] == "PLAQUETTE"


def test_article_absent_from_the_reference_still_gets_features():
    """15,523 article ids in the fact table have no row in articles.csv at all."""
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.article_features(win, make_articles(), fam)
    assert out.loc["Z"]["family"] == pf.FAMILY_UNKNOWN
    assert out.loc["Z"]["enseigne"] == ""


def test_context_carries_every_precomputed_piece():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    assert len(ctx.win) > 0
    assert ctx.matrix.shape == (len(ctx.clients), len(ctx.articles))
    assert ctx.similarity.shape == (len(ctx.clients), len(ctx.clients))
    assert list(ctx.client_feats.columns) == pf.CLIENT_COLUMNS
    assert list(ctx.article_feats.columns) == pf.ARTICLE_COLUMNS


def test_context_holds_only_the_requested_window():
    """The 2025 rows must be invisible to a 2024 context — that is the leakage guarantee."""
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    assert ctx.win["month"].dt.year.unique().tolist() == [2024]
    assert "P1" in ctx.clients
    assert ctx.matrix[ctx.clients.get_loc("P1"), ctx.articles.get_loc("X")] == 0


def test_signature_with_lift_returns_values_not_just_ids():
    win = pf.window(make_fact(), 2024)
    sig = pf.signature_with_lift(win, "X")
    assert set(sig.index) == {"S1", "S2", "S3"}
    assert sig.iloc[0] == sig.max()                    # sorted descending
    assert sig["S1"] > sig["S3"]
    assert "N" not in sig.index                        # popular but lift < 1
    assert "X" not in sig.index                        # never its own signature


def test_signature_is_refused_below_the_minimum_buyer_count():
    win = pf.window(make_fact(), 2024)
    assert pf.signature_with_lift(win, "Z").empty


def test_affinity_counts_signature_overlap_and_weights_it_by_lift():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "P2", "P3"], sig)
    assert list(out.columns) == pf.AFFINITY_COLUMNS
    assert out.loc["P1"]["n_signature_held"] == 3
    assert out.loc["P2"]["n_signature_held"] == 1
    assert out.loc["P3"]["n_signature_held"] == 0
    assert out.loc["P1"]["share_signature_held"] == pytest.approx(1.0)
    assert out.loc["P1"]["lift_weighted_overlap"] > out.loc["P2"]["lift_weighted_overlap"]


def test_affinity_measures_brand_and_family_holdings():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "P3"], sig)
    # X is MRQ01/PLAQUETTE. P1 holds S1 (MRQ01) -> brand_n_held 1; S2 is PLAQUETTE -> family 1.
    assert out.loc["P1"]["brand_n_held"] == 1
    assert out.loc["P1"]["family_n_held"] == 1
    assert out.loc["P3"]["brand_n_held"] == 0
    assert out.loc["P3"]["family_n_held"] == 0


def test_affinity_knn_counts_buyers_among_nearest_neighbours():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "F1"], sig, k=3)
    assert out.loc["P1"]["knn_buyers"] > out.loc["F1"]["knn_buyers"]


def test_affinity_returns_a_typed_frame_for_clients_absent_from_the_window():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["GHOST"], sig)
    assert list(out.columns) == pf.AFFINITY_COLUMNS
    assert out.loc["GHOST"]["n_signature_held"] == 0


def test_candidates_exclude_everyone_who_already_holds_the_article():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    cands = pf.candidates_for(ctx, "X")
    assert "P1" in cands and "P2" in cands and "P3" in cands
    for holder in ["B1", "B2", "B3", "B4", "P4"]:
        assert holder not in cands


def test_build_features_returns_every_column_in_order():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X")
    assert list(out.columns) == pf.FEATURE_COLUMNS
    assert set(out.index) == set(pf.candidates_for(ctx, "X"))
    assert out.notna().all().all()


def test_build_features_categoricals_are_strings():
    """CatBoost rejects a categorical column that arrives as float or NaN."""
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X")
    for col in pf.CATEGORICAL_COLUMNS:
        assert out[col].map(type).eq(str).all()


def test_build_features_accepts_an_explicit_client_list():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X", client_ids=["P1"])
    assert list(out.index) == ["P1"]


def test_no_feature_can_see_the_label_window():
    """The leakage guard: a 2024 frame must be identical whether or not 2025 exists.

    If any feature reached into the label window, dropping those rows would change it — which
    is precisely the bug that produces a beautiful offline score and a useless model.
    """
    fact = make_fact()
    ctx_with = pf.build_context(fact, make_articles(), 2024)
    ctx_without = pf.build_context(fact[fact["month"].dt.year == 2024],
                                   make_articles(), 2024)
    a = pf.build_features(ctx_with, "X")
    b = pf.build_features(ctx_without, "X")
    pd.testing.assert_frame_equal(a, b)


# ---- Half-year windows ------------------------------------------------------
# The ledger ends 30 June 2026, so calendar years give a 12/12/6 split: the test
# label window would be HALF the length of the training one. That is a different
# question, not merely less data — a 6-month adoption rate is structurally lower
# than a 12-month one. Half-years make every window the same size.


def halfyear_fact():
    rows = []
    for y in (2024, 2025):
        for m in range(1, 13):
            rows.append({"vendor_code": "V1", "client_id": f"C{m}", "article_id": "A1",
                         "month": pd.Timestamp(year=y, month=m, day=1), "ca": 100.0})
    return pd.DataFrame(rows)


def test_a_half_year_window_keeps_only_its_own_six_months():
    win = pf.window(halfyear_fact(), "2024-H1")
    assert sorted(win["month"].dt.month.unique()) == [1, 2, 3, 4, 5, 6]
    assert win["month"].dt.year.unique().tolist() == [2024]


def test_the_second_half_is_july_to_december():
    win = pf.window(halfyear_fact(), "2024-H2")
    assert sorted(win["month"].dt.month.unique()) == [7, 8, 9, 10, 11, 12]


def test_half_years_do_not_bleed_across_years():
    win = pf.window(halfyear_fact(), "2025-H1")
    assert win["month"].dt.year.unique().tolist() == [2025]


def test_an_integer_period_still_means_the_whole_calendar_year():
    """Backwards compatible: every existing caller passes an int."""
    win = pf.window(halfyear_fact(), 2024)
    assert win["month"].dt.month.nunique() == 12


def test_a_half_year_window_still_drops_returns():
    frame = halfyear_fact()
    frame.loc[len(frame)] = {"vendor_code": "V1", "client_id": "CR", "article_id": "A1",
                             "month": pd.Timestamp("2024-02-01"), "ca": -50.0}
    win = pf.window(frame, "2024-H1")
    assert "CR" not in set(win["client_id"])


def test_latest_complete_half_is_h1_when_data_ends_in_june():
    """The ledger ends 30 June 2026, so 2026-H1 is complete and H2 has not begun.
    Serving features from a PARTIAL window would hand the model half the history
    it was trained on — train/serve skew, silent."""
    frame = pd.DataFrame([{"vendor_code": "V1", "client_id": "C1", "article_id": "A1",
                           "month": pd.Timestamp(2026, m, 1), "ca": 100.0}
                          for m in range(1, 7)])
    assert pf.latest_complete_half(frame) == "2026-H1"


def test_a_partial_second_half_does_not_count():
    """Data through September: H2 needs July-December, so the newest COMPLETE
    window is still H1."""
    frame = pd.DataFrame([{"vendor_code": "V1", "client_id": "C1", "article_id": "A1",
                           "month": pd.Timestamp(2026, m, 1), "ca": 100.0}
                          for m in range(1, 10)])
    assert pf.latest_complete_half(frame) == "2026-H1"


def test_a_full_year_yields_the_second_half():
    frame = pd.DataFrame([{"vendor_code": "V1", "client_id": "C1", "article_id": "A1",
                           "month": pd.Timestamp(2025, m, 1), "ca": 100.0}
                          for m in range(1, 13)])
    assert pf.latest_complete_half(frame) == "2025-H2"


def test_a_partial_first_half_falls_back_to_the_previous_year():
    frame = pd.DataFrame([{"vendor_code": "V1", "client_id": "C1", "article_id": "A1",
                           "month": pd.Timestamp(y, m, 1), "ca": 100.0}
                          for y, m in [(2025, 7), (2025, 12), (2026, 1), (2026, 2)]])
    assert pf.latest_complete_half(frame) == "2025-H2"
