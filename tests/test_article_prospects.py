# -*- coding: utf-8 -*-
"""Unit tests for src/article_prospects.py. Synthetic frames only — no dependency on data/."""
import pandas as pd
import pytest

import article_prospects as ap


def make_articles():
    return pd.DataFrame([
        {"article_id": "X1", "enseigne": "MRQ01", "designation": "HUILE MOTEUR 5W40 5L"},
        {"article_id": "X2", "enseigne": "MRQ02", "designation": "HUILE ATF 1L"},
        {"article_id": "9931", "enseigne": "MRQ01", "designation": "PLAQUETTE FREIN AV"},
        {"article_id": "7712", "enseigne": "MRQ03", "designation": None},
    ])


def test_search_matches_designation_case_insensitively():
    out = ap.search_articles(make_articles(), "huile")
    assert set(out["article_id"]) == {"X1", "X2"}


def test_search_matches_reference_so_unnamed_articles_stay_reachable():
    out = ap.search_articles(make_articles(), "7712")
    assert list(out["article_id"]) == ["7712"]
    assert "(désignation inconnue)" in out.iloc[0]["label"]


def test_search_matches_brand():
    out = ap.search_articles(make_articles(), "mrq01")
    assert set(out["article_id"]) == {"X1", "9931"}


def test_search_respects_the_limit():
    out = ap.search_articles(make_articles(), "", limit=2)
    assert len(out) == 2


def test_blank_query_returns_the_expected_columns():
    out = ap.search_articles(make_articles(), "   ")
    assert list(out.columns) == ["article_id", "enseigne", "designation", "label"]


def make_fact():
    """Synthetic monthly-CA table.

    Articles : X (target), S1 S2 S3 (its signature), N (noise)
    Clients  : B1..B4 buy X under V1  |  P1 P2 P3 are V1 clients who never bought X
               P4 is a V1 client who buys X under ANOTHER vendor (V2)
               O1 is a V2-only client  |  F1..F6 buy only N
    """
    rows = []

    def add(vendor, client, article, ca=100.0, month="2025-03"):
        rows.append({"vendor_code": vendor, "client_id": client, "article_id": article,
                     "month": pd.Timestamp(month), "ca": ca})

    for c in ["B1", "B2", "B3", "B4"]:
        add("V1", c, "X")
    add("V2", "P4", "X")                       # P4 buys X elsewhere
    for c in ["B1", "B2", "B3", "B4", "P1", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S1")
    for c in ["B1", "B2", "B3", "P1", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S2")
    for c in ["B1", "B2", "P1", "P2", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S3")
    for c in ["B1", "F1", "F2", "F3", "F4", "F5", "F6"]:
        add("V1", c, "N")
    add("V1", "P3", "Z")                       # V1 client with no signature overlap
    add("V1", "B1", "X", ca=-50.0)             # a return: must not create a buyer
    return pd.DataFrame(rows)


def test_vendor_clients_uses_the_fact_table_not_the_scores():
    v1 = ap.vendor_clients(make_fact(), "V1")
    assert {"B1", "B2", "B3", "B4", "P1", "P2", "P3", "P4"} <= v1
    assert "O1" not in v1


def test_buyers_of_ignores_vendor_so_a_client_buying_elsewhere_still_counts():
    buyers = ap.buyers_of(make_fact(), "X")
    assert buyers == {"B1", "B2", "B3", "B4", "P4"}


def test_negative_ca_alone_does_not_make_a_buyer():
    fact = pd.DataFrame([{"vendor_code": "V1", "client_id": "R1", "article_id": "X",
                          "month": pd.Timestamp("2025-03"), "ca": -80.0}])
    assert ap.buyers_of(fact, "X") == set()
    assert ap.vendor_clients(fact, "V1") == set()


def make_scores():
    """pair_scores_2026.csv shape. Note B4's vendor_code says V2 while the fact table
    says V1 owns B4 — the ownership rule must follow the fact table."""
    return pd.DataFrame([
        {"client_id": "B1", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 100.0, "p_model": 0.90},
        {"client_id": "B2", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 1000.0, "p_model": 0.50},
        {"client_id": "B3", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 10.0, "p_model": 0.99},
        {"client_id": "B4", "article_id": "X", "vendor_code": "V2",
         "ca_sum": 50.0, "p_model": 0.20},
        {"client_id": "O1", "article_id": "X", "vendor_code": "V2",
         "ca_sum": 9999.0, "p_model": 0.99},
        {"client_id": "B1", "article_id": "S1", "vendor_code": "V1",
         "ca_sum": 700.0, "p_model": 0.80},
    ])


def test_block1_ranks_by_expected_value_not_by_probability():
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1")
    assert list(out["client_id"]) == ["B2", "B1", "B4", "B3"]
    assert out.iloc[0]["expected_value"] == pytest.approx(500.0)


def test_block1_ownership_follows_the_fact_table_not_the_scores_vendor():
    """B4's row says V2, but V1 trades with B4 — B4 belongs in V1's list."""
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1")
    assert "B4" in set(out["client_id"])
    assert "O1" not in set(out["client_id"])


def test_block1_is_restricted_to_the_requested_article():
    out = ap.existing_customers(make_scores(), make_fact(), "S1", "V1")
    assert list(out["client_id"]) == ["B1"]


def test_block1_returns_typed_empty_frame_when_no_client_matches():
    out = ap.existing_customers(make_scores(), make_fact(), "UNKNOWN", "V1")
    assert out.empty
    assert list(out.columns) == ["client_id", "p_model", "ca_sum", "expected_value"]


def test_block1_passes_evidence_columns_through_when_the_scores_carry_them():
    """The page explains WHY a client ranks from n_orders / days_since_last /
    median_interval. They ride along with the ranking rather than being re-read."""
    scores = make_scores()
    scores["n_orders"] = 3
    scores["days_since_last"] = 40
    scores["median_interval"] = 20.0

    out = ap.existing_customers(scores, make_fact(), "X", "V1")

    assert out.iloc[0]["n_orders"] == 3
    assert out.iloc[0]["days_since_last"] == 40


def test_block1_still_ranks_when_the_scores_carry_no_evidence_columns():
    """Evidence enriches the explanation; it is not an input to the order. A score
    table without it must still rank, or a caller is forced to supply columns the
    ranking never reads."""
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1")

    assert list(out["client_id"]) == ["B2", "B1", "B4", "B3"]


def test_block1_honours_the_top_limit():
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1", top=2)
    assert list(out["client_id"]) == ["B2", "B1"]


def test_signature_finds_the_over_represented_articles():
    sig = ap.signature_articles(make_fact(), "X")
    assert set(sig) == {"S1", "S2", "S3"}


def test_signature_excludes_the_article_itself():
    assert "X" not in ap.signature_articles(make_fact(), "X")


def test_signature_excludes_a_popular_but_unrelated_article():
    """N is bought by 7 clients but by only 1 of X's 5 buyers — lift below 1."""
    assert "N" not in ap.signature_articles(make_fact(), "X")


def test_signature_is_ordered_by_lift():
    sig = ap.signature_articles(make_fact(), "X")
    assert sig[0] == "S1"


def test_signature_is_empty_below_the_minimum_buyer_count():
    fact = make_fact()
    rare = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "B1", "article_id": "RARE",
         "month": pd.Timestamp("2025-03"), "ca": 10.0},
        {"vendor_code": "V1", "client_id": "B2", "article_id": "RARE",
         "month": pd.Timestamp("2025-03"), "ca": 10.0},
    ])
    assert ap.signature_articles(pd.concat([fact, rare]), "RARE") == []


def test_signature_honours_the_top_limit():
    assert len(ap.signature_articles(make_fact(), "X", top=2)) == 2


def test_block2_ranks_prospects_by_signature_overlap():
    out, sig = ap.prospects(make_fact(), "X", "V1")
    assert len(sig) == 3
    assert list(out["client_id"]) == ["P1", "P2"]
    assert out.iloc[0]["n_signature"] == 3
    assert out.iloc[1]["n_signature"] == 1


def test_block2_excludes_clients_who_already_buy_it_under_another_vendor():
    """P4 is a V1 client and buys the whole signature — but buys X through V2."""
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "P4" not in set(out["client_id"])


def test_block2_excludes_clients_of_other_vendors():
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "O1" not in set(out["client_id"])


def test_block2_excludes_clients_with_no_signature_overlap():
    """P3 is a V1 client who has never bought any signature article."""
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "P3" not in set(out["client_id"])


def test_block2_returns_typed_empty_frame_when_the_signature_is_refused():
    fact = make_fact()
    rare = pd.DataFrame([{"vendor_code": "V1", "client_id": "B1", "article_id": "RARE",
                          "month": pd.Timestamp("2025-03"), "ca": 10.0}])
    out, sig = ap.prospects(pd.concat([fact, rare]), "RARE", "V1")
    assert sig == []
    assert out.empty
    assert list(out.columns) == ["client_id", "n_signature", "client_ca"]


def test_block2_honours_the_top_limit():
    out, _ = ap.prospects(make_fact(), "X", "V1", top=1)
    assert list(out["client_id"]) == ["P1"]


def test_block2_is_empty_when_every_client_already_buys_the_article():
    """Signature exists, but this vendor has no non-buyer left to propose it to.

    V3's only client is B1, who already buys X — so `candidates` is empty while the
    signature is perfectly valid. The page must say "aucun prospect", not "trop peu
    d'acheteurs": the two messages have different meanings.
    """
    fact = pd.concat([make_fact(), pd.DataFrame([
        {"vendor_code": "V3", "client_id": "B1", "article_id": "S1",
         "month": pd.Timestamp("2025-04"), "ca": 25.0}])])
    out, sig = ap.prospects(fact, "X", "V3")
    assert sig != []
    assert out.empty
    assert list(out.columns) == ["client_id", "n_signature", "client_ca"]


def test_a_vendor_with_a_single_client_degrades_cleanly():
    """Vendor size runs from 1 to 944 clients; the smallest must not crash either block."""
    fact = pd.concat([make_fact(), pd.DataFrame([
        {"vendor_code": "V9", "client_id": "SOLO", "article_id": "S1",
         "month": pd.Timestamp("2025-03"), "ca": 40.0}])])
    out, sig = ap.prospects(fact, "X", "V9")
    assert sig != []
    assert list(out["client_id"]) == ["SOLO"]
    assert ap.existing_customers(make_scores(), fact, "X", "V9").empty


# ---- Block 2: dormant buyers -----------------------------------------------
# Half of all buyer pairs sit outside the v4 snapshot's 12-month window, so the
# model has no opinion on them. They must not vanish: without their own block
# they leave block 1 (no score) and are excluded from block 3 (they ARE buyers).


def dormant_fact():
    """B1 and B2 bought X recently; B5 bought it long ago and stopped.
    P1 never bought X. O1 belongs to another vendor."""
    rows = []

    def add(vendor, client, article, ca=100.0, month="2026-03"):
        rows.append({"vendor_code": vendor, "client_id": client, "article_id": article,
                     "month": pd.Timestamp(month), "ca": ca})

    add("V1", "B1", "X", ca=100.0)
    add("V1", "B2", "X", ca=200.0)
    add("V1", "B5", "X", ca=900.0, month="2024-09")   # dormant, and the biggest
    add("V1", "B5", "X", ca=50.0, month="2024-11")    # its true last month
    add("V1", "P1", "S1")                             # never bought X
    add("V2", "O1", "X", ca=5000.0)                   # another vendor's client
    return pd.DataFrame(rows)


def dormant_scores():
    """v4 scored B1 and B2 only — B5 is outside the 12-month window."""
    return pd.DataFrame([
        {"client_id": "B1", "article_id": "X", "ca_sum": 100.0, "p_model": 0.6},
        {"client_id": "B2", "article_id": "X", "ca_sum": 200.0, "p_model": 0.4},
    ])


def test_a_scored_client_is_not_dormant():
    """B1 has a v4 score, so it belongs to block 1 and must not be duplicated."""
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert "B1" not in set(out["client_id"])


def test_a_client_who_never_bought_the_article_is_not_dormant():
    """P1 is a prospect, not a lapsed buyer — that is block 3's population."""
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert "P1" not in set(out["client_id"])


def test_the_dormant_buyer_is_found():
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert list(out["client_id"]) == ["B5"]


def test_another_vendors_client_is_excluded():
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert "O1" not in set(out["client_id"])


def test_dormant_buyers_carry_no_probability():
    """v4 has no opinion on these pairs. Any percentage here would be invented —
    the same rule block 3 already follows."""
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert not [c for c in out.columns if "p_model" in c or "proba" in c]


def test_ranked_by_article_ca_descending():
    fact = dormant_fact()
    fact = pd.concat([fact, pd.DataFrame([
        {"vendor_code": "V1", "client_id": "B6", "article_id": "X",
         "month": pd.Timestamp("2024-09"), "ca": 2000.0}])], ignore_index=True)
    out = ap.dormant_buyers(fact, dormant_scores(), "X", "V1")
    assert list(out["client_id"]) == ["B6", "B5"]


def test_last_month_is_the_last_positive_month_on_that_article():
    """B5 bought X in 2024-09 and again in 2024-11; the vendor needs the later
    one to say when it stopped."""
    out = ap.dormant_buyers(dormant_fact(), dormant_scores(), "X", "V1")
    assert out.iloc[0]["last_month"] == pd.Timestamp("2024-11")


def test_a_return_does_not_make_someone_a_dormant_buyer():
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "R1", "article_id": "X",
         "month": pd.Timestamp("2024-09"), "ca": -80.0},
        {"vendor_code": "V1", "client_id": "R1", "article_id": "S1",
         "month": pd.Timestamp("2024-09"), "ca": 80.0},
    ])
    out = ap.dormant_buyers(fact, dormant_scores(), "X", "V1")
    assert "R1" not in set(out["client_id"])


def test_buyers_of_sees_a_purchase_outside_january_to_may():
    """Regression guard for the 400,353 buyer pairs that ca_monthly.parquet hid.
    A September buyer counted as a prospect would be pitched a part they already
    buy, exposing a colleague's business."""
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "B9", "article_id": "X",
         "month": pd.Timestamp("2025-09"), "ca": 100.0}])
    assert "B9" in ap.buyers_of(fact, "X")
