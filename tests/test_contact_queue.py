# -*- coding: utf-8 -*-
"""The score-source preference. Synthetic frames only.

The queue must prefer the v4 live scores, fall back to v1, and fall back again
to the manual score — each step a real degradation, so which one won is printed
rather than guessed.
"""
import pandas as pd
import pytest

from contact_queue import attach_model_scores


def lapsed_frame():
    return pd.DataFrame({
        "client_id": ["C1", "C2"],
        "article_id": ["A1", "A2"],
        "score": [10.0, 5.0],
        "ca_2025": [1000.0, 500.0],
    })


def write_live(scratch_dir, p1=0.9, p2=0.8):
    pd.DataFrame({"client_id": ["C1", "C2"], "article_id": ["A1", "A2"],
                  "vendor_code": ["V1", "V1"], "p": [p1, p2]}).to_csv(
        scratch_dir / "pair_scores_live.csv", index=False, encoding="utf-8-sig")


def write_v1(scratch_dir, p1=0.1, p2=0.2):
    pd.DataFrame({"client_id": ["C1", "C2"], "article_id": ["A1", "A2"],
                  "p_model": [p1, p2]}).to_csv(
        scratch_dir / "pair_scores_2026.csv", index=False, encoding="utf-8-sig")


def test_live_scores_win_when_both_files_exist(scratch_dir):
    write_live(scratch_dir)
    write_v1(scratch_dir)
    out = attach_model_scores(lapsed_frame(), scratch_dir)
    assert out["p_model"].tolist() == [0.9, 0.8]


def test_v1_is_used_when_there_is_no_live_file(scratch_dir):
    write_v1(scratch_dir)
    out = attach_model_scores(lapsed_frame(), scratch_dir)
    assert out["p_model"].tolist() == [0.1, 0.2]


def test_manual_score_is_the_last_resort(scratch_dir):
    out = attach_model_scores(lapsed_frame(), scratch_dir)
    assert out["p_model"].tolist() == [1.0, 0.5]


def test_pairs_missing_from_the_score_file_get_the_median_not_nan(scratch_dir):
    """A live snapshot only contains pairs active in its feature window, so
    lapsed pairs can legitimately be absent. They must not become NaN and sink
    the ranking silently."""
    pd.DataFrame({"client_id": ["C1"], "article_id": ["A1"],
                  "vendor_code": ["V1"], "p": [0.9]}).to_csv(
        scratch_dir / "pair_scores_live.csv", index=False, encoding="utf-8-sig")
    out = attach_model_scores(lapsed_frame(), scratch_dir)
    assert out["p_model"].notna().all()
    assert out["p_model"].tolist() == [0.9, 0.9]


# ---- v4 candidate universe -------------------------------------------------

from contact_queue import build_v4_candidates


def live_scores():
    return pd.DataFrame({
        "client_id": ["C1", "C1", "C2"],
        "article_id": ["A1", "A2", "A1"],
        "vendor_code": ["V1", "V1", "V2"],
        "p": [0.7, 0.3, 0.5],
        "amount_sum": [1000.0, 200.0, 800.0],
        "days_since_last": [120.0, 10.0, 40.0],
        "median_interval": [30.0, 30.0, 35.0],
        "n_orders": [6, 1, 4],
    })


ARTICLES = pd.DataFrame({"article_id": ["A1", "A2"], "enseigne": ["FEBI", "SACHS"],
                         "designation": ["Filtre a huile", "Embrayage"]})
CLIENTS = pd.DataFrame({"client_id": ["C1", "C2"],
                        "client_name": ["GARAGE DUPONT", "STE FERME ***"]})
VENDORS = pd.DataFrame({"vendor_code": ["V1", "V2"],
                        "vendor_name": ["Ali", "Sonia"]})


def test_candidates_expose_the_columns_the_queue_assembles_from():
    out = build_v4_candidates(live_scores(), ARTICLES, CLIENTS, VENDORS)
    for col in ("vendor_code", "vendor_name", "client_id", "client_name",
                "client_flagged", "article_id", "enseigne", "designation",
                "pair_ca", "p_model", "reason"):
        assert col in out.columns, col


def test_pair_ca_is_the_ledger_amount_not_a_borrowed_extract_figure():
    """This is the whole point of ranking on the v4 universe: expected value can
    use the pair's own NAV revenue instead of ca_2025 from the older basis."""
    out = build_v4_candidates(live_scores(), ARTICLES, CLIENTS, VENDORS)
    row = out[(out["client_id"] == "C1") & (out["article_id"] == "A1")].iloc[0]
    assert row["pair_ca"] == 1000.0
    assert row["p_model"] == 0.7


def test_flagged_clients_are_marked_from_the_star_convention():
    out = build_v4_candidates(live_scores(), ARTICLES, CLIENTS, VENDORS)
    flags = dict(zip(out["client_id"], out["client_flagged"]))
    assert flags["C2"] is True or flags["C2"] == True   # noqa: E712
    assert not flags["C1"]


def test_an_overdue_pair_says_so_in_the_reason():
    """120 days since last against a 30-day habit is overdue; the vendor is told
    why the row is there, in rule-based text, never model-generated prose."""
    out = build_v4_candidates(live_scores(), ARTICLES, CLIENTS, VENDORS)
    row = out[(out["client_id"] == "C1") & (out["article_id"] == "A1")].iloc[0]
    assert "retard" in row["reason"].lower()


def test_an_unknown_article_still_produces_a_usable_row(scratch_dir):
    live = live_scores()
    live.loc[len(live)] = ["C1", "A99", "V1", 0.4, 50.0, 5.0, 20.0, 2]
    out = build_v4_candidates(live, ARTICLES, CLIENTS, VENDORS)
    row = out[out["article_id"] == "A99"].iloc[0]
    assert row["designation"]
    assert pd.notna(row["enseigne"])


def test_a_long_dormant_pair_is_never_called_recent():
    """A single purchase 357 days ago fell through to the "Acheteur récent"
    branch, which a vendor would read as a contradiction of the number printed
    beside it. Silence about recency is fine; a false claim is not."""
    live = pd.DataFrame({
        "client_id": ["C1"], "article_id": ["A1"], "vendor_code": ["V1"],
        "p": [0.15], "amount_sum": [500.0], "days_since_last": [357.0],
        "median_interval": [float("nan")], "n_orders": [1],
    })
    out = build_v4_candidates(live, ARTICLES, CLIENTS, VENDORS)
    reason = out.loc[0, "reason"].lower()
    assert "récent" not in reason and "recent" not in reason
    assert "357" in reason


def test_a_genuinely_recent_pair_may_say_so():
    live = pd.DataFrame({
        "client_id": ["C1"], "article_id": ["A1"], "vendor_code": ["V1"],
        "p": [0.6], "amount_sum": [500.0], "days_since_last": [12.0],
        "median_interval": [float("nan")], "n_orders": [1],
    })
    out = build_v4_candidates(live, ARTICLES, CLIENTS, VENDORS)
    assert "récent" in out.loc[0, "reason"].lower()


# ---- blocked-ness comes from NAV, not from asterisks in a name --------------


def test_the_blocked_column_wins_over_the_star_convention():
    """NAV has a real field. The '***' convention was a workaround for an extract
    that had nowhere else to put it, and it saw only a fraction of the truth."""
    clients = pd.DataFrame({"client_id": ["C1", "C2"],
                            "client_name": ["GARAGE DUPONT", "STE FERME"],
                            "blocked": [False, True]})
    out = build_v4_candidates(live_scores(), ARTICLES, clients, VENDORS)
    flags = dict(zip(out["client_id"], out["client_flagged"]))
    assert flags["C1"] is False or flags["C1"] == False   # noqa: E712
    assert flags["C2"] is True or flags["C2"] == True     # noqa: E712


def test_a_starless_name_can_still_be_blocked():
    """The whole point: NAV blocks clients whose NAME says nothing. Falling back
    to the asterisk here would silently keep a non-payer in the queue."""
    clients = pd.DataFrame({"client_id": ["C1", "C2"],
                            "client_name": ["GARAGE DUPONT", "AUTO PLUS"],
                            "blocked": [True, False]})
    out = build_v4_candidates(live_scores(), ARTICLES, clients, VENDORS)
    assert dict(zip(out["client_id"], out["client_flagged"]))["C1"]


def test_the_star_convention_still_works_without_a_blocked_column():
    """Backwards compatible: a lookup that predates the projection must not
    suddenly mark every client as callable."""
    clients = pd.DataFrame({"client_id": ["C1", "C2"],
                            "client_name": ["GARAGE DUPONT", "STE FERME ***"]})
    out = build_v4_candidates(live_scores(), ARTICLES, clients, VENDORS)
    flags = dict(zip(out["client_id"], out["client_flagged"]))
    assert flags["C2"] and not flags["C1"]


def test_a_blocked_column_read_back_as_text_is_parsed_not_cast():
    """csv has no booleans, and the queue reads clients.csv with dtype=str, so
    `blocked` arrives as "True"/"False". astype(bool) on the string "False" is
    True, which flags every client and empties the queue - observed, not
    hypothetical."""
    clients = pd.DataFrame({"client_id": ["C1", "C2"],
                            "client_name": ["GARAGE DUPONT", "STE FERME"],
                            "blocked": ["False", "True"]})
    out = build_v4_candidates(live_scores(), ARTICLES, clients, VENDORS)
    flags = dict(zip(out["client_id"], out["client_flagged"]))
    assert not flags["C1"], "the string 'False' must not flag a client"
    assert flags["C2"]


# --- the reason baked into clients_to_contact.csv --------------------------------
# _v4_reason writes reason_1..3, which every queue surface renders verbatim. It had
# no notion of "served recently": a client with 27 orders on a 12-day cadence who
# ordered yesterday was labelled "Réachat régulier — 27 commandes sur 12 mois".
# True, and useless — that is why they rank, not a reason to call them today.

import unicodedata

from contact_queue import _v4_reason


def _fold(text: str) -> str:
    return (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore").decode("ascii").lower())


def _row(days, interval, n_orders):
    return {"days_since_last": days, "median_interval": interval, "n_orders": n_orders}


def test_v4_reason_flags_a_regular_buyer_who_was_just_served():
    reason = _v4_reason(_row(1, 12.0, 27))

    assert "recemment" in _fold(reason)
    assert "regulier" not in _fold(reason)


def test_v4_reason_quotes_the_cadence_the_vendor_can_check():
    """A vendor must be able to verify the claim against the ledger themselves —
    the rule this module already follows for every other branch."""
    reason = _v4_reason(_row(1, 12.0, 27))

    assert "il y a 1 j" in reason
    assert "12 j" in reason


def test_v4_reason_still_calls_a_late_client_late():
    """Lateness outranks recency and cannot both be true. Order must not regress."""
    reason = _v4_reason(_row(40, 12.0, 27))

    assert "retard" in _fold(reason)


def test_v4_reason_keeps_the_regular_buyer_label_once_they_are_due():
    reason = _v4_reason(_row(13, 12.0, 27))

    assert "regulier" in _fold(reason)


def test_v4_reason_invents_no_cadence_for_a_client_that_has_none():
    """median_interval is NaN for a single purchase. No cadence, no "too soon"."""
    reason = _v4_reason(_row(5, float("nan"), 1))

    assert "rythme" not in _fold(reason)
    assert "recemment" not in _fold(reason)


# --- priority_score damping ------------------------------------------------------
# priority_score is the single sort key behind six surfaces in services.py (the
# vendor queue, the overview, the brand rollups, the seller page). It was
# max(valeur_attendue, manque_a_gagner) x boost, where boost is only ever >= 1.0 --
# there was no term anywhere that could lower a client for having just been served.

from contact_queue import client_damping


def pair_frame(rows):
    return pd.DataFrame(rows)


def test_client_damping_reads_the_line_the_client_took_delivery_of_last():
    """A client is "just served" if ANY line reached them recently, not only the
    line the queue happens to pitch. C1 bought A2 yesterday against a 12-day
    cadence, so C1 was dealt with yesterday whatever the pitch is."""
    lapsed = pair_frame([
        {"client_id": "C1", "days_since_last": 200, "median_interval": 90.0},
        {"client_id": "C1", "days_since_last": 1, "median_interval": 12.0},
    ])

    assert client_damping(lapsed)["C1"] == pytest.approx(0.05)


def test_client_damping_leaves_a_client_due_on_every_line_untouched():
    lapsed = pair_frame([
        {"client_id": "C1", "days_since_last": 200, "median_interval": 90.0},
        {"client_id": "C1", "days_since_last": 30, "median_interval": 12.0},
    ])

    assert client_damping(lapsed)["C1"] == pytest.approx(1.0)


def test_client_damping_covers_every_client_in_the_frame():
    """A missing client would become NaN on multiplication and sort last -- a
    silent deletion from the queue rather than a demotion."""
    lapsed = pair_frame([
        {"client_id": "C1", "days_since_last": 1, "median_interval": 12.0},
        {"client_id": "C2", "days_since_last": 99, "median_interval": 12.0},
        {"client_id": "C3", "days_since_last": 5, "median_interval": float("nan")},
    ])

    out = client_damping(lapsed)

    assert set(out.index) == {"C1", "C2", "C3"}
    assert out.notna().all()


def test_client_damping_is_inert_on_the_v1_fallback_universe():
    """lapsed_reorders.csv carries no cadence columns. The queue must still build
    on it -- degraded, not broken -- exactly as attach_model_scores degrades."""
    lapsed = pair_frame([
        {"client_id": "C1", "pair_ca": 100.0},
        {"client_id": "C2", "pair_ca": 50.0},
    ])

    out = client_damping(lapsed)

    assert (out == 1.0).all()
    assert set(out.index) == {"C1", "C2"}


def test_client_damping_never_promotes():
    lapsed = pair_frame([
        {"client_id": "C1", "days_since_last": 900, "median_interval": 12.0},
    ])

    assert client_damping(lapsed)["C1"] <= 1.0
