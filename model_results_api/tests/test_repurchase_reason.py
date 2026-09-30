# -*- coding: utf-8 -*-
"""Block A must say why THIS client ranks, not why the block exists.

The page carried one hardcoded sentence for every client -- "Ce client a deja
achete la piece. Son score de rachat et son ancien CA donnent la valeur attendue"
-- which restates the formula already printed above it and is equally true of
every row. Block B, next to it, cites real counts per prospect.

The facts were never missing. pair_scores_live.csv carries n_orders,
days_since_last and median_interval per pair, and the ranking dropped them:
BLOCK1_COLUMNS kept only the score, the CA and their product. Those three are
also what the model ranks on -- client_days_since_last is its top feature -- so
surfacing them explains the ranking in the terms that produced it.
"""
import math
import unicodedata

from fastapi.testclient import TestClient

from model_results_api.app.main import app
from model_results_api.app.schemas import RepurchaseSignals
from model_results_api.app.services import _repurchase_reason


client = TestClient(app)


def _fold(text: str) -> str:
    """Compare without accents: the assertions are about which claim the sentence
    makes, not about how French spells it."""
    return (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore").decode("ascii").lower())


def _signals(n_orders=11, days_since_last=14, median_interval=22.0):
    ratio = (days_since_last / median_interval) if median_interval else None
    return RepurchaseSignals(
        n_orders=n_orders,
        days_since_last=days_since_last,
        median_interval_days=median_interval,
        overdue_ratio=ratio,
    )


# --- the facts the sentence must carry ------------------------------------------

def test_reason_quotes_this_client_s_own_numbers():
    reason = _repurchase_reason(_signals(n_orders=11, days_since_last=14))

    assert "11" in reason
    assert "14" in reason


def test_two_different_clients_do_not_get_the_same_sentence():
    """The defect restated: one string described every row."""
    a = _repurchase_reason(_signals(n_orders=11, days_since_last=14))
    b = _repurchase_reason(_signals(n_orders=2, days_since_last=90))

    assert a != b


# --- the clause derived from the model's own overdue_ratio ----------------------

def test_a_client_well_past_their_rhythm_is_called_late():
    reason = _repurchase_reason(_signals(days_since_last=44, median_interval=22.0))

    assert "retard" in reason.lower()


def test_a_client_at_their_rhythm_is_called_due():
    reason = _repurchase_reason(_signals(days_since_last=25, median_interval=22.0))

    assert "échéance" in reason.lower()


def test_a_client_inside_their_rhythm_is_not_called_late():
    reason = _repurchase_reason(_signals(days_since_last=8, median_interval=22.0))

    assert "retard" not in reason.lower()


def test_a_client_served_early_in_their_cycle_says_so():
    """The sentence must name the fact that demoted them, in the vendor's terms."""
    reason = _repurchase_reason(_signals(days_since_last=8, median_interval=22.0))

    assert "recemment" in _fold(reason)


def test_a_client_served_early_never_claims_to_be_due():
    """The defect: "encore dans son cycle" was rendered under a panel headed
    "Pourquoi il remonte". The box printed the single strongest reason to rank the
    client DOWN as its justification for ranking them up, and at 1 day into a
    12-day cadence it was as wrong as it can get."""
    reason = _repurchase_reason(_signals(days_since_last=1, median_interval=12.0))

    assert "echeance" not in _fold(reason)
    assert "encore dans son cycle" not in _fold(reason)


def test_a_client_served_early_says_their_priority_was_reduced():
    """Ranking and copy must agree. existing_customers damps this client's place in
    the list; the sentence has to account for the demotion the vendor can see."""
    reason = _repurchase_reason(_signals(days_since_last=1, median_interval=12.0))

    assert "priorite reduite" in _fold(reason)


def test_a_single_purchase_has_no_rhythm_to_be_late_against():
    """median_interval is NaN for one purchase, and snapshots.py keeps that
    distinct from zero on purpose. The sentence must not invent a cadence."""
    reason = _repurchase_reason(_signals(n_orders=1, median_interval=None))

    assert "retard" not in reason.lower()
    assert "seul achat" in reason.lower()


# --- what the endpoint actually returns -----------------------------------------

def test_the_endpoint_carries_signals_and_a_reason_per_client():
    vendor = client.get("/api/v1/vendors").json()["items"][0]["vendor_code"]
    article = client.get("/api/v1/articles?limit=1").json()["items"][0]["article_id"]

    body = client.get(
        f"/api/v1/vendors/{vendor}/articles/{article}/recommendations").json()
    items = body["repurchases"]["items"]
    if not items:
        return

    first = items[0]
    assert first["reason"]
    assert first["signals"]["n_orders"] >= 1
    assert first["signals"]["days_since_last"] >= 0
