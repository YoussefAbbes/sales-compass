# -*- coding: utf-8 -*-
"""Every ranked list must be fully reachable, not truncated by the transport.

Three caps existed and only two were policy. Blocks A and B were held at 8 by the
frontend against an API that already allowed 50. Articles-per-client was held at 3
by the SHAPE of clients_to_contact.csv, which pivots articles into article_txt_1..3
-- so `per_client` was `le=3` because the file could not answer for a fourth.

The pair-level score table can: it holds every scored article for a client. The
queue's three stay authoritative and untouched (contact_queue.py fills model
coverage gaps the pair table has no row for), and the rest are appended from the
score table, ranked by the same damped expected value block A uses.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from model_results_api.app.main import app
from model_results_api.app.services import extra_article_order

client = TestClient(app)


def _vendor() -> str:
    return client.get("/api/v1/vendors").json()["items"][0]["vendor_code"]


def _client_with_many_articles(vendor: str):
    """A client whose scored catalogue is deep enough to test expansion."""
    body = client.get(f"/api/v1/vendors/{vendor}/repurchases?per_client=50&limit_clients=50")
    for c in body.json()["clients"]:
        if len(c["opportunities"]) > 3:
            return c["client_id"]
    return None


# --- articles per client ---------------------------------------------------------

def test_per_client_above_three_is_accepted():
    """The old ceiling was the CSV's column count, not a policy."""
    r = client.get(f"/api/v1/vendors/{_vendor()}/repurchases?per_client=10")

    assert r.status_code == 200


def test_per_client_returns_more_than_the_three_pivoted_columns():
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep, "no client in fixtures has more than 3 scored articles"

    body = client.get(
        f"/api/v1/vendors/{vendor}/repurchases?client_id={deep}&per_client=10").json()

    assert len(body["clients"][0]["opportunities"]) > 3


def test_the_first_three_are_untouched_by_asking_for_more():
    """Expansion is additive. A vendor who expands must not see the top of their
    list reshuffle or re-word underneath them."""
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep

    base = client.get(
        f"/api/v1/vendors/{vendor}/repurchases?client_id={deep}&per_client=3"
    ).json()["clients"][0]["opportunities"]
    more = client.get(
        f"/api/v1/vendors/{vendor}/repurchases?client_id={deep}&per_client=10"
    ).json()["clients"][0]["opportunities"]

    assert [o["article_id"] for o in more[:3]] == [o["article_id"] for o in base]
    assert [o["reason"] for o in more[:3]] == [o["reason"] for o in base]


def test_the_appended_articles_are_not_duplicates_of_the_first_three():
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep

    ops = client.get(
        f"/api/v1/vendors/{vendor}/repurchases?client_id={deep}&per_client=20"
    ).json()["clients"][0]["opportunities"]

    ids = [o["article_id"] for o in ops]
    assert len(ids) == len(set(ids))


def test_ranks_are_contiguous_so_the_page_can_show_absolute_positions():
    """The frontend reveals in pages; #9 must mean ninth overall, not first of page 2."""
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep

    ops = client.get(
        f"/api/v1/vendors/{vendor}/repurchases?client_id={deep}&per_client=20"
    ).json()["clients"][0]["opportunities"]

    assert [o["rank"] for o in ops] == list(range(1, len(ops) + 1))


def test_the_appended_articles_are_ordered_by_damped_expected_value():
    """The ordering rule, on synthetic rows so it is exact rather than inferred.

    Same rule block A uses: rank on expected value x recency damping, report the
    expected value undamped. BIG is the client's largest line by far but they took
    delivery yesterday against a 12-day cadence, so it must not lead the pitch --
    while its reported TND stays the real figure.
    """
    pool = pd.DataFrame([
        {"article_id": "BIG",  "p_model": 0.90, "days_since_last": 1,  "median_interval": 12.0,
         "n_orders": 27},
        {"article_id": "DUE",  "p_model": 0.50, "days_since_last": 80, "median_interval": 40.0,
         "n_orders": 6},
        {"article_id": "SMALL","p_model": 0.40, "days_since_last": 90, "median_interval": 30.0,
         "n_orders": 4},
    ])
    previous_ca = {"BIG": 1000.0, "DUE": 600.0, "SMALL": 100.0}

    out = extra_article_order(pool, previous_ca)

    assert list(out["article_id"]) == ["DUE", "BIG", "SMALL"]
    assert out.set_index("article_id").loc["BIG", "expected_value"] == pytest.approx(900.0)


def test_the_ordering_rule_needs_no_cadence_columns():
    """A pool without cadence must still order, by plain expected value."""
    pool = pd.DataFrame([{"article_id": "A", "p_model": 0.2},
                         {"article_id": "B", "p_model": 0.9}])

    out = extra_article_order(pool, {"A": 1000.0, "B": 100.0})

    assert list(out["article_id"]) == ["A", "B"]


def test_per_client_is_still_bounded():
    """Generous, not unlimited -- the README promises the frontend bounded JSON."""
    assert client.get(f"/api/v1/vendors/{_vendor()}/repurchases?per_client=51").status_code == 422


# --- the client dossier ----------------------------------------------------------

def test_the_client_dossier_can_ask_for_more_articles():
    """It hardcoded per_client=3, so the dossier page could never expand."""
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep

    r = client.get(f"/api/v1/vendors/{vendor}/clients/{deep}?per_client=10")

    assert r.status_code == 200
    assert len(r.json()["opportunities"]) > 3


def test_the_client_dossier_still_defaults_to_three():
    vendor = _vendor()
    deep = _client_with_many_articles(vendor)
    assert deep

    body = client.get(f"/api/v1/vendors/{vendor}/clients/{deep}").json()

    assert len(body["opportunities"]) == 3


# --- blocks A and B --------------------------------------------------------------

def test_the_article_blocks_accept_a_generous_limit():
    vendor = _vendor()
    article = client.get("/api/v1/articles?limit=1").json()["items"][0]["article_id"]

    r = client.get(
        f"/api/v1/vendors/{vendor}/articles/{article}/recommendations?limit=200")

    assert r.status_code == 200


def test_the_article_blocks_stay_bounded():
    vendor = _vendor()
    article = client.get("/api/v1/articles?limit=1").json()["items"][0]["article_id"]

    assert client.get(
        f"/api/v1/vendors/{vendor}/articles/{article}/recommendations?limit=201"
    ).status_code == 422
    assert client.get(
        f"/api/v1/vendors/{vendor}/articles/{article}/prospects?limit=201"
    ).status_code == 422
