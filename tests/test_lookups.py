# -*- coding: utf-8 -*-
"""Lookup tables projected from the NAV masters. Synthetic frames only.

articles.csv and clients.csv came from the old January-May Excel via
build_dataset.py. The NAV masters cover 99.9% and 100% of them respectively and
carry far more, so the lookups are now a PROJECTION of the masters rather than a
separate source.

Projection, not replacement. Two merges take the lookup wholesale
(baseline_lists and contact_queue), so handing them the 21-column master would
put phone numbers, addresses, tax ids and credit limits into
clients_to_contact.csv — a file the app renders and vendor_sheets.py exports to
Excel for vendors. The projection keeps those out by construction.
"""
import pandas as pd

from build_masters import project_articles, project_clients


def articles_master():
    return pd.DataFrame({
        "article_id": ["A1", "A2"],
        "enseigne": ["FEBI", "SACHS"],
        "designation": ["Filtre a huile", "Embrayage"],
        "unit_price": ["10.5", "220.0"],
        "stock": ["4", "0"],
        "unit_cost": ["7.0", "150.0"],
    })


def clients_master():
    return pd.DataFrame({
        "client_id": ["C1", "C2", "C3"],
        "client_name": ["GARAGE DUPONT", "STE FERME", "AUTO PLUS"],
        "blocked": ["False", "True", "False"],
        "blocked_reason": [None, "Factures non reglees", None],
        "phone_1": ["20000000", "21000000", "22000000"],
        "address": ["rue A", "rue B", "rue C"],
        "tax_id": ["111", "222", "333"],
        "credit_limit": ["1000", "0", "5000"],
    })


def test_article_lookup_keeps_only_the_three_display_columns():
    out = project_articles(articles_master())
    assert list(out.columns) == ["article_id", "enseigne", "designation"]


def test_article_lookup_drops_price_and_stock():
    """unit_price, unit_cost, stock and stock_class describe the CURRENT state.
    train_rolling refuses them as features; they must not reach a lookup either."""
    out = project_articles(articles_master())
    for leaked in ("unit_price", "unit_cost", "stock"):
        assert leaked not in out.columns


def test_client_lookup_carries_the_blocked_flag():
    """The old file encoded blocked-ness as '***' inside the name because the
    extract had nowhere else to put it. NAV has a real column."""
    out = project_clients(clients_master())
    assert list(out.columns) == ["client_id", "client_name", "blocked"]
    assert out.set_index("client_id")["blocked"].to_dict() == {
        "C1": False, "C2": True, "C3": False}


def test_client_lookup_drops_every_pii_column():
    """These merges land in clients_to_contact.csv, which vendor_sheets.py
    exports to Excel and hands to vendors."""
    out = project_clients(clients_master())
    for leaked in ("phone_1", "address", "tax_id", "credit_limit", "blocked_reason"):
        assert leaked not in out.columns


def test_blocked_is_a_real_boolean_not_a_string():
    """'False' is truthy. Leaving it as text would flag every client."""
    out = project_clients(clients_master())
    assert out["blocked"].dtype == bool


def test_a_missing_blocked_value_is_treated_as_not_blocked():
    """Absence of evidence is not a block: excluding a client from every vendor's
    queue needs NAV to actually say so."""
    m = clients_master()
    m.loc[0, "blocked"] = None
    out = project_clients(m)
    assert out.loc[0, "blocked"] is False or out.loc[0, "blocked"] == False  # noqa: E712
