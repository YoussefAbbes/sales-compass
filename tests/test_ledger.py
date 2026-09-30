# -*- coding: utf-8 -*-
"""Monthly revenue from the NAV ledger. Synthetic frames only.

ca_monthly.parquet only ever held January-May of each year, because that is all
the v1 extract covered. Anything derived from it — "last activity", the history
chart, the cross-sell "already owns this" set — silently inherited that hole.
The ledger has every month, so these are computed from it instead.
"""
import pandas as pd

from ledger import monthly_ca


def tx(date, client="C1", article="A1", amount=100.0, vendor="V1"):
    return {"date": pd.Timestamp(date), "client_id": client, "vendor_code": vendor,
            "article_id": article, "amount": amount}


def test_every_month_survives_not_just_january_to_may():
    """The whole point: a purchase in October must not vanish."""
    frame = pd.DataFrame([tx("2025-03-10"), tx("2025-10-04"), tx("2025-12-31")])
    out = monthly_ca(frame)
    months = sorted(out["month"].dt.strftime("%Y-%m"))
    assert months == ["2025-03", "2025-10", "2025-12"]


def test_amounts_are_summed_within_a_client_article_month():
    frame = pd.DataFrame([tx("2025-03-04", amount=100.0),
                          tx("2025-03-19", amount=50.0)])
    out = monthly_ca(frame)
    assert len(out) == 1
    assert out.loc[0, "ca"] == 150.0


def test_returns_net_off_within_the_month():
    """A credit note is a negative amount on the ledger. Netting it here is what
    lets the history chart show a negative bar, as it always has."""
    frame = pd.DataFrame([tx("2025-03-04", amount=100.0),
                          tx("2025-03-19", amount=-30.0)])
    out = monthly_ca(frame)
    assert out.loc[0, "ca"] == 70.0


def test_month_is_normalised_to_the_first_of_the_month():
    out = monthly_ca(pd.DataFrame([tx("2025-03-19")]))
    assert out.loc[0, "month"] == pd.Timestamp("2025-03-01")


def test_clients_and_articles_stay_separate():
    frame = pd.DataFrame([tx("2025-03-04", client="C1", article="A1"),
                          tx("2025-03-04", client="C1", article="A2"),
                          tx("2025-03-04", client="C2", article="A1")])
    out = monthly_ca(frame)
    assert len(out) == 3


def test_the_schema_matches_what_ca_monthly_provided():
    """Drop-in replacement: consumers group by these exact names.

    vendor_code is NOT optional. article_prospects.vendor_clients() reads it to
    decide which clients a vendor owns, and the Prospection page builds its
    vendor picker from it — omitting it raised KeyError on page load.
    """
    out = monthly_ca(pd.DataFrame([tx("2025-03-04")]))
    assert set(out.columns) == {"vendor_code", "client_id", "article_id", "month", "ca"}


def test_the_same_pair_under_two_vendors_stays_two_rows():
    """Ownership is per vendor: 78% of clients are served by several. Collapsing
    them would hand one vendor another's business."""
    frame = pd.DataFrame([tx("2025-03-04", vendor="V1"),
                          tx("2025-03-04", vendor="V2")])
    out = monthly_ca(frame)
    assert len(out) == 2
    assert set(out["vendor_code"]) == {"V1", "V2"}
