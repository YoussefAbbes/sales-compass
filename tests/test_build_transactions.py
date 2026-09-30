# -*- coding: utf-8 -*-
"""normalise_ledger is pure: synthetic frames only, never the 121 MB workbook."""
import pandas as pd
import pytest

from build_transactions import normalise_ledger


def raw_row(**over):
    """One raw ledger row using the exact French column names NAV exports."""
    row = {
        "Date comptabilisation": pd.Timestamp("2024-01-02"),
        "Type écriture": "Vente",
        "Type document": "Expédition vente",
        "N° document": "B241000001",
        "N° article": "01381",
        "Désignation article": "ANTIGEL 1.5L ROUGE / CONCENT",
        "Enseigne": "FEBI",
        "N° origine": "0289",
        "Nom": "STE EXEMPLE",
        "Code vendeur": "007",
        "Nom Vend.": "v10 EXEMPLE",
        "Code magasin": "A02",
        "Quantité": 12.0,
        "Montant vente (réel)": 191.592,
        "Coût total (réel)": 140.0,
        "Discount Amount": 0.0,
        "CU": 1.0,
    }
    row.update(over)
    return row


def test_identifiers_stay_strings_with_leading_zeros():
    out = normalise_ledger(pd.DataFrame([raw_row()]))
    assert out.loc[0, "client_id"] == "0289"
    assert out.loc[0, "article_id"] == "01381"
    assert out.loc[0, "vendor_code"] == "007"


def test_returns_are_flagged_not_dropped():
    out = normalise_ledger(pd.DataFrame([
        raw_row(),
        raw_row(**{"Type document": "Réception retour vente",
                   "Quantité": -2.0, "Montant vente (réel)": -31.932}),
    ]))
    assert out["is_return"].tolist() == [False, True]
    assert len(out) == 2, "returns are evidence, not noise — never dropped at ingest"


def test_amounts_are_rounded_to_millimes():
    out = normalise_ledger(pd.DataFrame([
        raw_row(**{"Montant vente (réel)": 191.5924999})
    ]))
    assert out.loc[0, "amount"] == 191.592


def test_whitespace_is_stripped_from_identifiers():
    out = normalise_ledger(pd.DataFrame([
        raw_row(**{"N° origine": "  0289 ", "N° article": " 01381"})
    ]))
    assert out.loc[0, "client_id"] == "0289"
    assert out.loc[0, "article_id"] == "01381"


def test_date_becomes_datetime64():
    out = normalise_ledger(pd.DataFrame([raw_row()]))
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_rows_with_no_identifier_are_dropped():
    out = normalise_ledger(pd.DataFrame([raw_row(), raw_row(**{"N° origine": None})]))
    assert len(out) == 1


def test_unexpected_entry_type_raises():
    """Type écriture is uniformly 'Vente' in both workbooks. Anything else means the
    export changed shape and every downstream assumption needs re-checking."""
    with pytest.raises(ValueError, match="Type écriture"):
        normalise_ledger(pd.DataFrame([raw_row(**{"Type écriture": "Achat"})]))


def test_numeric_columns_are_float64_even_when_all_values_are_whole_numbers():
    """qty, amount, cost and discount are declared float in the interface. When
    every raw cell is integral (as openpyxl returns for whole-unit quantities),
    pandas infers int64 unless the cast is explicit — a fixture built on
    fractional values would pass even with that bug present, so this one uses
    whole-number Python ints, matching what the real export actually contains."""
    out = normalise_ledger(pd.DataFrame([
        raw_row(**{"Quantité": 12, "Montant vente (réel)": 192,
                   "Coût total (réel)": 140, "Discount Amount": 0}),
    ]))
    for col in ["qty", "amount", "cost", "discount"]:
        assert out[col].dtype == "float64", f"{col} dtype is {out[col].dtype}, expected float64"


from build_masters import normalise_articles, normalise_clients


def test_article_master_keeps_leading_zero_ids_and_splits_vehicle():
    raw = pd.DataFrame([{
        "N°": "000323", "Enseigne": "XTRA", "Désignation": "DEFLECTEUR AIR KDH",
        "Code catégorie article": "PCE-CARR", "Marque par défaut": "TY",
        "Marque": "TY-HIACE", "N° origine parent": "42311-26290",
        "Prix unitaire": 16.124, "Coût unitaire": 14.021, "Stocks": 12,
        "Plage de Stock": "MORT-", "Pareto": "DD",
    }])
    out = normalise_articles(raw)
    assert out.loc[0, "article_id"] == "000323"
    assert out.loc[0, "vehicle_make"] == "TY"
    assert out.loc[0, "vehicle_model"] == "TY-HIACE"
    assert out.loc[0, "stock_class"] == "MORT-"


def test_client_master_normalises_blocking_to_a_boolean_plus_detail():
    raw = pd.DataFrame([
        {"N°": "0001", "Nom": "A.P.R DISTRIBUTION", "Bloqué": "Facturer",
         "Cause dernier blocage": "Factures non réglées"},
        {"N°": "0002", "Nom": "Mr EXEMPLE", "Bloqué": None,
         "Cause dernier blocage": None},
    ])
    out = normalise_clients(raw)
    assert out["blocked"].tolist() == [True, False]
    assert out.loc[0, "blocked_detail"] == "Facturer"
    assert out.loc[0, "blocked_reason"] == "Factures non réglées"


def test_client_master_collects_all_three_vendor_codes():
    raw = pd.DataFrame([{
        "N°": "0003", "Nom": "Mr EXEMPLE", "Bloqué": None, "Code vendeur": "550",
        "<Code Vendeur 2>": "585", "<Code Vendeur 3>": None,
    }])
    out = normalise_clients(raw)
    assert out.loc[0, "vendor_1"] == "550"
    assert out.loc[0, "vendor_2"] == "585"
    assert out.loc[0, "vendor_3"] == ""


def test_normalise_clients_raises_when_bloque_is_absent():
    """Bloqué drives the `blocked` safety filter. Blank-filling it would make
    every client look unblocked, silently disabling the rule that stops
    vendors calling non-paying accounts — so a missing export column must
    raise, not fill blank."""
    raw = pd.DataFrame([{"N°": "0001", "Nom": "A.P.R DISTRIBUTION"}])
    with pytest.raises(ValueError, match="Bloqué"):
        normalise_clients(raw)


def test_normalise_articles_raises_when_plage_de_stock_is_absent():
    """Plage de Stock drives the stock_class scoring-time filter, same
    reasoning as the Bloqué check above."""
    raw = pd.DataFrame([{"N°": "000323", "Enseigne": "XTRA"}])
    with pytest.raises(ValueError, match="Plage de Stock"):
        normalise_articles(raw)


def test_normalise_clients_still_tolerates_a_non_required_missing_column():
    """Proves the required-column check narrowed the tolerance rather than
    removed it: Ville is a display field, not a filter, so its absence must
    still blank-fill and build rather than raise."""
    raw = pd.DataFrame([{"N°": "0001", "Nom": "A.P.R DISTRIBUTION", "Bloqué": None}])
    out = normalise_clients(raw)
    assert out.loc[0, "city"] == ""
