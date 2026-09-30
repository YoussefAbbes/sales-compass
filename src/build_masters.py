# -*- coding: utf-8 -*-
"""
Build the article and client master tables from the NAV master exports.

Input : exported-data/deatils articles nav.xlsx  (sheet "Liste des articles")
        exported-data/deatils clients nav.xlsx   (sheet "Liste des clients")
Output: data/articles_master.csv, data/clients_master.csv

WARNING — these files are CURRENT-STATE SNAPSHOTS, not history. Stocks, Bloqué,
Cause dernier blocage, Solde, Plage de Stock and Crédit autorisé describe
2026-08-10 and nothing else.
Using them as TRAINING FEATURES is leakage: a client blocked today was very
often blocked *because* they stopped buying, so a model would learn the outcome
it is meant to predict. They are safe only as SCORING-TIME FILTERS, which is
where project.md §7 step 5 puts them.

Stable columns that ARE safe as features: category, vehicle make/model, OEM
reference, enseigne.
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exported-data"
DATA = ROOT / "data"

ARTICLE_COLUMNS = {
    "N°": "article_id",
    "Enseigne": "enseigne",
    "Désignation": "designation",
    "Code catégorie article": "category",
    "Marque par défaut": "vehicle_make",
    "Marque": "vehicle_model",
    "N° origine parent": "oem_ref",
    "Prix unitaire": "unit_price",
    "Coût unitaire": "unit_cost",
    "Stocks": "stock",
    "Plage de Stock": "stock_class",
    "Pareto": "pareto",
}

CLIENT_COLUMNS = {
    "N°": "client_id",
    "Nom": "client_name",
    "Client parent": "parent_id",
    "Bloqué": "blocked_detail",
    "Cause dernier blocage": "blocked_reason",
    "Type client": "client_type",
    "Ville": "city",
    "Gouvernerat": "governorate",
    "Code circuit": "circuit",
    "N° téléphone": "phone_1",
    "N° téléphone 2": "phone_2",
    "N° téléphone 3": "phone_3",
    "Contact": "contact",
    "Adresse": "address",
    "N° identif. intracomm.": "tax_id",
    "Solde": "balance",
    "Crédit autorisé DS": "credit_limit",
    "Code vendeur": "vendor_1",
    "<Code Vendeur 2>": "vendor_2",
    "<Code Vendeur 3>": "vendor_3",
}

ARTICLE_NUMERIC = ["unit_price", "unit_cost", "stock"]
CLIENT_NUMERIC = ["balance", "credit_limit"]

# Columns whose silent absence would disable a safety filter rather than merely
# blank a display field. _project fails loud on these and tolerates the rest.
REQUIRED_ARTICLE = ("N°", "Stocks", "Plage de Stock")
REQUIRED_CLIENT = ("N°", "Bloqué")


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _project(raw: pd.DataFrame, mapping: dict, required=()) -> pd.DataFrame:
    """Rename the columns we use, tolerating absent ones — except the required.

    Blank-filling a missing column is the right call for a display field: a
    slightly different export should still build. It is the WRONG call for a
    column that drives a filter. If "Bloque" were absent, blocked_detail would
    be "" for every row and `blocked` would be False for every client, silently
    switching off the rule that stops vendors calling non-paying accounts — with
    no exception and no log. So the filter-bearing columns fail loud, matching
    normalise_ledger in src/build_transactions.py, which raises on a shape change
    rather than inventing data.
    """
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"required columns absent from the export: {missing}. These drive "
            "scoring-time filters; blank-filling them would disable the filter "
            "silently. Fix the export or update the mapping deliberately."
        )
    present = {src: dst for src, dst in mapping.items() if src in raw.columns}
    df = raw.rename(columns=present)[list(present.values())].copy()
    for dst in mapping.values():
        if dst not in df.columns:
            df[dst] = ""
    return df[list(mapping.values())]


def normalise_articles(raw: pd.DataFrame) -> pd.DataFrame:
    df = _project(raw, ARTICLE_COLUMNS, REQUIRED_ARTICLE)
    for col in df.columns:
        if col not in ARTICLE_NUMERIC:
            df[col] = df[col].map(_clean)
    for col in ARTICLE_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["article_id"] != ""].reset_index(drop=True)


def normalise_clients(raw: pd.DataFrame) -> pd.DataFrame:
    df = _project(raw, CLIENT_COLUMNS, REQUIRED_CLIENT)
    for col in df.columns:
        if col not in CLIENT_NUMERIC:
            df[col] = df[col].map(_clean)
    for col in CLIENT_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Bloqué is empty / Facturer / Livrer / Tous. Keep the detail, add a boolean
    # so the scoring-time filter is a single unambiguous column.
    df["blocked"] = df["blocked_detail"] != ""
    return df[df["client_id"] != ""].reset_index(drop=True)


# The lookups the app and the pipelines actually join on. Projected from the
# masters rather than replaced BY them: baseline_lists and contact_queue merge the
# lookup wholesale, so a 21-column master would put phone numbers, addresses, tax
# ids and credit limits into clients_to_contact.csv - which the app renders and
# vendor_sheets.py exports to Excel for vendors.
ARTICLE_LOOKUP = ["article_id", "enseigne", "designation"]
CLIENT_LOOKUP = ["client_id", "client_name", "blocked"]


def project_articles(master: pd.DataFrame) -> pd.DataFrame:
    """articles.csv from articles_master.csv.

    Price, cost and stock are deliberately dropped. They describe the CURRENT
    state, which train_rolling.py refuses as features (FORBIDDEN_FEATURES); a
    lookup is no place to reintroduce them.
    """
    return master[ARTICLE_LOOKUP].copy()


def project_clients(master: pd.DataFrame) -> pd.DataFrame:
    """clients.csv from clients_master.csv, with blocked-ness as a real column.

    The old file encoded it as '***' inside the client's name because the Excel
    extract had nowhere else to put it. NAV has a proper field, so the convention
    can go. A missing value means NOT blocked: dropping a client from every
    vendor's queue requires NAV to actually say so.
    """
    out = master[["client_id", "client_name", "blocked"]].copy()
    out["blocked"] = (out["blocked"].astype(str).str.strip().str.lower()
                      .eq("true"))
    return out[CLIENT_LOOKUP]


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    DATA.mkdir(exist_ok=True)
    arts = normalise_articles(pd.read_excel(
        EXPORTS / "deatils articles nav.xlsx", sheet_name="Liste des articles", dtype=object))
    clis = normalise_clients(pd.read_excel(
        EXPORTS / "deatils clients nav.xlsx", sheet_name="Liste des clients", dtype=object))

    arts.to_csv(DATA / "articles_master.csv", index=False, encoding="utf-8-sig")
    clis.to_csv(DATA / "clients_master.csv", index=False, encoding="utf-8-sig")

    # The join-level lookups, so nothing downstream still depends on the old
    # January-May Excel that build_dataset.py read.
    project_articles(arts).to_csv(DATA / "articles.csv", index=False,
                                  encoding="utf-8-sig")
    project_clients(clis).to_csv(DATA / "clients.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"[lookups] articles.csv {len(arts):,} rows, "
          f"clients.csv {len(clis):,} rows "
          f"({int(project_clients(clis)['blocked'].sum()):,} blocked)")

    print("=== MASTER BUILD REPORT ===")
    print(f"articles          : {len(arts):,}")
    print(f"  categories      : {arts['category'].nunique()}")
    print(f"  vehicle makes   : {arts['vehicle_make'].nunique()}")
    print(f"  with OEM ref    : {(arts['oem_ref'] != '').mean():.1%}")
    print(f"  stock classes   : {dict(arts['stock_class'].value_counts().head(5))}")
    print(f"clients           : {len(clis):,}")
    print(f"  blocked         : {clis['blocked'].sum():,} ({clis['blocked'].mean():.1%})")
    print(f"  with a vendor   : {(clis['vendor_1'] != '').sum():,}")
    print(f"  with a phone    : {(clis['phone_1'] != '').mean():.1%}")


if __name__ == "__main__":
    main()
