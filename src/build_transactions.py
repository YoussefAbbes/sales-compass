# -*- coding: utf-8 -*-
"""
Build the day-level transaction table from the NAV sales ledger export.

Input : exported-data/vente 01-01-24 -- 28-05-2025.xlsx
        exported-data/vente 29-05-2025 -- 30-06-2026.xlsx
        (sheet "Écritures comptables article", 17 columns each)
Output: data/transactions.parquet

The two workbooks are split at Excel's 1,048,576-row ceiling, not at a data
boundary: file 1 ends 2025-05-28 and file 2 begins 2025-05-29. They concatenate
directly. Expect a third file for later periods.

Cleaning rules (mirroring src/build_dataset.py, which this supersedes):
  - IDs kept as strings, whitespace stripped (leading zeros are significant).
  - Amounts rounded to 3 decimals (TND millime precision).
  - Returns are KEPT and flagged via is_return. They are evidence about a pair,
    not noise; each analysis decides how to treat them.
  - Rows with no client or article identifier are dropped.
  - cost carries NAV's sign UNCHANGED: "Coût total (réel)" is negative while
    "Montant vente (réel)" is positive. Margin is therefore amount + cost,
    never amount - cost. Inventing a sign convention here would hide it.
"""
import io
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exported-data"
DATA = ROOT / "data"

SHEET = "Écritures comptables article"
RETURN_DOC_TYPE = "Réception retour vente"

LEDGER_COLUMNS = {
    "Date comptabilisation": "date",
    "Type document": "doc_type",
    "N° document": "doc_no",
    "N° article": "article_id",
    "Désignation article": "designation",
    "Enseigne": "enseigne",
    "N° origine": "client_id",
    "Code vendeur": "vendor_code",
    "Code magasin": "warehouse",
    "Quantité": "qty",
    "Montant vente (réel)": "amount",
    "Coût total (réel)": "cost",
    "Discount Amount": "discount",
}

ID_COLUMNS = ["doc_no", "article_id", "client_id", "vendor_code", "warehouse"]
NUMERIC_COLUMNS = ["qty", "amount", "cost", "discount"]


def _clean_id(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def normalise_ledger(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw NAV columns -> the analysis schema. Pure: no I/O, no globals."""
    entry_types = set(raw["Type écriture"].dropna().unique())
    if entry_types - {"Vente"}:
        raise ValueError(
            f"unexpected Type écriture values {sorted(entry_types - {'Vente'})}: the "
            "export is no longer sales-only and downstream assumptions need review"
        )

    df = raw.rename(columns=LEDGER_COLUMNS)[list(LEDGER_COLUMNS.values())].copy()

    df["date"] = pd.to_datetime(df["date"])
    for col in ID_COLUMNS:
        df[col] = df[col].map(_clean_id)
    df["designation"] = df["designation"].fillna("").astype(str).str.strip()
    df["enseigne"] = df["enseigne"].fillna("").astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        # astype is not redundant: when every value is integral pandas infers
        # int64, and the declared interface promises float for all four.
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                   .fillna(0.0).round(3).astype("float64"))

    df["is_return"] = df["doc_type"] == RETURN_DOC_TYPE

    df = df[(df["client_id"] != "") & (df["article_id"] != "")]
    return df.reset_index(drop=True)


def read_ledger(path: Path) -> pd.DataFrame:
    """Stream one workbook into a DataFrame.

    Kept separate from normalise_ledger so the cleaning rules stay unit-testable
    without a 121 MB fixture.

    pd.read_excel is deliberately NOT used: it routes every cell through an
    openpyxl Cell object, which for 1.05M rows x 17 columns costs several GB and
    many minutes. read_only=True + iter_rows streams raw tuples instead.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb[SHEET].iter_rows(values_only=True)
        header = list(next(rows))
        return pd.DataFrame(list(rows), columns=header)
    finally:
        wb.close()


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    DATA.mkdir(exist_ok=True)
    paths = sorted(EXPORTS.glob("vente*.xlsx"))
    if not paths:
        raise SystemExit(f"no vente*.xlsx found under {EXPORTS}")

    frames = []
    for path in paths:
        print(f"reading {path.name} ...", flush=True)
        part = normalise_ledger(read_ledger(path))
        print(f"  {len(part):,} rows  {part['date'].min():%Y-%m-%d} -> "
              f"{part['date'].max():%Y-%m-%d}")
        frames.append(part)

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df.to_parquet(DATA / "transactions.parquet", index=False)

    print("\n=== TRANSACTION BUILD REPORT ===")
    print(f"rows              : {len(df):,}")
    print(f"date range        : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"documents         : {df['doc_no'].nunique():,}")
    print(f"clients / articles / vendors : {df['client_id'].nunique():,} / "
          f"{df['article_id'].nunique():,} / {df['vendor_code'].nunique():,}")
    print(f"returns           : {df['is_return'].sum():,} ({df['is_return'].mean():.2%})")
    print(f"total amount      : {df['amount'].sum():,.3f} TND")
    print(f"total cost        : {df['cost'].sum():,.3f} TND")
    gaps = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    present = set(df["date"].dt.to_period("M").astype(str))
    missing = [f"{d:%Y-%m}" for d in gaps if f"{d:%Y-%m}" not in present]
    print(f"missing months    : {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
