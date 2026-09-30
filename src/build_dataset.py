# -*- coding: utf-8 -*-
"""
Build the clean long-format dataset from the raw CA Excel extract.

Input : CA client vendeur article Janv-mai  24-25-26.xlsx  (2 sheets)
Output: data/ca_monthly.parquet   fact table: vendor_code, client_id, article_id, month, ca
        data/vendors.csv          vendor_code -> vendor_name
        data/clients.csv          client_id   -> client_name
        data/articles.csv         article_id  -> enseigne, designation

Cleaning rules (see docs/background/progress.md):
  - IDs kept as strings (leading zeros preserved), whitespace stripped.
  - '-' / empty cells = no observation -> not stored (sparse fact table).
  - CA rounded to 3 decimals (TND millime precision) to remove float artifacts.
  - Rows whose rounded CA is exactly 0.000 are dropped (pure rounding noise).
  - Duplicate (vendor, client, article, month) keys are summed — the Mai sheet
    splits some keys across several rows (partial amounts / correction rows).
  - Negative CA (net returns / credit notes) is KEPT and flagged downstream.
"""
import io
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "CA client vendeur article Janv-mai  24-25-26.xlsx"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Month columns of sheet 1, in file order -> (year, month)
SHEET1_MONTHS = [
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2), (2025, 3), (2025, 4),
    (2026, 1), (2026, 2), (2026, 3), (2026, 4),
]
SHEET2_MONTHS = [(2024, 5), (2025, 5), (2026, 5)]


def clean_id(v) -> str:
    return str(v).strip() if v is not None else ""


def to_ca(v):
    """Return CA rounded to millimes, or None if the cell is empty ('-')."""
    if isinstance(v, (int, float)):
        return round(float(v), 3)
    return None  # '-' or None or unexpected text


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, read_only=True)

    facts = []          # (vendor_code, client_id, article_id, month, ca)
    dropped_zero = 0

    # ---- Sheet 1: Janv-fev-mars-avril (codes only, 12 month columns) ----
    ws = wb["Janv-fev-mars-avril"]
    rows = ws.iter_rows(values_only=True)
    next(rows)  # header
    for row in rows:
        vendor, client, article = clean_id(row[0]), clean_id(row[1]), clean_id(row[2])
        if not (vendor or client or article):
            continue
        for (year, month), cell in zip(SHEET1_MONTHS, row[3:15]):
            ca = to_ca(cell)
            if ca is None:
                continue
            if ca == 0.0:
                dropped_zero += 1
                continue
            facts.append((vendor, client, article, date(year, month, 1), ca))

    # ---- Sheet 2: Mai (has names + designations, 3 month columns) ----
    ws2 = wb["Mai"]
    rows2 = ws2.iter_rows(values_only=True)
    next(rows2)  # header
    vendor_names = {}                       # code -> name (last seen)
    client_names = defaultdict(Counter)     # id -> name frequencies
    article_info = defaultdict(Counter)     # id -> (enseigne, designation) frequencies
    for row in rows2:
        vendor, client, article = clean_id(row[0]), clean_id(row[2]), clean_id(row[4])
        if not (vendor or client or article):
            continue
        if row[1]:
            vendor_names[vendor] = str(row[1]).strip()
        if row[3]:
            client_names[client][str(row[3]).strip()] += 1
        enseigne = str(row[5]).strip() if row[5] else ""
        designation = str(row[6]).strip() if row[6] else ""
        if enseigne or designation:
            article_info[article][(enseigne, designation)] += 1
        for (year, month), cell in zip(SHEET2_MONTHS, row[7:10]):
            ca = to_ca(cell)
            if ca is None:
                continue
            if ca == 0.0:
                dropped_zero += 1
                continue
            facts.append((vendor, client, article, date(year, month, 1), ca))

    wb.close()

    # ---- Fact table ----
    df = pd.DataFrame(facts, columns=["vendor_code", "client_id", "article_id", "month", "ca"])
    df["month"] = pd.to_datetime(df["month"])

    # Sum duplicate keys (Mai sheet splits some keys across several rows),
    # then drop keys whose net CA is 0.000 after aggregation.
    dup = df.duplicated(subset=["vendor_code", "client_id", "article_id", "month"]).sum()
    df = (
        df.groupby(["vendor_code", "client_id", "article_id", "month"], as_index=False)["ca"]
        .sum()
    )
    df["ca"] = df["ca"].round(3)
    net_zero = (df["ca"] == 0.0).sum()
    df = df[df["ca"] != 0.0].reset_index(drop=True)
    df.to_parquet(DATA / "ca_monthly.parquet", index=False)

    # ---- Lookup tables (most frequent name wins when several exist) ----
    pd.DataFrame(
        sorted(vendor_names.items()), columns=["vendor_code", "vendor_name"]
    ).to_csv(DATA / "vendors.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [(cid, names.most_common(1)[0][0]) for cid, names in sorted(client_names.items())],
        columns=["client_id", "client_name"],
    ).to_csv(DATA / "clients.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [(aid, *info.most_common(1)[0][0]) for aid, info in sorted(article_info.items())],
        columns=["article_id", "enseigne", "designation"],
    ).to_csv(DATA / "articles.csv", index=False, encoding="utf-8-sig")

    # ---- Validation report ----
    print("=== BUILD REPORT ===")
    print(f"fact rows           : {len(df):,}")
    print(f"dropped 0.000 cells : {dropped_zero:,}")
    print(f"duplicate keys summed: {dup}  | net-zero keys dropped: {net_zero}")
    print(f"vendors / clients / articles : "
          f"{df.vendor_code.nunique()} / {df.client_id.nunique():,} / {df.article_id.nunique():,}")
    print(f"negative-CA rows    : {(df.ca < 0).sum():,}  "
          f"({(df.ca < 0).mean():.1%} of observations)")
    print(f"lookup coverage     : vendors {len(vendor_names)}, "
          f"clients {len(client_names):,}, articles {len(article_info):,}")
    print("\nCA total per month (conservation check vs. source sheets):")
    totals = df.groupby("month")["ca"].agg(["sum", "size"])
    for m, r in totals.iterrows():
        print(f"  {m:%Y-%m}  total={r['sum']:>14,.3f}  rows={int(r['size']):>8,}")


if __name__ == "__main__":
    main()
