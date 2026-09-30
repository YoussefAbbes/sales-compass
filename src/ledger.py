# -*- coding: utf-8 -*-
"""Monthly revenue straight from the NAV ledger.

This exists to replace `data/ca_monthly.parquet`, which only ever contained
January-May of 2024, 2025 and 2026 — all the v1 extract covered. Everything
derived from it inherited that hole silently:

  * "Dernier achat" showed May even when the client bought in October, and the
    call script then told the client "plus rien depuis mai" to their face;
  * the history chart showed five months a year and had to apologise for it;
  * the cross-sell "already owns this" set missed half the year, so an article
    the client had bought in September could be suggested as new.

The ledger has every month from 2024-01 to 2026-06, so all three read from here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["date", "vendor_code", "client_id", "article_id", "amount"]


def monthly_ca(tx: pd.DataFrame) -> pd.DataFrame:
    """Revenue per client, article and month. Pure.

    Returns the same schema `ca_monthly.parquet` provided — vendor_code,
    client_id, article_id, month, ca — so it is a drop-in replacement.

    vendor_code stays in the grouping key because ownership is per vendor: 78% of
    clients are served by more than one, so collapsing them would hand one
    vendor another's business.
    Returns are negative amounts on the ledger and net off within the month,
    which is what lets the history chart show a negative bar as it always has.
    """
    out = tx[COLUMNS].copy()
    out["month"] = out["date"].values.astype("datetime64[M]")
    return (out.groupby(["vendor_code", "client_id", "article_id", "month"],
                        as_index=False)["amount"]
               .sum()
               .rename(columns={"amount": "ca"}))


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    """Rebuild data/ca_monthly.parquet from the NAV ledger.

    Replaces the January-May aggregate that build_dataset.py derived from the old
    Excel extract. Same schema, same filename, 30 months instead of 15.

    The January-May aggregate is gone entirely: the adoption model that depended
    on it was retrained on half-year windows drawn from this ledger.
    """
    src = DATA / "transactions.parquet"
    if not src.exists():
        raise SystemExit(f"ledger not found: {src}")

    out = monthly_ca(pd.read_parquet(src, columns=COLUMNS))
    out.to_parquet(DATA / "ca_monthly.parquet", index=False)

    # to_period, not strftime: this is a summary line over ~1.75M rows, and a
    # per-row format call costs more than writing the parquet it describes.
    months = out["month"].dt.to_period("M")
    print(f"[ca_monthly] {len(out):,} rows, {months.nunique()} months "
          f"({months.min()} -> {months.max()})")
    print(f"[ca_monthly] total CA {out['ca'].sum():,.0f} TND")
    print(f"[written] {DATA / 'ca_monthly.parquet'}")


if __name__ == "__main__":
    main()
