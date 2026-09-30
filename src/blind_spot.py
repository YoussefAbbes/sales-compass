# -*- coding: utf-8 -*-
"""
Measure how much the January-May label window understated repurchase.

Every rate this project has reported was qualified as a lower bound, because the
v1 extract could only observe January-May and a pair repurchased in September was
scored as a failure. The NAV ledger covers all twelve months, so the size of that
understatement is now measurable instead of assumed.

This is deliberately a DATA measurement, not a model re-evaluation. The clean
model version is unavailable: the v1 test window is 2025 -> 2026 and the ledger
stops at 2026-06-30, so extending it would buy a single month. The base-rate
comparison below needs no model and is exact.

Writes reports/blind_spot.md (aggregates only — safe to commit).
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

out = []


def emit(s=""):
    print(s)
    out.append(s)


def _bought(tx: pd.DataFrame, year: int, months=None) -> pd.DataFrame:
    """Distinct (client, article) pairs with a real purchase in the window."""
    sel = tx[(tx["date"].dt.year == year) & (~tx["is_return"]) & (tx["amount"] > 0)]
    if months is not None:
        sel = sel[sel["date"].dt.month.isin(months)]
    return sel[["client_id", "article_id"]].drop_duplicates()


def repurchase_rates(tx: pd.DataFrame, feature_year: int, label_year: int) -> dict:
    """Share of pairs active in Jan-May of feature_year that came back in
    label_year — measured the way v1 could see it, and the way it really was."""
    base = _bought(tx, feature_year, months=range(1, 6))
    if base.empty:
        return {"n_pairs": 0, "rate_jan_may": 0.0, "rate_full_year": 0.0,
                "understatement": float("nan")}

    jan_may = _bought(tx, label_year, months=range(1, 6))
    full = _bought(tx, label_year)

    keys = ["client_id", "article_id"]
    seen_jm = base.merge(jan_may.assign(hit=1), on=keys, how="left")["hit"].notna()
    seen_fy = base.merge(full.assign(hit=1), on=keys, how="left")["hit"].notna()

    r_jm, r_fy = float(seen_jm.mean()), float(seen_fy.mean())
    return {
        "n_pairs": int(len(base)),
        "rate_jan_may": r_jm,
        "rate_full_year": r_fy,
        "understatement": (r_fy / r_jm) if r_jm else float("nan"),
    }


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    tx = pd.read_parquet(DATA / "transactions.parquet")

    emit("# The January-May blind spot, measured")
    emit()
    emit("Pairs active in January-May of the feature year, and whether they came back "
         "in the label year — as the v1 basis could see it (January-May only) versus "
         "the full twelve months the NAV ledger now covers.")
    emit()
    emit("| Feature year | Label year | Pairs | Rate Jan-May | Rate full year | Understated by |")
    emit("| --- | --- | --- | --- | --- | --- |")

    results = {}
    for fy, ly in ((2024, 2025), (2025, 2026)):
        r = repurchase_rates(tx, fy, ly)
        results[f"{fy}->{ly}"] = r
        note = " *" if ly == 2026 else ""
        emit(f"| {fy} | {ly}{note} | {r['n_pairs']:,} | {r['rate_jan_may']:.1%} | "
             f"{r['rate_full_year']:.1%} | ×{r['understatement']:.2f} |")
    emit()
    emit("\\* 2026 is only observed to 30 June, so its full-year figure is itself a "
         "lower bound. The 2024->2025 row is the complete measurement.")
    emit()

    r = results["2024->2025"]
    emit("## What this means for the reported numbers")
    emit()
    emit(f"- Observed repurchase rate, January-May only: **{r['rate_jan_may']:.1%}**")
    emit(f"- True repurchase rate over the full year: **{r['rate_full_year']:.1%}**")
    emit(f"- The v1 label understated repurchase by a factor of "
         f"**×{r['understatement']:.2f}**.")
    emit()
    emit("Consequence for training, which matters more than the reported rate: pairs "
         "that did repurchase outside January-May were labelled 0 and taught the model "
         "that their pattern does not lead to a purchase. That is label noise in the "
         "target, and no amount of feature work or tuning could have corrected it.")

    (REPORTS / "blind_spot.md").write_text("\n".join(out), encoding="utf-8")
    print("\n[written] reports/blind_spot.md")


if __name__ == "__main__":
    main()
