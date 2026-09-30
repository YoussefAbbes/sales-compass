# -*- coding: utf-8 -*-
"""
Prove the new ledger reproduces the monthly table the validated system was built on.

Writes reports/reconciliation.md (aggregates only — safe to commit).

Two questions, in order:
  1. Do the identifier spaces match? If client/article/vendor ids do not overlap,
     nothing downstream is comparable and the run stops here.
  2. For the 15 overlapping Jan-May months, does the ledger aggregated to
     (month) -> ca reproduce ca_monthly.parquet?

Exact agreement is not expected to the millime: the v1 extract was a pre-aggregated
report and may apply its own rounding or scope. A per-month relative gap under
RELATIVE_TOLERANCE is treated as agreement; anything larger is reported as a
discrepancy for a human to judge.
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

RELATIVE_TOLERANCE = 0.01  # 1% per month

out = []


def emit(s=""):
    print(s)
    out.append(s)


def reconcile_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    """Ledger -> v1 monthly grain. Returns are netted in, matching v1 semantics."""
    df = tx.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")
    agg = df.groupby("month", as_index=False)["amount"].sum()
    agg["ca"] = agg["amount"].round(3)
    return agg[["month", "ca"]]


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    tx = pd.read_parquet(DATA / "transactions.parquet")
    v1 = pd.read_parquet(DATA / "ca_monthly.parquet")

    emit("# Reconciliation — NAV ledger vs the v1 monthly extract")
    emit()

    # ---- 1. Identifier spaces ----
    emit("## Identifier overlap")
    emit()
    emit("| Entity | v1 | ledger | shared | v1 covered |")
    emit("| --- | --- | --- | --- | --- |")
    coverage = {}
    for label, col in (("clients", "client_id"), ("articles", "article_id"),
                       ("vendors", "vendor_code")):
        a, b = set(v1[col].unique()), set(tx[col].unique())
        shared = a & b
        pct = len(shared) / max(len(a), 1)
        coverage[label] = pct
        emit(f"| {label} | {len(a):,} | {len(b):,} | {len(shared):,} | {pct:.1%} |")
    emit()

    if min(coverage.values()) < 0.5:
        emit("**STOP — identifier spaces do not match.** The ledger uses different keys "
             "than the v1 extract; no downstream comparison is meaningful until the "
             "mapping is understood.")
        (REPORTS / "reconciliation.md").write_text("\n".join(out), encoding="utf-8")
        raise SystemExit("identifier spaces do not match — see reports/reconciliation.md")

    # ---- 2. Monthly totals on the overlapping window ----
    new = reconcile_monthly(tx).rename(columns={"ca": "ca_ledger"})
    old = v1.groupby("month", as_index=False)["ca"].sum().rename(columns={"ca": "ca_v1"})
    cmp = old.merge(new, on="month", how="inner")
    cmp["gap"] = cmp["ca_ledger"] - cmp["ca_v1"]
    cmp["rel"] = cmp["gap"] / cmp["ca_v1"]

    emit("## Monthly totals on the 15 overlapping months")
    emit()
    emit("| Month | v1 CA | ledger CA | gap | rel |")
    emit("| --- | --- | --- | --- | --- |")
    for _, r in cmp.iterrows():
        emit(f"| {r['month']:%Y-%m} | {r['ca_v1']:,.0f} | {r['ca_ledger']:,.0f} | "
             f"{r['gap']:,.0f} | {r['rel']:+.2%} |")
    emit()

    worst = cmp["rel"].abs().max()
    bad = cmp[cmp["rel"].abs() > RELATIVE_TOLERANCE]
    emit(f"- months compared: **{len(cmp)}**")
    emit(f"- worst relative gap: **{worst:.2%}** (tolerance {RELATIVE_TOLERANCE:.0%})")
    emit(f"- months outside tolerance: **{len(bad)}**")
    emit()
    emit("## Months present only in the ledger")
    emit()
    extra = sorted(set(new["month"]) - set(old["month"]))
    emit(f"{len(extra)} new months: {extra[0]:%Y-%m} … {extra[-1]:%Y-%m}"
         if extra else "none")
    emit()
    emit("These are the months the v1 basis could never see. They are the reason "
         "labels stop being lower bounds.")

    (REPORTS / "reconciliation.md").write_text("\n".join(out), encoding="utf-8")
    print(f"\n[written] reports/reconciliation.md")

    if len(bad):
        raise SystemExit(
            f"{len(bad)} month(s) outside the {RELATIVE_TOLERANCE:.0%} tolerance — "
            "review reports/reconciliation.md before building snapshots")


if __name__ == "__main__":
    main()
