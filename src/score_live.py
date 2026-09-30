# -*- coding: utf-8 -*-
"""Score the live snapshot — the one window that has not happened yet.

train_rolling.py scores the TEST window (2026-01..04) to prove the model works.
That output is a backtest: every row carries the outcome. This module exists
because a vendor needs the opposite — the next 90 days, unlabelled.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshots import (  # noqa: E402
    CATEGORICAL_FEATURES, HORIZON_MONTHS, NUMERIC_FEATURES, build_snapshot_at,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def live_snapshot_date(tx: pd.DataFrame) -> pd.Timestamp:
    """First day of the month AFTER the last observed transaction.

    Derived, never hardcoded, so the pipeline stays correct when the next NAV
    export lands. A snapshot at T draws features strictly before T, so the last
    day of data must fall before T — hence "the month after", not "the month of".
    """
    last = pd.Timestamp(tx["date"].max())
    return (last + pd.offsets.MonthBegin(1)).normalize()


def live_window_record(T: pd.Timestamp, n_pairs: int, generated_at: str) -> dict:
    """The window this score file answers for, as a record rather than a guess.

    Nothing in pair_scores_live.csv records T: the columns are features, and the
    date they were cut at is lost the moment the file is written. A consumer left
    to infer it reaches for whatever date-shaped thing is nearest — which is how
    the API came to describe live recommendations using the TRAINING metrics'
    snapshot span.

    Pure, and the horizon comes from snapshots rather than a local literal, so a
    retrain that changes the horizon re-labels this by itself.
    """
    end = T + pd.DateOffset(months=HORIZON_MONTHS) - pd.Timedelta(days=1)
    return {
        "snapshot_date": f"{T:%Y-%m-%d}",
        "window_end": f"{end:%Y-%m-%d}",
        "horizon_months": HORIZON_MONTHS,
        "n_pairs": int(n_pairs),
        "generated_at": generated_at,
    }


# Carried through from the snapshot alongside the probability. amount_sum is the
# pair's revenue ON THE LEDGER, which lets the queue compute expected value without
# borrowing ca_2025 from the older Jan-May extract basis — the two disagree by up to
# 9.89% (reports/reconciliation.md). The recency and cadence columns back the
# rule-based reason text a vendor reads next to the suggestion.
CONTEXT_COLUMNS = ["amount_sum", "days_since_last", "median_interval", "n_orders"]
LIVE_COLUMNS = ["client_id", "article_id", "vendor_code", "p"] + CONTEXT_COLUMNS


def score_live(tx: pd.DataFrame, model, T: pd.Timestamp) -> pd.DataFrame:
    """One row per candidate pair at T, with the model's probability.

    Pure: no I/O, no clock. The label column produced by build_snapshot_at is
    DROPPED rather than ignored — see test_live_scores_carry_no_label_column.
    """
    snap = build_snapshot_at(tx, T)
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing = [c for c in feats if c not in snap.columns]
    if missing:
        raise ValueError(
            f"snapshot at {T:%Y-%m-%d} is missing model features: {missing}")

    out = snap[["client_id", "article_id", "vendor_code"] + CONTEXT_COLUMNS].copy()
    out["p"] = model.predict_proba(snap[feats])[:, 1]
    return out[LIVE_COLUMNS]


MODEL_PATH = DATA / "model_rolling.cbm"
OUT_PATH = DATA / "pair_scores_live.csv"
WINDOW_PATH = REPORTS / "live_window.json"


def main() -> None:
    from catboost import CatBoostClassifier

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"model not found: {MODEL_PATH}\n"
            "Run src/train_rolling.py first — this module scores, it does not train.")
    tx_path = DATA / "transactions.parquet"
    if not tx_path.exists():
        raise SystemExit(f"ledger not found: {tx_path}")

    tx = pd.read_parquet(tx_path)
    T = live_snapshot_date(tx)
    end = T + pd.DateOffset(months=HORIZON_MONTHS) - pd.Timedelta(days=1)
    print(f"[live] last transaction {tx['date'].max():%Y-%m-%d} -> T={T:%Y-%m-%d}")
    print(f"[live] predicting purchases {T:%Y-%m} .. {end:%Y-%m}")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    out = score_live(tx, model, T)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[live] {len(out):,} pairs scored with {MODEL_PATH.name}")
    print(f"[written] {OUT_PATH}")

    # Written alongside the scores, never separately: the record is only true of
    # the file produced in this same run.
    REPORTS.mkdir(parents=True, exist_ok=True)
    record = live_window_record(
        T, len(out), datetime.now().isoformat(timespec="seconds"))
    WINDOW_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[written] {WINDOW_PATH}")


if __name__ == "__main__":
    main()
