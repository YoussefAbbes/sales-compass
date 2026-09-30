# -*- coding: utf-8 -*-
"""
Rolling chronological snapshots over the day-level transaction table.

One snapshot at date T:
    features from [T - LOOKBACK_MONTHS, T)      strictly before T
    label    from [T, T + HORIZON_MONTHS)       strictly at or after T
    candidates = (client, article) pairs with a real purchase in the feature window

Why rolling, and why fixed-width:

The v1 model had exactly ONE honest test window, and that single fact produced
three separate dead-ends — tuning was abandoned (a 4,000-iteration run improved a
random validation split while degrading the real test), calibration was abandoned
(an isotonic curve fitted on that split made test calibration worse), and the
queue could not be replayed. All three came from a validation split that shared a
year with training. Sixteen chronological snapshots give a genuinely out-of-time
validation set, which is what makes tuning and calibration legitimate again.

The lookback is FIXED, not expanding, for the same reason v1 used "year Y only":
with an expanding window an early snapshot sees 3 months of history and a late one
sees 30, so train and test would carry different feature distributions and the
comparison would be meaningless.

The EMBARGOES are not optional, and there are TWO of them. A snapshot's label
window runs HORIZON_MONTHS forward, so EVERY split boundary needs that many
months of dropped snapshots in front of it, not just the last one.

The val->test embargo was correct from the start. The train->val embargo was
missing until 2026-08-12, and its absence cost two tuning attempts: the last
training snapshot's label window overlapped the validation snapshots, so the
same purchase events defined labels on both sides. Validation then paid the
model for memorising those events. A 6000-iteration run duly pushed validation
AUC from 0.781 to 0.788 while the real test AUC FELL from 0.775 to 0.771 —
the identical failure this docstring already blamed for the v1 dead-ends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_MONTHS = 12
HORIZON_MONTHS = 3


def _months(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


# 16 snapshots: T = 2025-01 .. 2026-04. Earliest T needs 12 months of history
# (data starts 2024-01-02); latest T needs 3 months of future (data ends 2026-06-30).
TRAIN_T = _months("2025-01-01", "2025-06-01")     # 6  was 8, shortened for EMBARGO_TV
EMBARGO_TV = _months("2025-07-01", "2025-08-01")  # 2  discarded
VAL_T = _months("2025-09-01", "2025-10-01")       # 2  time-forward, never random
EMBARGO_VT = _months("2025-11-01", "2025-12-01")  # 2  discarded
TEST_T = _months("2026-01-01", "2026-04-01")      # 4  untouched

NUMERIC_FEATURES = [
    "n_orders", "n_months", "qty_sum", "amount_sum", "amount_per_order",
    "days_since_last", "days_since_first", "median_interval", "interval_cv",
    "overdue_ratio", "return_rate",
    "client_amount", "client_n_articles", "client_n_orders", "client_active_months",
    "client_days_since_last",
    "article_n_clients", "article_amount", "article_repurchase_share",
    "share_of_client", "share_of_article",
]
CATEGORICAL_FEATURES = ["enseigne", "vendor_code"]


def _intervals(days: np.ndarray) -> tuple[float, float]:
    """Median gap between purchase days and its coefficient of variation.

    A single purchase has no interval. Returning NaN rather than 0 keeps 'unknown'
    distinct from 'bought twice on the same day', which CatBoost handles natively.
    """
    if len(days) < 2:
        return np.nan, np.nan
    gaps = np.diff(np.sort(np.unique(days)))
    if len(gaps) == 0:
        return np.nan, np.nan
    med = float(np.median(gaps))
    mean = float(np.mean(gaps))
    cv = float(np.std(gaps) / mean) if mean > 0 else np.nan
    return med, cv


def build_snapshot_at(tx: pd.DataFrame, T: pd.Timestamp,
                      lookback_months: int = LOOKBACK_MONTHS,
                      horizon_months: int = HORIZON_MONTHS) -> pd.DataFrame:
    """One row per candidate pair at T. Pure: no I/O, no globals, no clock."""
    T = pd.Timestamp(T)
    start = T - pd.DateOffset(months=lookback_months)
    end = T + pd.DateOffset(months=horizon_months)

    # Half-open windows. A transaction exactly at T belongs to the label side.
    hist = tx[(tx["date"] >= start) & (tx["date"] < T)]
    hist = hist[(~hist["is_return"]) & (hist["amount"] > 0)]
    if hist.empty:
        return pd.DataFrame(columns=(
            ["client_id", "article_id", "snapshot", "label"]
            + NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    all_hist = tx[(tx["date"] >= start) & (tx["date"] < T)]
    hist = hist.assign(_day=(hist["date"] - start).dt.days)

    pair = (
        hist.groupby(["client_id", "article_id"])
        .agg(n_orders=("doc_no", "nunique"),
             n_months=("date", lambda s: s.dt.to_period("M").nunique()),
             qty_sum=("qty", "sum"),
             amount_sum=("amount", "sum"),
             last_date=("date", "max"),
             first_date=("date", "min"))
        .reset_index()
    )
    pair["amount_per_order"] = pair["amount_sum"] / pair["n_orders"].clip(lower=1)
    pair["days_since_last"] = (T - pair["last_date"]).dt.days
    pair["days_since_first"] = (T - pair["first_date"]).dt.days

    gaps = (
        hist.groupby(["client_id", "article_id"])["_day"]
        .apply(lambda s: _intervals(s.to_numpy()))
        .reset_index(name="_iv")
    )
    gaps[["median_interval", "interval_cv"]] = pd.DataFrame(
        gaps["_iv"].tolist(), index=gaps.index)
    pair = pair.merge(gaps.drop(columns="_iv"), on=["client_id", "article_id"])
    pair["overdue_ratio"] = pair["days_since_last"] / pair["median_interval"]

    rets = (
        all_hist[all_hist["is_return"]]
        .groupby(["client_id", "article_id"]).size().rename("n_returns").reset_index()
    )
    pair = pair.merge(rets, on=["client_id", "article_id"], how="left")
    pair["n_returns"] = pair["n_returns"].fillna(0)
    pair["return_rate"] = pair["n_returns"] / (pair["n_orders"] + pair["n_returns"])
    pair = pair.drop(columns="n_returns")

    cli = (
        hist.groupby("client_id")
        .agg(client_amount=("amount", "sum"),
             client_n_articles=("article_id", "nunique"),
             client_n_orders=("doc_no", "nunique"),
             client_active_months=("date", lambda s: s.dt.to_period("M").nunique()),
             client_last=("date", "max"))
        .reset_index()
    )
    cli["client_days_since_last"] = (T - cli["client_last"]).dt.days
    cli = cli.drop(columns="client_last")

    art = (
        hist.groupby("article_id")
        .agg(article_n_clients=("client_id", "nunique"),
             article_amount=("amount", "sum"))
        .reset_index()
    )
    repeat = (
        hist.groupby(["article_id", "client_id"])["date"]
        .apply(lambda s: s.dt.to_period("M").nunique())
        .rename("m").reset_index()
        .groupby("article_id")["m"].apply(lambda s: float((s >= 2).mean()))
        .rename("article_repurchase_share").reset_index()
    )
    art = art.merge(repeat, on="article_id", how="left")

    x = pair.merge(cli, on="client_id").merge(art, on="article_id")
    x["share_of_client"] = x["amount_sum"] / x["client_amount"]
    x["share_of_article"] = x["amount_sum"] / x["article_amount"]

    # Categoricals: the pair's dominant vendor and the article's brand, both from
    # the feature window only.
    vend = (
        hist.groupby(["client_id", "article_id", "vendor_code"])["amount"].sum()
        .reset_index().sort_values("amount", ascending=False)
        .drop_duplicates(["client_id", "article_id"])[
            ["client_id", "article_id", "vendor_code"]]
    )
    x = x.merge(vend, on=["client_id", "article_id"], how="left")
    x["vendor_code"] = x["vendor_code"].replace("", np.nan).fillna("UNK")

    ens = (hist.sort_values("date").drop_duplicates("article_id", keep="last")
           [["article_id", "enseigne"]])
    x = x.merge(ens, on="article_id", how="left")
    x["enseigne"] = x["enseigne"].replace("", np.nan).fillna("UNK")

    # Label
    fut = tx[(tx["date"] >= T) & (tx["date"] < end)]
    fut = fut[(~fut["is_return"]) & (fut["amount"] > 0)]
    won = fut[["client_id", "article_id"]].drop_duplicates().assign(label=1)
    x = x.merge(won, on=["client_id", "article_id"], how="left")
    x["label"] = x["label"].fillna(0).astype(int)

    x["snapshot"] = T
    cols = (["client_id", "article_id", "snapshot", "label"]
            + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    return x[cols].reset_index(drop=True)


def build_many(tx: pd.DataFrame, dates) -> pd.DataFrame:
    """Stack several snapshots. Rows from different T are correlated, so any split
    over the result must be by snapshot date, never random."""
    frames = [build_snapshot_at(tx, T) for T in dates]
    return pd.concat(frames, ignore_index=True)
