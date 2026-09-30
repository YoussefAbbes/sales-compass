# -*- coding: utf-8 -*-
"""Metric computation for the model-validation page.

Deliberately free of any Streamlit import so it can be unit-tested without a browser
or a running server. The page (src/pages/1_Validation.py) is only a display layer.

Vocabulary used throughout:
  p      model probability that a (client, article) pair is bought again
  y      the truth: 1 if the pair was actually bought in the label year
  split  "test" = the four untouched 2026-01..04 snapshots scored by the v4 model
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# This module is imported by BOTH the API (api/routes/validation.py) and the Streamlit
# page (src/pages/1_Validation.py), so the override has to live here too — otherwise
# DATA_ROOT=fixtures would still serve the real data/ on the four validation endpoints.
# Unset resolves to the repo root: Streamlit sees no change.
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"
REPORTS = DATA_ROOT / "reports"

# v4 only. pair_scores_2026.csv came from the v1 model, whose validation set was
# contaminated (reports/tuning-log.md); showing its numbers under a heading that
# says "Validation" is the one thing this page must not do. v4 writes no separate
# train-split score file, so "test" is the only split with scored pairs.
SCORE_FILES = {
    "test": DATA / "pair_scores_rolling.csv",
}


def _window(months) -> str:
    """['2025-01', '2025-02'] -> '2025-01 … 2025-02'."""
    return f"{months[0]} … {months[-1]}" if months else "—"


def load_metrics() -> dict:
    """reports/rolling_metrics.json, reshaped for display.

    train_rolling.py writes a flat record; the page wants one entry per split.
    Reshaping here keeps the display layer free of JSON archaeology, and lets the
    two embargoes be first-class rather than something a reader has to infer.
    """
    path = REPORTS / "rolling_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `py src/train_rolling.py` to generate it.")
    raw = json.loads(path.read_text(encoding="utf-8"))

    splits, rows, auc = raw["splits"], raw["rows"], raw["auc"]
    out = {}
    for name in ("train", "validation", "test"):
        out[name] = {
            "window": _window(splits[name]),
            "n_snapshots": len(splits[name]),
            "n_pairs": rows[name],
            "auc": auc[name],
        }
    # Dropped on purpose, and named so a reader sees the cost of the design.
    for name in ("embargo_train_val", "embargo_val_test"):
        out[name] = {"window": _window(splits[name]),
                     "n_snapshots": len(splits[name])}

    out["base_rate_test"] = raw["base_rate_test"]
    out["gap_train_test"] = auc["train"] - auc["test"]
    # The number that proves the leak is gone. On the contaminated 6000-iteration
    # run this was +0.0163 and GREW with training; it is now slightly negative,
    # i.e. validation is pessimistic about test, which is the safe direction.
    out["gap_validation_test"] = auc["validation"] - auc["test"]

    for k in ("iterations_cap", "early_stopping_rounds", "best_iteration",
              "converged", "auc_per_snapshot", "precision_top30_model",
              "precision_top30_revenue", "feature_importance",
              "lookback_months", "horizon_months"):
        out[k] = raw.get(k)
    return out


def load_scores(split: str = "test") -> pd.DataFrame:
    """Scored pairs for one split, with the true label attached."""
    if split not in SCORE_FILES:
        raise ValueError(f"split must be one of {sorted(SCORE_FILES)}, got {split!r}")
    path = SCORE_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `py src/train_model.py` to generate it.")
    df = pd.read_csv(
        path,
        dtype={"client_id": str, "article_id": str, "vendor_code": str},
        encoding="utf-8-sig",
    )
    # train_rolling speaks `p` and `amount_sum`; this page and queue_replay have
    # always spoken `p_model` and `ca_sum`. Rename here rather than at a dozen
    # display call sites.
    return df.rename(columns={"p": "p_model", "amount_sum": "ca_sum"})


def calibration_table(p: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Are the probabilities true? Bin by predicted p, compare to observed frequency.

    `gap` = mean_pred - observed. Positive means the model is over-confident in that bin.
    Empty bins are dropped, so the result may have fewer than n_bins rows.
    """
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    d = pd.DataFrame({"p": p, "y": y, "bin": idx})
    g = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                             observed=("y", "mean")).reset_index()
    g["bin_low"] = edges[g["bin"].to_numpy()]
    g["bin_high"] = edges[g["bin"].to_numpy() + 1]
    g["gap"] = g["mean_pred"] - g["observed"]
    return g[["bin_low", "bin_high", "n", "mean_pred", "observed", "gap"]]


def decile_table(p: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Repurchase rate per probability decile (10 = most confident)."""
    d = pd.DataFrame({"p": pd.Series(p).reset_index(drop=True),
                      "y": pd.Series(y).reset_index(drop=True)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    g = (d.groupby("decile", observed=True)
           .agg(n=("y", "size"), rate=("y", "mean"))
           .reset_index())
    g["decile"] = g["decile"].astype(int)
    return g.sort_values("decile").reset_index(drop=True)


def confusion_at(p: pd.Series, y: pd.Series, threshold: float) -> dict:
    """Confusion matrix treating p >= threshold as 'will buy'."""
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    tn = int(((~flagged) & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_flagged": int(flagged.sum()),
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
    }


# --------------------------------------------------------------------------
# Recommendation-level grading.
#
# IMPORTANT — this is an approximation of the production queue, not a replay.
# src/backtest.py documents why a literal replay is impossible: the lapsed rule
# needs two prior years and 2023 is not in the extract. Replicated here:
#   - expected-value ranking (p_model x prior-year CA)
#   - the both-years eligibility restriction
#   - top-k articles per client
# NOT replicated (both need a 2023 basis that does not exist):
#   - the lapsed filter (bought in prior years, absent this year)
#   - the client status boost from the reactivation list
# The page must show this caveat in the UI.
# --------------------------------------------------------------------------

def eligible_pairs(fact: pd.DataFrame, years: tuple) -> pd.DataFrame:
    """(client, article) pairs with a positive-CA month in BOTH given years."""
    y1, y2 = years
    pos = fact[fact["ca"] > 0].copy()
    pos["year"] = pos["month"].dt.year
    a = pos[pos["year"] == y1][["client_id", "article_id"]].drop_duplicates()
    b = pos[pos["year"] == y2][["client_id", "article_id"]].drop_duplicates()
    return a.merge(b, on=["client_id", "article_id"], how="inner").reset_index(drop=True)


def queue_replay(scores: pd.DataFrame, eligible: pd.DataFrame,
                 top_k: int = 3) -> pd.DataFrame:
    """Top-k articles per client, ranked by expected value, within the eligible pool."""
    df = scores.merge(eligible[["client_id", "article_id"]],
                      on=["client_id", "article_id"], how="inner").copy()
    df["valeur_attendue"] = df["p_model"] * df["ca_sum"]
    df = df.sort_values(["client_id", "valeur_attendue"], ascending=[True, False])
    df["rank_in_client"] = df.groupby("client_id").cumcount() + 1
    return df[df["rank_in_client"] <= top_k].reset_index(drop=True)


def precision_at_n(df: pd.DataFrame, score_col: str, n: int,
                   group_col: str = "vendor_code") -> dict:
    """If each vendor works only their top-n pairs, what share were actually bought?"""
    top = (df.sort_values(score_col, ascending=False)
             .groupby(group_col, group_keys=False)
             .head(n))
    n_pairs = int(len(top))
    n_hits = int(top["label"].sum())
    return {
        "n": n,
        "n_pairs": n_pairs,
        "n_hits": n_hits,
        "precision": (n_hits / n_pairs) if n_pairs else 0.0,
        "n_groups": int(top[group_col].nunique()),
    }


def realized_ca(fact: pd.DataFrame, year: int) -> pd.DataFrame:
    """Actual positive CA per (client, article) in a given year — what really happened."""
    pos = fact[(fact["ca"] > 0) & (fact["month"].dt.year == year)]
    return (pos.groupby(["client_id", "article_id"], as_index=False)["ca"]
               .sum()
               .rename(columns={"ca": "ca_realized"}))


# --------------------------------------------------------------------------
# Probability calibration.
#
# The raw model probabilities are over-confident: in the 80-90% band it announces
# ~84% where reality is ~74%. train_model.py fits an isotonic curve on the
# validation split (data the model never learned, and never the test window) and
# stores it in reports/model_metrics.json. Applying it makes a displayed "80 %"
# mean roughly 80 %, which is what allows the vendor sheet to show percentages at
# all instead of Élevée/Moyenne/Faible labels.
#
# Isotonic regression is monotonic, so applying it never reorders anything.
# --------------------------------------------------------------------------

def load_calibration() -> dict | None:
    """The stored isotonic curve, or None if the model predates calibration."""
    try:
        return load_metrics().get("calibration")
    except FileNotFoundError:
        return None


def apply_calibration(p, calibration: dict | None):
    """Map raw probabilities through the curve. Returns them unchanged if absent.

    Values outside the fitted range are clipped to its ends, matching the
    `out_of_bounds="clip"` setting used when fitting.
    """
    values = np.asarray(p, dtype=float)
    if not calibration or not calibration.get("x"):
        return values
    return np.interp(values, np.asarray(calibration["x"], dtype=float),
                     np.asarray(calibration["y"], dtype=float))
