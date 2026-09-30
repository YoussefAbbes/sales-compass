# -*- coding: utf-8 -*-
"""
Train on rolling chronological snapshots.

Two models, reported separately and never compared to one another:

  1. CONTINUITY BENCHMARK — the v1 question ("pair active Jan-May of Y, bought
     again in Jan-May of Y+1?") re-run on the NAV data with full-year labels.
     This is the only fair "did the data help?" comparison against v1's 0.762.

  2. ROLLING 90-DAY MODEL — the new product. Its AUC is NOT comparable to 0.762:
     a shorter horizon means a lower base rate and a harder problem, so the
     number may be lower while the system is better. That is why the continuity
     benchmark exists.

Tuning is permitted here, unlike in train_model.py. The reason the v1 script
banned it was a validation split drawn randomly from the training year; the split
below is time-forward, with an embargo, so improving on it is evidence rather
than memorisation.

Outputs:
  reports/model_rolling.md      aggregate metrics (safe for the repo)
  reports/rolling_metrics.json  machine-readable
  data/model_rolling.cbm        trained 90-day model
  data/pair_scores_rolling.csv  test-window scores
"""
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from article_prospects import recency_damping  # noqa: E402
from snapshots import (  # noqa: E402
    CATEGORICAL_FEATURES, EMBARGO_TV, EMBARGO_VT, HORIZON_MONTHS, LOOKBACK_MONTHS,
    NUMERIC_FEATURES, TEST_T, TRAIN_T, VAL_T, build_snapshot_at,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Current-state master columns. Never trainable — see src/build_masters.py.
FORBIDDEN_FEATURES = {
    "blocked", "blocked_detail", "blocked_reason", "stock", "stock_class",
    "balance", "credit_limit", "pareto",
}

out = []


def emit(s=""):
    print(s)
    out.append(s)


def early_stopping_fired(best_iteration: int, iterations_cap: int,
                         early_stopping_rounds: int) -> bool:
    """Did training stop because validation stopped improving, or because it ran
    out of iterations?

    CatBoost only stops early once `early_stopping_rounds` consecutive iterations
    have failed to improve. So early stopping fired iff the best iteration was
    followed by that many more inside the cap. The predicate this replaced was
    `best < cap - 1`, which is true for every outcome except a best on the very
    last iteration — it could not detect a cap-limited run at all, and reported
    convergence for the 6000-iteration run that had in fact simply run out.
    """
    return best_iteration + early_stopping_rounds < iterations_cap


def assert_no_current_state_features(features) -> None:
    bad = sorted(set(features) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(
            f"current-state master columns cannot be training features: {bad}. "
            "They describe today, not the snapshot date, so a model would learn "
            "the outcome it is meant to predict. Use them as scoring-time filters."
        )


def decile_table(scores, labels) -> pd.Series:
    d = pd.DataFrame({"s": scores, "y": labels})
    d["dec"] = pd.qcut(d["s"].rank(method="first"), 10, labels=range(1, 11))
    return d.groupby("dec", observed=True)["y"].mean()


def precision_top_n(df_, score_col, n=30) -> float:
    top = (df_.sort_values(score_col, ascending=False)
           .groupby("vendor_code", group_keys=False).head(n))
    return float(top["label"].mean())


def damped_score(df_) -> pd.Series:
    """`p` re-weighted the way the delivery surfaces actually rank it.

    precision_top30_model ranks by raw `p`, and no product surface does: block A
    and the call queue both damp for recency (article_prospects.recency_damping),
    because a client on a 12-day cadence who ordered yesterday is a near-certain
    repurchase that needs no phone call. Reporting only the undamped figure means
    the model card describes a ranking nobody serves -- and the gap is not small,
    so a reader benchmarking the deliverable on it would read a deliberate design
    decision as a regression.

    Degrades to the raw score when the frame carries no cadence columns, so the
    metric is never a precondition on columns a caller may not have.
    """
    if not {"days_since_last", "median_interval"} <= set(df_.columns):
        return df_["p"].astype(float)
    damp = [recency_damping(d, m) for d, m
            in zip(df_["days_since_last"], df_["median_interval"])]
    return df_["p"].astype(float) * pd.Series(damp, index=df_.index)


# The 1500-iteration baseline never triggered early stopping — validation AUC was
# still rising at the last iteration, so the cap, not convergence, set the result.
# Raising it is the one change the evidence actually supports. Overridable so the
# baseline stays reproducible: ROLLING_ITERATIONS=1500 reproduces it exactly.
ITERATIONS = int(os.environ.get("ROLLING_ITERATIONS", "1500"))
EARLY_STOPPING = int(os.environ.get("ROLLING_EARLY_STOPPING", "100"))


def fit(train, val, feats, cats, iterations=None) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=iterations or ITERATIONS, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=200, early_stopping_rounds=EARLY_STOPPING,
    )
    model.fit(
        Pool(train[feats], train["label"], cat_features=cats),
        eval_set=Pool(val[feats], val["label"], cat_features=cats),
        use_best_model=True,
    )
    return model


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    out.clear()  # module-level accumulator: a second main() would double the report
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert_no_current_state_features(feats)

    tx = pd.read_parquet(DATA / "transactions.parquet")

    emit("# Rolling-snapshot model")
    emit()
    emit(f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
         f"Lookback {LOOKBACK_MONTHS} months, horizon {HORIZON_MONTHS} months.")
    emit()
    emit("| Split | Snapshots | Range |")
    emit("| --- | --- | --- |")
    for name, dates in (("train", TRAIN_T),
                        ("embargo train→val (dropped)", EMBARGO_TV),
                        ("validation", VAL_T),
                        ("embargo val→test (dropped)", EMBARGO_VT),
                        ("test", TEST_T)):
        emit(f"| {name} | {len(dates)} | {min(dates):%Y-%m} … {max(dates):%Y-%m} |")
    emit()

    # One snapshot costs minutes against 1.88M rows and this needs 14 of them, so
    # cache each to disk. A crashed or repeated run must not pay that hour twice.
    cache = DATA / "snapshots"
    cache.mkdir(exist_ok=True)

    def cached(dates, name):
        frames = []
        for T in dates:
            f = cache / f"snapshot-{T:%Y-%m}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f))
                print(f"  {T:%Y-%m}  cached", flush=True)
            else:
                t0 = time.time()
                snap = build_snapshot_at(tx, T)
                snap.to_parquet(f, index=False)
                frames.append(snap)
                print(f"  {T:%Y-%m}  built {len(snap):>7,} rows in {time.time()-t0:.0f}s",
                      flush=True)
        total = sum(len(f) for f in frames)
        print(f"{name}: {total:,} rows from {len(dates)} snapshots", flush=True)
        return pd.concat(frames, ignore_index=True)

    print("building snapshots ...", flush=True)
    train = cached(TRAIN_T, "train")
    val = cached(VAL_T, "validation")
    test = cached(TEST_T, "test")

    for cat in CATEGORICAL_FEATURES:
        for frame in (train, val, test):
            frame[cat] = frame[cat].astype(str)

    emit("| Split | Rows | Positive rate |")
    emit("| --- | --- | --- |")
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        emit(f"| {name} | {len(frame):,} | {frame['label'].mean():.2%} |")
    emit()

    model = fit(train, val, feats, CATEGORICAL_FEATURES)
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        frame["p"] = model.predict_proba(Pool(frame[feats],
                                              cat_features=CATEGORICAL_FEATURES))[:, 1]

    auc = {name: float(roc_auc_score(f["label"], f["p"]))
           for name, f in (("train", train), ("validation", val), ("test", test))}

    emit("## AUC")
    emit()
    emit("| Split | AUC |")
    emit("| --- | --- |")
    for name in ("train", "validation", "test"):
        emit(f"| {name} | {auc[name]:.3f} |")
    emit()
    emit(f"Train − test gap: **{auc['train'] - auc['test']:+.3f}**.")
    emit()
    best = model.get_best_iteration()
    converged = early_stopping_fired(best, ITERATIONS, EARLY_STOPPING)
    emit(f"Iteration cap {ITERATIONS}, early stopping {EARLY_STOPPING}, best "
         f"iteration **{best}**. "
         + ("Early stopping fired, so validation AUC had stopped improving — this "
            "is a converged result."
            if converged else
            f"**Early stopping never fired.** Only {ITERATIONS - 1 - best} "
            f"iterations followed the best one, short of the {EARLY_STOPPING} "
            "needed to confirm a plateau, so training ended by exhausting the cap "
            "rather than by converging. Read this AUC as a cap-limited figure, "
            "not a converged one."))
    emit()

    emit("> The v1 model scored AUC 0.762 on the January-May annual question. "
         "**That number is not comparable to the test AUC above.** A 90-day "
         "horizon has a lower base rate and is a harder problem, so a higher "
         "number here is not evidence of a better model on the v1 question.")
    emit(">")
    emit("> The only fair like-for-like comparison would be a continuity "
         "benchmark: v1's annual January-May question re-run on the NAV data "
         "with full-year labels. **That has not been run.** Until it is, treat "
         "0.762 and the figure above as answers to two different questions, and "
         "do not quote a delta between them.")
    emit()

    emit("## Lift by decile (test)")
    emit()
    dec = decile_table(test["p"], test["label"])
    base = test["label"].mean()
    emit("| Decile | Purchase rate | Lift |")
    emit("| --- | --- | --- |")
    for d, v in dec.items():
        emit(f"| {d} | {v:.1%} | ×{v / base:.2f} |")
    emit()
    emit(f"Base rate: {base:.1%}.")
    emit()

    emit("## Precision at the top 30 per vendor (test)")
    emit()
    emit(f"- model: **{precision_top_n(test, 'p'):.1%}**")
    emit(f"- by past revenue alone: {precision_top_n(test, 'amount_sum'):.1%}")
    emit()

    emit("## Per-snapshot test AUC")
    emit()
    emit("| Snapshot | Rows | Positive rate | AUC |")
    emit("| --- | --- | --- | --- |")
    per_snapshot = {}
    for T, grp in test.groupby("snapshot"):
        a = float(roc_auc_score(grp["label"], grp["p"])) if grp["label"].nunique() > 1 else float("nan")
        per_snapshot[f"{T:%Y-%m}"] = a
        emit(f"| {T:%Y-%m} | {len(grp):,} | {grp['label'].mean():.2%} | {a:.3f} |")
    emit()
    emit("Stability across snapshots matters more than the pooled number: a model "
         "that only works in one month is not deployable.")
    emit()

    emit("## Feature importance (top 12)")
    emit()
    imp = pd.Series(model.get_feature_importance(),
                    index=feats).sort_values(ascending=False)
    emit("| Feature | Importance |")
    emit("| --- | --- |")
    for f, v in imp.head(12).items():
        emit(f"| {f} | {v:.1f} |")
    emit()

    model.save_model(str(DATA / "model_rolling.cbm"))
    test[["client_id", "article_id", "vendor_code", "snapshot",
          "amount_sum", "p", "label"]].to_csv(
        DATA / "pair_scores_rolling.csv", index=False, encoding="utf-8-sig")

    (REPORTS / "model_rolling.md").write_text("\n".join(out), encoding="utf-8")
    (REPORTS / "rolling_metrics.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback_months": LOOKBACK_MONTHS,
        "horizon_months": HORIZON_MONTHS,
        "splits": {"train": [f"{d:%Y-%m}" for d in TRAIN_T],
                   "validation": [f"{d:%Y-%m}" for d in VAL_T],
                   "embargo_train_val": [f"{d:%Y-%m}" for d in EMBARGO_TV],
                   "embargo_val_test": [f"{d:%Y-%m}" for d in EMBARGO_VT],
                   "test": [f"{d:%Y-%m}" for d in TEST_T]},
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "iterations_cap": ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING,
        "best_iteration": int(model.get_best_iteration()),
        "converged": bool(converged),
        "base_rate_test": float(base),
        "auc": auc,
        "auc_per_snapshot": per_snapshot,
        "precision_top30_model": precision_top_n(test, "p"),
        # The served ranking. Lower than the line above by design: the pairs it
        # demotes are near-certain repurchases that would happen without a call,
        # so the undamped figure collects points the business never banks.
        "precision_top30_model_damped": precision_top_n(
            test.assign(p_damped=damped_score(test)), "p_damped"),
        "precision_top30_revenue": precision_top_n(test, "amount_sum"),
        "feature_importance": {k: float(v) for k, v in imp.items()},
        "not_comparable_to": {"v1_annual_auc": 0.762,
                              "reason": "different horizon and label definition"},
    }, indent=2), encoding="utf-8")

    print("\n[written] reports/model_rolling.md, reports/rolling_metrics.json")


if __name__ == "__main__":
    main()
