# -*- coding: utf-8 -*-
"""Train and evaluate the block-2 prospect model.

Run:  py src/train_prospect_model.py

  TRAIN : features 2024 -> label = adopted in 2025
  TEST  : features 2025 -> label = adopted in 2026   (untouched until the final evaluation)

Mirrors src/train_model.py: same split, same snapshot-year symmetry, a human-readable report
in reports/ and a machine-readable JSON beside it so the validation page and the API can read
the numbers exactly as they read model_metrics.json today.

The model must beat the incumbent heuristic on precision@5 to be worth shipping. If it does
not, that is a result: record it and keep the heuristic.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prospect_features as pf
import prospect_model as pm

# Overridable so a cut-down smoke run can prove the script end-to-end (~8 min) before
# committing to the full one (~90 min), without editing the file.
TRAIN_ARTICLES = int(os.environ.get("PROSPECT_TRAIN_ARTICLES", 4000))
EVAL_ARTICLES = int(os.environ.get("PROSPECT_EVAL_ARTICLES", 2000))
MIN_BUYERS = pf.ap.MIN_BUYERS

LINES = []


def emit(s=""):
    # flush: this script runs for tens of minutes and its stdout is usually a
    # pipe, so without it there is no way to tell slow from stuck.
    print(s, flush=True)
    LINES.append(s)


def eligible_articles(ctx, rng, n):
    """Articles with enough buyers to have a signature, sampled without replacement."""
    counts = ctx.article_feats["article_n_buyers"]
    pool = list(counts[counts >= MIN_BUYERS].index)
    if len(pool) <= n:
        return sorted(pool)
    return sorted(rng.choice(pool, size=n, replace=False))


def main() -> None:
    fact = pd.read_parquet(pf.DATA / "ca_monthly.parquet")
    articles = pd.read_csv(pf.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")

    emit("# Prospect model — block 2")
    emit()
    emit("Label: the client bought an article they had NEVER bought in the snapshot window.")
    emit("Features come from the snapshot window only, so train and test see the same shapes.")
    emit()

    emit("## Building windows")
    # Half-year windows, not calendar years. The ledger ends 30 June 2026, so
    # calendar years give 12 / 12 / 6: the test label window would be HALF the
    # length of the training one, and a 6-month adoption rate is structurally
    # lower than a 12-month one. That is a different question, not less data.
    #
    #   2024-H1 -> 2024-H2   train
    #   2024-H2 -> 2025-H1   train
    #   2025-H1 -> 2025-H2   EMBARGO, dropped
    #   2025-H2 -> 2026-H1   test, untouched
    #
    # The embargo is the same rule snapshots.py enforces: without it, the last
    # training LABEL window (2025-H1) would be the test FEATURE window, so the
    # evaluation would not be honestly out-of-time.
    TRAIN_PAIRS = [("2024-H1", "2024-H2"), ("2024-H2", "2025-H1")]
    EMBARGO_PAIR = ("2025-H1", "2025-H2")
    TEST_PAIR = ("2025-H2", "2026-H1")

    ctx_test = pf.build_context(fact, articles, TEST_PAIR[0])
    win_test = pf.window(fact, TEST_PAIR[1])

    emit("| Role | Features | Labels | Clients |")
    emit("| --- | --- | --- | --- |")
    train_ctx = {fw: pf.build_context(fact, articles, fw) for fw, _ in TRAIN_PAIRS}
    for feat_w, lab_w in TRAIN_PAIRS:
        emit(f"| train | {feat_w} | {lab_w} | "
             f"{len(train_ctx[feat_w].clients):,} |")
    emit(f"| embargo (dropped) | {EMBARGO_PAIR[0]} | {EMBARGO_PAIR[1]} | - |")
    emit(f"| test | {TEST_PAIR[0]} | {TEST_PAIR[1]} | {len(ctx_test.clients):,} |")
    emit()

    rng = np.random.default_rng(pm.SEED)
    parts = []
    for feat_w, lab_w in TRAIN_PAIRS:
        ctx = train_ctx[feat_w]
        ids = eligible_articles(ctx, rng, TRAIN_ARTICLES)
        parts.append(pm.build_training_set(ctx, pf.window(fact, lab_w), ids))
        emit(f"- {feat_w} -> {lab_w}: {len(parts[-1]):,} rows, "
             f"{int(parts[-1]['label'].sum()):,} positives")
    train = pd.concat(parts, ignore_index=True)
    emit(f"## {len(TRAIN_PAIRS)} training windows, {pm.NEG_RATIO} negatives per positive")
    emit(f"- rows: {len(train):,}   positives: {int(train['label'].sum()):,} "
         f"({100 * train['label'].mean():.2f}%)")
    emit()

    feats = pf.FEATURE_COLUMNS
    cat_idx = [feats.index(c) for c in pf.CATEGORICAL_COLUMNS]
    cut = int(0.85 * len(train))
    tr, val = train.iloc[:cut], train.iloc[cut:]

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        eval_metric="AUC", random_seed=pm.SEED,
        early_stopping_rounds=50, verbose=100)
    model.fit(Pool(tr[feats], tr["label"], cat_features=cat_idx),
              eval_set=Pool(val[feats], val["label"], cat_features=cat_idx))

    pm.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(pm.MODEL_PATH))
    emit(f"- model saved to {pm.MODEL_PATH}")
    emit()

    emit(f"## Held-out evaluation — features {TEST_PAIR[0]}, labels {TEST_PAIR[1]}")
    eval_ids = eligible_articles(ctx_test, np.random.default_rng(pm.SEED + 1), EVAL_ARTICLES)

    def model_rank(ctx, article_id, client_ids):
        return pm.model_ranking(ctx, article_id, client_ids, model)

    results = {}
    for name, fn in [("model", model_rank),
                     ("heuristic", pm.heuristic_ranking),
                     ("client size", pm.size_ranking),
                     ("client breadth", pm.popularity_ranking),
                     ("random", pm.random_ranking)]:
        t0 = time.time()
        results[name] = pm.evaluate_ranker(ctx_test, win_test, eval_ids, fn)
        print(f"  {name:<16} {time.time() - t0:6.1f}s")

    emit()
    emit("| ranker | precision@5 | recall@5 | articles |")
    emit("| --- | --- | --- | --- |")
    for name, r in results.items():
        emit(f"| {name} | {r['precision_at_k']:.4f} | {r['recall_at_k']:.4f} "
             f"| {r['n_articles']:,} |")
    emit()

    test = pm.build_training_set(ctx_test, win_test, eval_ids[:500])
    auc = float(roc_auc_score(test["label"], model.predict_proba(test[feats])[:, 1])) \
        if test["label"].nunique() > 1 else float("nan")
    emit(f"AUC on a sampled held-out set: {auc:.4f}")
    emit()

    imp = pd.Series(model.get_feature_importance(), index=feats).sort_values(ascending=False)
    emit("## Feature importance (top 12)")
    emit()
    emit("| feature | importance |")
    emit("| --- | --- |")
    for k, v in imp.head(12).items():
        emit(f"| {k} | {v:.2f} |")
    emit()

    won = results["model"]["precision_at_k"] > results["heuristic"]["precision_at_k"]
    emit(f"## Verdict: {'model beats the heuristic' if won else 'HEURISTIC NOT BEATEN'}")
    if not won:
        emit()
        emit("Ship criterion not met. Keep the heuristic and record the finding.")

    reports = pf.DATA_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "prospect_model.md").write_text("\n".join(LINES), encoding="utf-8")
    (reports / "prospect_metrics.json").write_text(json.dumps({
        "rankers": results,
        "auc": auc,
        "beats_heuristic": bool(won),
        "train_rows": int(len(train)),
        "train_positive_rate": float(train["label"].mean()),
        "feature_importance": {k: float(v) for k, v in imp.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {reports / 'prospect_model.md'} and prospect_metrics.json")


if __name__ == "__main__":
    main()
