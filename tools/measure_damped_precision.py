# -*- coding: utf-8 -*-
"""Add `precision_top30_model_damped` to reports/rolling_metrics.json.

`train_rolling.py` now emits this metric, but producing it that way costs a full
retrain (~1 h) and would change the model the number describes. This recomputes it
from artifacts that already exist -- the backtest scores plus the snapshot features
they were scored from -- so the figure in the model card has a producer anyone can
re-run in a couple of minutes.

The definition is imported, never restated: `precision_top_n` and `damped_score`
come from train_rolling, so this tool cannot drift from what a retrain would write.
It re-derives the UNDAMPED figure too and refuses to write unless that reproduces
the stored one, which is the guard that proves the join and the top-30 rule match.

    py tools/measure_damped_precision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapshots import TEST_T                                    # noqa: E402
from train_rolling import damped_score, precision_top_n         # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TOLERANCE = 5e-4


def load() -> pd.DataFrame:
    months = [f"{d:%Y-%m}" for d in TEST_T]
    ids = {"client_id": str, "article_id": str, "vendor_code": str}
    scores = pd.read_csv(DATA / "pair_scores_rolling.csv", encoding="utf-8-sig", dtype=ids)
    scores["snap"] = scores["snapshot"].astype(str).str[:7]
    scores = scores[scores["snap"].isin(months)]

    frames = []
    for month in months:
        one = pd.read_parquet(
            DATA / "snapshots" / f"snapshot-{month}.parquet",
            columns=["client_id", "article_id", "days_since_last", "median_interval"])
        one["snap"] = month
        frames.append(one)
    cadence = pd.concat(frames, ignore_index=True)
    for col in ("client_id", "article_id"):
        cadence[col] = cadence[col].astype(str)

    return scores.merge(cadence, on=["client_id", "article_id", "snap"], how="left")


def main() -> None:
    path = REPORTS / "rolling_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = load()

    undamped = precision_top_n(test, "p")
    stored = metrics["precision_top30_model"]
    if abs(undamped - stored) > TOLERANCE:
        raise SystemExit(
            f"refusing to write: recomputed undamped precision {undamped:.4f} does not "
            f"reproduce the stored {stored:.4f}. The join or the split no longer matches "
            f"the run that produced this file.")

    damped = precision_top_n(test.assign(p_damped=damped_score(test)), "p_damped")
    metrics["precision_top30_model_damped"] = damped
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    demoted = test[damped_score(test) < test["p"]]
    print(f"rows            : {len(test):,}   cadence present {test.median_interval.notna().mean():.1%}")
    print(f"undamped        : {undamped:.4f}  (reproduces stored {stored:.4f})")
    print(f"damped          : {damped:.4f}")
    print(f"demoted pairs   : {len(demoted):,} ({len(demoted)/len(test):.1%}), "
          f"buy rate {demoted.label.mean():.1%} vs {test[damped_score(test) >= test['p']].label.mean():.1%}")
    print(f"[written] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
