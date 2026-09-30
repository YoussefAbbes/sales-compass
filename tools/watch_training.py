# -*- coding: utf-8 -*-
"""Live view of a running CatBoost job: where it is, and when it may stop.

A dev utility, not part of the modelled pipeline — it only reads the files
CatBoost writes as it trains, so it never touches the run.

    py tools/watch_training.py                # one snapshot
    py tools/watch_training.py --watch        # refresh every 30s until it ends

Early stopping cannot be predicted ahead of time: CatBoost stops once
`patience` consecutive iterations fail to beat the best validation AUC so far.
So the earliest possible stop is always `best + patience`, and that target
moves forward every time the model improves.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFO = ROOT / "catboost_info"


def read_curve(path: Path) -> list[tuple[int, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return rows


def status(cap: int, patience: int) -> str:
    curve = read_curve(INFO / "test_error.tsv")
    if not curve:
        return "no iterations recorded yet"

    now, _ = curve[-1]
    best_i, best_auc = max(curve, key=lambda r: r[1])
    used = now - best_i
    out = [
        f"iteration      {now} / {cap}   ({now / cap:.1%})",
        f"best val AUC   {best_auc:.6f} at iteration {best_i}",
        f"patience       {used} / {patience}",
        f"earliest stop  {best_i + patience} (moves forward on every improvement)",
    ]

    # Recent slope: is it still learning, or grinding?
    window = [r for r in curve if r[0] >= now - 500]
    if len(window) > 1:
        gain = window[-1][1] - window[0][1]
        out.append(f"last {now - window[0][0]} iters  {gain:+.6f} val AUC")

    tl = INFO / "time_left.tsv"
    if tl.exists():
        last = tl.read_text(encoding="utf-8").splitlines()[-1].split("\t")
        if len(last) >= 3:
            try:
                out.append(f"elapsed {int(last[1]) / 60000:.1f} min | "
                           f"{int(last[2]) / 60000:.1f} min left if it runs to {cap}")
            except ValueError:
                pass

    if used >= patience:
        out.append(">>> patience exhausted: early stopping should fire now")
    elif now >= cap - 1:
        out.append(">>> reached the cap WITHOUT early stopping firing")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()

    while True:
        print(f"--- {time.strftime('%H:%M:%S')} ---")
        print(status(a.cap, a.patience), flush=True)
        if not a.watch:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
