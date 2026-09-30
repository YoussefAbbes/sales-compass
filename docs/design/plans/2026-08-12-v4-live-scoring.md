# v4 Live Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the vendor "clients to contact" queue from the v4 embargoed model scoring a future snapshot, leaving the Streamlit UI untouched.

**Architecture:** A new pure module `src/score_live.py` builds the snapshot at T = the month after the last transaction, scores it with `data/model_rolling.cbm`, and writes `data/pair_scores_live.csv`. `src/contact_queue.py` gains a score-source preference (live -> v1 -> manual) extracted into one testable function. Nothing downstream changes: the queue still sees a column named `p_model`.

**Tech Stack:** Python 3.13, pandas, CatBoost, pytest. Run tests with `py -m pytest`.

## Global Constraints

- Python is invoked as `py` on this machine, never `python`.
- All CSV I/O uses `encoding="utf-8-sig"`, matching every other reader in `src/`.
- `client_id`, `article_id`, `vendor_code` are ALWAYS read as `str`. They have leading zeros; reading them as int silently corrupts joins.
- Tests use synthetic frames only. No test may read `data/transactions.parquet` or load a real `.cbm`.
- `src/snapshots.py` is NOT modified by this plan. `build_snapshot_at` is reused as-is.
- The label column must never appear in live output. This is a correctness rule, not a style preference.

---

### Task 1: Derive the live snapshot date

**Files:**
- Create: `src/score_live.py`
- Test: `tests/test_score_live.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `live_snapshot_date(tx: pd.DataFrame) -> pd.Timestamp`

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_live.py`:

```python
# -*- coding: utf-8 -*-
"""Live scoring contract. Synthetic frames only.

A live snapshot predicts a window that has not happened yet. The two things
that can go wrong silently are scoring the wrong date and shipping the
all-zero placeholder labels as if they were outcomes; both are pinned here.
"""
import numpy as np
import pandas as pd
import pytest

from score_live import live_snapshot_date


def frame(*dates):
    return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates]})


def test_live_date_is_the_month_after_the_last_transaction():
    """Data ends 2026-06-30, so the first snapshot with a complete feature
    window strictly before T is 2026-07-01."""
    assert live_snapshot_date(frame("2026-01-05", "2026-06-30")) \
        == pd.Timestamp("2026-07-01")


def test_a_last_date_mid_month_still_rolls_to_the_next_month():
    assert live_snapshot_date(frame("2026-07-15")) == pd.Timestamp("2026-08-01")


def test_a_last_date_on_the_first_of_a_month_rolls_forward_not_sideways():
    """2026-07-01 means July has data, so T must be August. Returning July
    would put the last day of data inside the feature window's own future."""
    assert live_snapshot_date(frame("2026-07-01")) == pd.Timestamp("2026-08-01")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_score_live.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'score_live'` (the file does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/score_live.py`:

```python
# -*- coding: utf-8 -*-
"""Score the live snapshot — the one window that has not happened yet.

train_rolling.py scores the TEST window (2026-01..04) to prove the model works.
That output is a backtest: every row carries the outcome. This module exists
because a vendor needs the opposite — the next 90 days, unlabelled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshots import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_snapshot_at,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def live_snapshot_date(tx: pd.DataFrame) -> pd.Timestamp:
    """First day of the month AFTER the last observed transaction.

    Derived, never hardcoded, so the pipeline stays correct when the next NAV
    export lands. A snapshot at T draws features strictly before T, so the last
    day of data must fall before T — hence "the month after", not "the month of".
    """
    last = pd.Timestamp(tx["date"].max())
    return (last + pd.offsets.MonthBegin(1)).normalize()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_score_live.py -q`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/score_live.py tests/test_score_live.py
git commit -m "feat(live): derive the live snapshot date from the data"
```

---

### Task 2: Score the live snapshot, without labels

**Files:**
- Modify: `src/score_live.py`
- Test: `tests/test_score_live.py`

**Interfaces:**
- Consumes: `live_snapshot_date` from Task 1.
- Produces: `score_live(tx: pd.DataFrame, model, T: pd.Timestamp) -> pd.DataFrame` returning exactly the columns `["client_id", "article_id", "vendor_code", "p"]`. `model` needs only `predict_proba(X) -> ndarray` of shape (n, 2).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score_live.py`:

```python
from score_live import score_live


class StubModel:
    """Stands in for CatBoostClassifier. score_live only needs predict_proba,
    so the test does not pay for training a real model."""

    def __init__(self, p=0.42):
        self.p = p
        self.seen_columns = None

    def predict_proba(self, X):
        self.seen_columns = list(X.columns)
        return np.column_stack([np.full(len(X), 1 - self.p),
                                np.full(len(X), self.p)])


def ledger(*rows):
    """Synthetic ledger lines, same shape as tests/test_snapshots.py::tx."""
    out = []
    for date, client, article in rows:
        out.append({"date": pd.Timestamp(date), "client_id": client,
                    "article_id": article, "vendor_code": "V1",
                    "enseigne": "FEBI", "amount": 100.0, "qty": 1.0,
                    "cost": 70.0, "is_return": False,
                    "doc_no": f"D{pd.Timestamp(date):%Y%m%d}{client}"})
    return pd.DataFrame(out)


LIVE_T = pd.Timestamp("2026-07-01")


def test_live_scores_carry_no_label_column():
    """build_snapshot_at derives labels from the horizon window. For a future T
    that window is empty, so every label would be 0 — the absence of an
    observation, not an observation of absence. Shipping it invites someone to
    compute an accuracy of zero against it."""
    out = score_live(ledger(("2026-03-10", "C1", "A1")), StubModel(), LIVE_T)
    assert "label" not in out.columns


def test_live_scores_have_exactly_the_expected_columns():
    out = score_live(ledger(("2026-03-10", "C1", "A1")), StubModel(), LIVE_T)
    assert list(out.columns) == ["client_id", "article_id", "vendor_code", "p"]


def test_one_row_per_candidate_pair_with_p_in_range():
    out = score_live(
        ledger(("2026-03-10", "C1", "A1"), ("2026-04-10", "C1", "A2")),
        StubModel(p=0.42), LIVE_T)
    assert len(out) == 2
    assert set(zip(out["client_id"], out["article_id"])) == {("C1", "A1"), ("C1", "A2")}
    assert out["p"].between(0, 1).all()


def test_the_model_is_given_the_canonical_feature_set():
    """Guards against scoring on a different column order or set than training
    used — the failure mode that produces confident nonsense."""
    from snapshots import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    model = StubModel()
    score_live(ledger(("2026-03-10", "C1", "A1")), model, LIVE_T)
    assert model.seen_columns == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_a_snapshot_missing_model_features_is_rejected_by_name():
    class Broken:
        def predict_proba(self, X):
            raise AssertionError("must not be reached")

    import score_live as mod
    original = mod.build_snapshot_at
    mod.build_snapshot_at = lambda tx, T: pd.DataFrame(
        {"client_id": ["C1"], "article_id": ["A1"], "vendor_code": ["V1"]})
    try:
        with pytest.raises(ValueError, match="n_orders"):
            score_live(ledger(("2026-03-10", "C1", "A1")), Broken(), LIVE_T)
    finally:
        mod.build_snapshot_at = original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_score_live.py -q`
Expected: FAIL — `ImportError: cannot import name 'score_live' from 'score_live'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/score_live.py`:

```python
LIVE_COLUMNS = ["client_id", "article_id", "vendor_code", "p"]


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

    out = snap[["client_id", "article_id", "vendor_code"]].copy()
    out["p"] = model.predict_proba(snap[feats])[:, 1]
    return out[LIVE_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_score_live.py -q`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/score_live.py tests/test_score_live.py
git commit -m "feat(live): score the live snapshot, dropping placeholder labels"
```

---

### Task 3: The I/O shell

**Files:**
- Modify: `src/score_live.py`

**Interfaces:**
- Consumes: `live_snapshot_date`, `score_live` from Tasks 1-2.
- Produces: `main() -> None`, and the file `data/pair_scores_live.csv`.

No unit test: `main` is pure I/O over a 60 MB parquet and a 93 MB model, which the Global Constraints forbid in tests. It is verified by running it.

- [ ] **Step 1: Write the implementation**

Append to `src/score_live.py`:

```python
MODEL_PATH = DATA / "model_rolling.cbm"
OUT_PATH = DATA / "pair_scores_live.csv"


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
    end = T + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    print(f"[live] last transaction {tx['date'].max():%Y-%m-%d} -> T={T:%Y-%m-%d}")
    print(f"[live] predicting purchases {T:%Y-%m} .. {end:%Y-%m}")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    out = score_live(tx, model, T)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[live] {len(out):,} pairs scored with {MODEL_PATH.name}")
    print(f"[written] {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `py src/score_live.py`
Expected: prints `T=2026-07-01`, a pair count in the hundreds of thousands, and writes `data/pair_scores_live.csv`. Takes several minutes — one snapshot build is roughly 7 minutes on this machine.

- [ ] **Step 3: Verify the output contract by hand**

Run: `head -1 data/pair_scores_live.csv`
Expected: exactly `client_id,article_id,vendor_code,p` — no `label`, no `snapshot`.

- [ ] **Step 4: Commit**

```bash
git add src/score_live.py
git commit -m "feat(live): add the scoring entry point"
```

---

### Task 4: Prefer live scores in the contact queue

**Files:**
- Modify: `src/contact_queue.py:59-72` (the `# ---- Model probabilities` block) and the module docstring
- Test: `tests/test_contact_queue.py`

**Interfaces:**
- Consumes: `data/pair_scores_live.csv` written by Task 3.
- Produces: `attach_model_scores(lapsed: pd.DataFrame, data_dir: Path) -> pd.DataFrame`, returning `lapsed` with a `p_model` column added.

This extracts the existing inline block into a function. That is what makes the preference order testable at all — `contact_queue.py` currently has no tests.

- [ ] **Step 1: Write the failing test**

Create `tests/test_contact_queue.py`:

```python
# -*- coding: utf-8 -*-
"""The score-source preference. Synthetic frames only.

The queue must prefer the v4 live scores, fall back to v1, and fall back again
to the manual score — each step a real degradation, so which one won is printed
rather than guessed.
"""
import pandas as pd

from contact_queue import attach_model_scores


def lapsed_frame():
    return pd.DataFrame({
        "client_id": ["C1", "C2"],
        "article_id": ["A1", "A2"],
        "score": [10.0, 5.0],
        "ca_2025": [1000.0, 500.0],
    })


def write_live(tmp_path, p1=0.9, p2=0.8):
    pd.DataFrame({"client_id": ["C1", "C2"], "article_id": ["A1", "A2"],
                  "vendor_code": ["V1", "V1"], "p": [p1, p2]}).to_csv(
        tmp_path / "pair_scores_live.csv", index=False, encoding="utf-8-sig")


def write_v1(tmp_path, p1=0.1, p2=0.2):
    pd.DataFrame({"client_id": ["C1", "C2"], "article_id": ["A1", "A2"],
                  "p_model": [p1, p2]}).to_csv(
        tmp_path / "pair_scores_2026.csv", index=False, encoding="utf-8-sig")


def test_live_scores_win_when_both_files_exist(tmp_path):
    write_live(tmp_path)
    write_v1(tmp_path)
    out = attach_model_scores(lapsed_frame(), tmp_path)
    assert out["p_model"].tolist() == [0.9, 0.8]


def test_v1_is_used_when_there_is_no_live_file(tmp_path):
    write_v1(tmp_path)
    out = attach_model_scores(lapsed_frame(), tmp_path)
    assert out["p_model"].tolist() == [0.1, 0.2]


def test_manual_score_is_the_last_resort(tmp_path):
    out = attach_model_scores(lapsed_frame(), tmp_path)
    assert out["p_model"].tolist() == [1.0, 0.5]


def test_pairs_missing_from_the_score_file_get_the_median_not_nan(tmp_path):
    """A live snapshot only contains pairs active in its feature window, so
    lapsed pairs can legitimately be absent. They must not become NaN and sink
    the ranking silently."""
    pd.DataFrame({"client_id": ["C1"], "article_id": ["A1"],
                  "vendor_code": ["V1"], "p": [0.9]}).to_csv(
        tmp_path / "pair_scores_live.csv", index=False, encoding="utf-8-sig")
    out = attach_model_scores(lapsed_frame(), tmp_path)
    assert out["p_model"].notna().all()
    assert out["p_model"].tolist() == [0.9, 0.9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_contact_queue.py -q`
Expected: FAIL — `ImportError: cannot import name 'attach_model_scores' from 'contact_queue'`.

- [ ] **Step 3: Write minimal implementation**

In `src/contact_queue.py`, add this function immediately above `def main():`

```python
def attach_model_scores(lapsed: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add p_model, preferring the v4 live scores.

    Order: pair_scores_live.csv (v4, future window) -> pair_scores_2026.csv
    (v1, backtest basis) -> the manual score. Each fallback is a genuine
    degradation, so the winner is printed rather than left to be inferred.
    """
    live = data_dir / "pair_scores_live.csv"
    v1 = data_dir / "pair_scores_2026.csv"

    if live.exists():
        scores = pd.read_csv(live, dtype={"client_id": str, "article_id": str},
                             encoding="utf-8-sig").rename(columns={"p": "p_model"})
        source = "pair_scores_live.csv (v4, live)"
    elif v1.exists():
        scores = pd.read_csv(v1, dtype={"client_id": str, "article_id": str},
                             encoding="utf-8-sig")
        source = "pair_scores_2026.csv (v1)"
    else:
        print("[model] no score file -> falling back to manual score")
        lapsed = lapsed.copy()
        lapsed["p_model"] = (lapsed["score"] / lapsed["score"].max()).clip(0, 1)
        return lapsed

    lapsed = lapsed.merge(scores[["client_id", "article_id", "p_model"]],
                          on=["client_id", "article_id"], how="left")
    joined = lapsed["p_model"].notna().mean()
    lapsed["p_model"] = lapsed["p_model"].fillna(lapsed["p_model"].median())
    print(f"[model] {source}: p_model joined on {joined:.1%} of lapsed pairs")
    return lapsed
```

Then replace the whole block from the line `    # ---- Model probabilities (CatBoost) -> expected repurchase value per pair ----` through the line `        lapsed["p_model"] = (lapsed["score"] / lapsed["score"].max()).clip(0, 1)` with:

```python
    # ---- Model probabilities -> expected repurchase value per pair ----
    lapsed = attach_model_scores(lapsed, DATA)
```

- [ ] **Step 4: Update the module docstring**

In `src/contact_queue.py`, replace the paragraph beginning `p_model comes from the CatBoost propensity model` with:

```
p_model comes from the v4 embargoed rolling model (src/train_rolling.py, test AUC
0.7743, converged at iteration 1345 - reports/v4-embargoed/), scoring the LIVE
snapshot written by src/score_live.py. It falls back to the v1 pair_scores_2026.csv,
then to the manual score. Probabilities are shown to vendors only as confidence
labels (Elevee/Moyenne/Faible) - calibration not yet validated (project.md section 9).

KNOWN INCONSISTENCY: valeur_attendue = p_model x ca_2025 multiplies a v4 probability
computed on the NAV ledger by a revenue figure computed on the older Jan-May extract
basis. reports/reconciliation.md measures those bases disagreeing by up to 9.89% per
month, 11 of 15 months outside the 1% tolerance. Deliberate: swap the score now,
rebase revenue later. Do not read valeur_attendue as a reconciled figure.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -m pytest tests/test_contact_queue.py -q`
Expected: PASS, 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/contact_queue.py tests/test_contact_queue.py
git commit -m "feat(queue): prefer v4 live scores, with disclosed basis mismatch"
```

---

### Task 5: End-to-end run and the join-rate gate

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Consumes: everything above.
- Produces: a go/no-go decision on shipping to vendors.

- [ ] **Step 1: Run the full suite**

Run: `py -m pytest -q`
Expected: 202 passing before this plan, plus 3 (Task 1) + 5 (Task 2) + 4 (Task 4) = **214 passed**.

- [ ] **Step 2: Regenerate the queue**

Run: `py src/contact_queue.py`
Expected: a line reading `[model] pair_scores_live.csv (v4, live): p_model joined on XX.X% of lapsed pairs`.

- [ ] **Step 3: Apply the acceptance gate**

The live snapshot only contains pairs with a purchase in 2025-07..2026-06, while
`lapsed` means "bought 2024 and 2025, nothing in 2026". These universes are not
the same, and a pair that went quiet in early 2025 is lapsed but unscoreable.

- If the join rate is comparable to the v1 path: proceed.
- If it is far below: **STOP. Do not ship.** Report the number. The fix is a
  design change — widen the lookback, or score the lapsed universe directly —
  not a tuning knob, and it needs its own spec.

- [ ] **Step 4: Confirm the app still renders**

Run: `py -m streamlit run src/app.py`
Expected: the queue page renders with the same columns as before, now driven by v4.

- [ ] **Step 5: Commit the regenerated queue if the gate passed**

```bash
git add data/clients_to_contact.csv
git commit -m "chore(queue): regenerate clients_to_contact from v4 live scores"
```
