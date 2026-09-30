# v4 live scoring into the vendor queue

**Date:** 2026-08-12
**Status:** approved, ready for implementation plan

## Problem

The v4 embargoed rolling model (test AUC 0.7743, converged at iteration 1345,
`reports/v4-embargoed/`) writes `data/pair_scores_rolling.csv` — 2,051,182
pair-level scores. Nothing reads that file. The Streamlit app still displays a
queue built from the v1 model's `data/pair_scores_2026.csv`.

Worse, the rolling scores cover snapshots 2026-01 … 2026-04: the **test
window**. Those are backtest scores with known outcomes. Feeding them to
vendors would present predictions about the past as if they were advice.

## Goal

Vendors see a "clients to contact" queue driven by the v4 model, scoring a
snapshot that is actually in the future, with the existing Streamlit display
unchanged.

## Non-goals

Deliberately excluded to keep this shippable:

- No UI or page changes.
- No rebasing of the queue's revenue inputs onto the NAV ledger (see
  "Accepted inconsistency").
- No probability calibration. Scores stay rankings; the UI keeps showing
  confidence bands, not probabilities.
- No backtest/"proof" page. The evidence lives in `reports/`.

## The live snapshot

Data ends 2026-06-30, so the latest snapshot with a complete 12-month feature
window is **T = 2026-07-01**, predicting purchases in 2026-07 … 09.

T is **derived from the data, never hardcoded**, so the pipeline stays correct
when the next NAV export lands: T = first day of the month following the last
transaction date.

`build_snapshot_at` is already pure (no I/O, no globals, no clock), so it is
reused unchanged.

### The all-zero label hazard

`build_snapshot_at` computes labels from `tx[(date >= T) & (date < T+3mo)]`.
For a future T that slice is empty, so **every row gets label = 0**.

Those zeros are not observations, they are the absence of observations. If they
ever reached a metric, they would read as "the model was wrong about
everything". `score_live` therefore **drops the label column**, and a test
enforces its absence.

## Architecture

```
transactions.parquet ─┐
                      ├─> build_snapshot_at(tx, T_live)   [reused, pure]
model_rolling.cbm ────┘              │
                                     v
                        data/pair_scores_live.csv
                        client_id, article_id, vendor_code, p
                                     │
                                     v
              contact_queue.py  (logic UNCHANGED, p -> p_model)
                                     │
                                     v
                   clients_to_contact.csv -> Streamlit (identical)
```

### New: `src/score_live.py`

One job: produce today's actionable pair scores.

| Function | Contract |
| --- | --- |
| `live_snapshot_date(tx) -> Timestamp` | First day of the month after the last transaction date. Pure. |
| `score_live(tx, model, T) -> DataFrame` | Pure. Builds the snapshot, predicts, returns `client_id, article_id, vendor_code, p`. No label column. |
| `main() -> None` | The I/O shell: load parquet and `.cbm`, write CSV, print T, row count and model path. |

The model loaded is **`data/model_rolling.cbm`** — the file `train_rolling.py`
writes, currently the v4 embargoed model. `data/model_rolling_v4_embargoed.cbm`
is an immutable archive copy and is NOT read by the pipeline; it exists so the
v4 result stays reproducible if someone retrains with different settings.

Keeping the two pure functions separate from `main` is what makes this testable
without touching a 60 MB parquet or a 93 MB model.

### Changed: `src/contact_queue.py`

Only the score-source block (currently lines 60–71). Preference order:

1. `data/pair_scores_live.csv` — v4, live. Rename `p` -> `p_model`.
2. `data/pair_scores_2026.csv` — v1. Existing behaviour, unchanged.
3. Manual score fallback. Existing behaviour, unchanged.

Downstream code keeps seeing a column named `p_model`, so `valeur_attendue`,
the Élevée/Moyenne/Faible bands and `proba` need no edits.

The existing `[model] p_model joined on X%` line is kept and extended to name
the source that won. A sudden fall in join rate is the early warning that the
snapshot and the queue disagree about the candidate universe.

## Error handling

| Condition | Behaviour |
| --- | --- |
| `transactions.parquet` or `model_rolling.cbm` missing | `score_live.main` exits with a clear message; no partial CSV is written. |
| Snapshot features ≠ model features | Hard fail naming the differing columns. Silently scoring on the wrong feature set is the failure mode that produces confident nonsense. |
| `pair_scores_live.csv` missing | `contact_queue` falls back to v1, then to the manual score. Degrades to today's behaviour; never crashes. |

## Accepted inconsistency (disclosed, not hidden)

`valeur_attendue = p_model x ca_2025`.

After this change `p_model` comes from the v4 model on the **NAV ledger basis**,
while `ca_2025` comes from `ca_monthly.parquet` on the **v1 Jan–May extract
basis**. `reports/reconciliation.md` measures those two bases disagreeing by up
to 9.89% per month, with 11 of 15 months outside the 1% tolerance.

This was an explicit choice: swap the score now, rebase revenue later. It will
be stated in the `contact_queue.py` docstring and in the generated report, so
nobody reads `valeur_attendue` as a reconciled figure.

## Known risk: the two candidate universes differ

`contact_queue` merges scores onto **lapsed** pairs — defined as bought in 2024
and 2025, nothing in 2026. The live snapshot's candidates are pairs with a real
purchase inside its feature window, **2025-07 … 2026-06**.

A lapsed pair whose last purchase was, say, 2025-03 falls outside that window
and will therefore have no v4 score. The join rate on the live path may be
materially lower than on the v1 path, where `pair_scores_2026.csv` was built
over a different, wider universe.

`contact_queue` already fills misses with the column median, so nothing breaks
— but a low join rate means many rows ranked by a constant, which quietly
degrades the ordering.

This is why the join-rate line is a first-class output rather than a debug
print. **Acceptance check:** if the live join rate is far below the v1 path's,
stop and reconsider before shipping to vendors — the fix would be to widen the
lookback or to score the lapsed universe directly, and that is a design change,
not a tuning knob.

## Test plan (TDD, RED first)

`contact_queue.py` currently has **no tests**. This adds the first.

`tests/test_score_live.py`:

1. `live_snapshot_date` returns 2026-07-01 for data ending 2026-06-30.
2. `live_snapshot_date` rolls correctly for other end dates, including a date
   that is already the first of a month.
3. `score_live` output has **no `label` column** — guards the all-zero hazard.
4. `score_live` returns one row per candidate pair, with `p` in [0, 1].
5. Feature mismatch between snapshot and model raises, naming the columns.

`tests/test_contact_queue.py`:

6. Live scores take precedence over `pair_scores_2026.csv`.
7. With no live file, the v1 path still works.
8. With neither file, the manual-score fallback still works.

Tests use small synthetic frames and a tiny model, following the fixture style
in `tests/test_snapshots.py`. No test loads the real parquet.

## Definition of done

- `py src/score_live.py` writes `data/pair_scores_live.csv` for T = 2026-07-01.
- `py src/contact_queue.py` reports it used the live source, with a join rate
  in the same range as the v1 path.
- Streamlit renders the queue unchanged, now driven by v4.
- Full suite green (202 passing before this change, plus the new tests).
