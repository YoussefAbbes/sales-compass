# Model lifecycle, retraining and deployment

How long the current models stay valid, what to run when they expire, and how to stand the
API up on a server. Written 2026-08-18 against the ledger ending 2026-06-30.

---

## 1. The short answer

There are **two expiry dates**, and they are properties of the prediction windows, not of
model age.

| What | Window it predicts | Valid until | Remaining as of 2026-08-18 |
| --- | --- | --- | --- |
| Repurchase scores (`pair_scores_live.csv`) | 2026-07 → 2026-09 | **2026-09-30** | ~6 weeks |
| Adoption ranking (`prospect_model.cbm` served on 2026-H1) | 2026-H2 | **2026-12-31** | ~4.5 months |

After those dates the output is not "slightly stale" — it is answering a question about a
period that has already finished. The repurchase model was asked "who buys in July–September
2026?"; on 1 October that has an observed answer and the file is a historical record.

**The constraint that matters:** you cannot refresh either one without a new NAV export.
Both serving windows are *derived from the last transaction date*, so with data stopping at
2026-06-30 the live snapshot is pinned at `T = 2026-07-01` no matter when you re-run the
pipeline. `src/build_transactions.py` already anticipates this — "expect a third file for
later periods".

So the operational trigger is not a calendar reminder. It is: **a new NAV export arrives.**

---

## 2. Two clocks: scores expire, weights drift

These are different jobs with different costs, and conflating them is the usual mistake.

| | Rescore | Retrain |
| --- | --- | --- |
| What changes | New features at a new `T`, same weights | The CatBoost weights themselves |
| Cost | Minutes | Tens of minutes to ~1 h per model |
| Trigger | Every new export | Every 2–3 exports, or when metrics degrade |
| Needs code edits | No | **Yes — window constants are hardcoded** |

**Rescoring** is what you almost always need. The features (recency, cadence, basket
composition) move constantly; the learned relationship between "bought 4 times, last 40 days
ago" and "buys again" is far more stable. A model whose weights are six months old but whose
features are current is usually fine.

**Retraining** matters when the *relationship* shifts — a pricing change, a new product line,
a vendor reorganisation, or a measured drop in held-out AUC.

### The hardcoded-window gotcha

Serving auto-advances. Training does not.

| Derived from data (no edit needed) | Hardcoded (must be edited) |
| --- | --- |
| `score_live.live_snapshot_date()` | `snapshots.py`: `TRAIN_T`, `EMBARGO_TV`, `VAL_T`, `EMBARGO_VT`, `TEST_T` |
| `prospect_features.latest_complete_half()` | `train_prospect_model.py`: `TRAIN_PAIRS`, `EMBARGO_PAIR`, `TEST_PAIR` |

Current values in [src/snapshots.py](../src/snapshots.py):

```python
TRAIN_T    = _months("2025-01-01", "2025-06-01")   # 6 snapshots
EMBARGO_TV = _months("2025-07-01", "2025-08-01")   # 2, discarded
VAL_T      = _months("2025-09-01", "2025-10-01")   # 2, time-forward
EMBARGO_VT = _months("2025-11-01", "2025-12-01")   # 2, discarded
TEST_T     = _months("2026-01-01", "2026-04-01")   # 4, untouched
```

Shift all five forward by the number of new months when you retrain. Keep the shape: the
embargoes exist because a 12-month lookback and a 3-month horizon make adjacent snapshots
overlap, and without the gaps the validation set sees months the training set already
learned. **Do not shrink them to gain training data** — that is the leakage this design was
rebuilt to remove.

Two hard limits on the earliest and latest usable `T`: the earliest needs 12 months of
history behind it, and the latest needs 3 months of future ahead of it. With data
2024-01 → 2026-06 that gives `T` from 2025-01 to 2026-04, which is exactly the 16 snapshots
above.

---

## 3. Runbook: a new NAV export has arrived

Drop the new workbook(s) into `exported-data/` and run the following **in order**. Every step
reads what the previous one wrote.

```powershell
# 0. environment (once)
py -m pip install -r requirements.txt

# 1. ledger + masters — rebuild the basis
py src/build_transactions.py     # NAV sales workbooks -> data/transactions.parquet
py src/build_masters.py          # NAV masters -> articles_master.csv, clients_master.csv,
                                 #                articles.csv, clients.csv
py src/ledger.py                 # transactions.parquet -> data/ca_monthly.parquet

# 2. repurchase model
#    EDIT src/snapshots.py window constants first if retraining (see §2)
py src/train_rolling.py          # -> data/model_rolling.cbm
                                 #    data/pair_scores_rolling.csv   (test-window scores)
                                 #    reports/rolling_metrics.json, reports/model_rolling.md
py src/score_live.py             # -> data/pair_scores_live.csv      (the FUTURE window)
                                 #    reports/live_window.json       (the served window)

# 3. delivery queue
py src/baseline_lists.py         # -> lapsed_reorders, client_reactivation, cross_sell_pairs
py src/contact_queue.py          # -> data/clients_to_contact.csv

# 4. adoption model
#    EDIT src/train_prospect_model.py window pairs first if retraining
py src/train_prospect_model.py   # -> data/prospect_model.cbm
                                 #    reports/prospect_metrics.json, reports/prospect_model.md

# 5. keep the API's vendored engine in sync with src/
py model_results_api/tools/sync_engine.py

# 6. verify
py -m pytest tests/                    # 262
py -m pytest model_results_api/tests   # 24
```

**Rescore only** (new export, weights kept): steps 1, `score_live.py`, 3, 5, 6. Skip both
trainers. This is the common path.

### Tuning knobs

All read from the environment, so you never edit code to change them:

| Variable | Default | Used in the shipped model |
| --- | --- | --- |
| `ROLLING_ITERATIONS` | 1500 | 6000 (converged at 1345) |
| `ROLLING_EARLY_STOPPING` | 100 | 200 |
| `PROSPECT_TRAIN_ARTICLES` | 4000 | 4000 |
| `PROSPECT_EVAL_ARTICLES` | 2000 | 2000 |

Tuning on the rolling split is **legitimate**, unlike in the deleted v1: the validation
window is time-forward with an embargo, so an improvement there is evidence rather than
memorisation.

### What to check before shipping a retrained model

`train_rolling.py` and `train_prospect_model.py` both write a report and a JSON. Read them,
do not assume:

- `converged: true` and `best_iteration` well below `iterations_cap`. If `best_iteration`
  equals the cap, the model never converged — raise `ROLLING_ITERATIONS`.
- `auc.test` in the region of the current 0.7743, and `auc_per_snapshot` **flat**. A spread
  across test months means the model is riding a period effect, not learning structure.
- Adoption: `beats_heuristic: true`. If it does not beat precision@5 of the heuristic, that
  is a result — record it and keep the heuristic. The API already falls back to it.

---

## 4. Deploying the API

The API is read-only: it loads artifacts and serves JSON. No database, no writes, no queue.

### Environment variables

| Variable | Required | Notes |
| --- | --- | --- |
| `MODEL_RESULTS_SOURCE_ROOT` | yes in cloud | Directory containing `data/` and `reports/`. Relative paths resolve against the repo root. Set to `fixtures` for anonymised mode. |
| `MODEL_RESULTS_API_KEY` | **yes in cloud** | Shared secret, sent as `X-API-Key`. Without it the process refuses to start over real data — see below. |
| `MODEL_RESULTS_CORS_ORIGINS` | yes | Comma-separated. Defaults to localhost dev origins — must be set to the frontend's real origin. |

> **The API fails closed** (since 2026-08-25, commit `0ea9d0b`). Serving real `data/`
> with neither a key nor an explicit opt-in **refuses to start** — the process raises at
> import and prints the three ways forward, rather than accepting a connection it should
> have rejected. It previously verified the key only when one was configured, so an unset
> variable did not fail a request, it removed the question: "open to anyone who can reach
> it" was the state reached by doing nothing, over real client names and revenue.

| Source root | `MODEL_RESULTS_API_KEY` | Result |
| --- | --- | --- |
| `fixtures` | unset | starts, no key required (anonymised by construction) |
| real `data/` | set | starts, key enforced on all `/api/v1` routes |
| real `data/` | unset | **refuses to start** |
| real `data/` | unset, `MODEL_RESULTS_ALLOW_ANONYMOUS=1` | starts unauthenticated |

`MODEL_RESULTS_ALLOW_ANONYMOUS` exists so that anonymous access to real data is
something a person typed, never something a deploy reached by leaving a variable
unset. Bind to `127.0.0.1` when using it.

Still missing, and not fixed by the above: **vendor-level authorization**. Any key
holder reads all 54 vendors. Treat frontend auth as a second layer, not the only one.

### Install and start

```bash
pip install -r model_results_api/requirements.txt
python -m uvicorn model_results_api.app.main:app --host 0.0.0.0 --port $PORT
```

No Dockerfile is needed — Railway, Render and Fly all build Python from
`requirements.txt` and take a start command. Use **one worker**: each process holds its own
copy of the data (see sizing).

### Artifacts the server needs

`data/` is gitignored, so these must be shipped separately — object storage, a mounted
volume, or a build-time copy. Total **~46 MB**:

| File | Size | Role |
| --- | --- | --- |
| `data/pair_scores_live.csv` | 28 MB | repurchase scores (the FUTURE window) |
| `data/ca_monthly.parquet` | 11 MB | every aggregate and chart |
| `data/articles.csv` | 3.1 MB | lookups |
| `data/prospect_model.cbm` | 2.4 MB | live CatBoost inference |
| `data/clients_to_contact.csv` | 930 KB | the delivery queue |
| `data/clients.csv` | 210 KB | lookups |
| `data/vendors.csv` | 1.1 KB | lookups |
| `reports/rolling_metrics.json` | 2.3 KB | model card in `/meta` |
| `reports/prospect_metrics.json` | 1.8 KB | model card in `/meta` |
| `reports/live_window.json` | 158 B | the served window, on `/health` and `/meta` |

**`data/pair_scores_rolling.csv` (121 MB) is no longer shipped.** It is the backtest —
labelled rows over a window that has already elapsed — and the API served it until
2026-08-24. `src/validation.py` still reads it locally; nothing in the deployed service
does. Dropping it is most of the fall from ~139 MB to ~46 MB.

> `reports/live_window.json` is small enough to overlook and must not be. Without it the
> API cannot say which window it is answering for, and `/health` reports the window as
> unknown. Ship it in the same copy as the scores it describes — the two are only true
> of each other when they come from the same `score_live.py` run.

**`data/model_rolling.cbm` (94 MB) is not needed at runtime** and should not be shipped. The
repurchase model is served from the precomputed score table, not by inference; the 94 MB file
is only read as a fallback for the version hash, which `rolling_metrics.json` already
provides.

Only the adoption model does live inference. That asymmetry matters for refresh: one is a
batch artifact, the other a loaded model, behind one uniform API.

### Sizing

Measured steady-state resident memory, real data:

```
python startup                        14 MB
+ pandas/catboost import             141 MB
+ ca_monthly.parquet (1.75M rows)    354 MB
+ branded_fact (merged copy)         464 MB
+ pair_scores_rolling.csv            516 MB
+ prospect_context (features)        550 MB
```

> **These figures predate the swap to `pair_scores_live.csv` and have not been
> re-measured end to end.** The score frame itself is 41 MB smaller in memory
> (145 MB → 104 MB, 613,641 → 518,278 rows), so the true steady state is lower.
> Take a fresh reading before sizing the instance rather than trusting either number.

- **≥ 1 GB instance**, single worker. Caches are per-process, so 2 workers ≈ 1.1 GB.
- **Always-on, not scale-to-zero.** A cold start reads an 11 MB parquet and a 28 MB CSV;
  first request costs ~5.8 s, every subsequent one ~40 ms.
- Warm latency after the caching work: `/dashboard/overview` ~41 ms, `/brands` ~8 ms,
  vendor dashboard ~620 ms, live adoption inference ~4.6 s.

### Health check

Point the platform's probe at `/api/v1/health`. It returns `degraded` rather than failing when
artifacts are missing or the served window has closed, so check the body, not just the status
code — HTTP stays 200 either way, which is what keeps a closed window from restart-looping
the instance:

```json
{"status":"ok","data_ready":true,"repurchase_ready":true,
 "prospect_model_ready":true,"mode":"real",
 "repurchase_window":{"snapshot_date":"2026-07-01","window_end":"2026-09-30",
                      "expired":false,"days_remaining":37}}
```

`mode` tells you whether it is serving real data or fixtures — worth asserting in a smoke
test, because a misconfigured `MODEL_RESULTS_SOURCE_ROOT` silently serves anonymised data
with identical contracts.

`repurchase_window` is the one field that goes bad on its own, with no deploy and no error.
`expired: true` flips `status` to `degraded` and means every repurchase number describes a
period that has already finished; the fix is a new NAV export, not a restart. `null` means
`reports/live_window.json` was not shipped — the window is then unknown, which is reported
as unknown rather than guessed from the ledger. Alert on `days_remaining`, not on a date in
someone's calendar.

---

## 5. Known gaps

Recorded so they are not rediscovered under pressure.

1. **`vendors.csv` cannot be regenerated from the NAV export.** Vendor *codes* are in the
   ledger; vendor *names* are not. The only producer is `src/build_dataset.py`, which reads
   the old v1 workbook `CA client vendeur article Janv-mai 24-25-26.xlsx` — and expects it at
   the repo root, while it actually sits in `exported-data/`. So the file runs only after
   copying that workbook up one level, and it would also overwrite the NAV-projected
   `articles.csv` / `clients.csv` from `build_masters.py`. **Treat the current 44-vendor
   `vendors.csv` as an artifact to preserve**, or source vendor names from NAV and retire
   `build_dataset.py`.

2. ~~**The API serves test-window scores, not live scores.**~~ **Fixed 2026-08-24.** The
   API now reads `pair_scores_live.csv`, the same file `contact_queue.py` builds the queue
   from, so the score-table override no longer replaces a live probability with a backtest
   one. `score_live.py` additionally writes `reports/live_window.json`, and `/health` and
   `/meta` report the served window and flag it once it closes — a closed window makes
   `/health` `degraded`. Note this only makes the API honest about the window; it does not
   refresh it. The window still ends 2026-09-30 and only a new NAV export moves it.

3. **2023 is missing.** 30 continuous months allow rolling snapshots, which is a better
   route to out-of-time validation than a held-out year — but there is still no independent
   fresh-year window, so calibration remains unfitted and both scores stay explicitly
   uncalibrated.

4. **The last ledger month may be partial.** `ca_monthly.parquet` ends 2026-06 and nothing
   confirms June was fully closed when the export ran. The API surfaces this as a notice
   rather than silently trusting it; keep that caveat until the business confirms.

5. **Blocked clients are excluded by decision.** 222 clients are filtered out
   of every queue on the NAV blocked flag — see
   [reports/blocked-clients-delta.md](../reports/blocked-clients-delta.md). This is settled,
   not pending. Re-running `contact_queue.py` reapplies it automatically.
