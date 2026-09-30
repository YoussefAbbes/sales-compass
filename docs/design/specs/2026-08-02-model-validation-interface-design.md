# Interface de validation du modèle — design

**Date:** 2026-08-02
**Status:** Approved, pending implementation plan

## Problem

The project can currently *state* that the CatBoost model reaches 0.762 AUC on a chronological
test window, but there is no way to **look at** that claim. `reports/model_catboost.md` is a static
markdown dump: it cannot be interrogated, it shows no train-vs-test comparison, it never measures
calibration, and it offers no way to inspect a single prediction against what actually happened.

For the report and the defense, the gap that matters is the last one. "The model has 0.762 AUC" is
an assertion. "Here is client 0049, we gave article 0241235752 a 0.83 probability, and they bought
it in March 2026" is evidence.

## Goal

A Streamlit page that lets a technical reader verify — interactively — that the model's predictions
were true, at three levels of zoom: the protocol, the aggregate, and the individual pair.

**Audience:** the author + encadrant/jury. Technical vocabulary is allowed, but every metric carries
a plain-language caption stating what it proves.

**Scope decision (settled during brainstorming):** this page grades the **retrospective** test window
2025→2026, where ground truth already exists in the data. Prospective tracking of live
recommendations (vendor feedback → outcome) is explicitly **out of scope**; `data/feedback.csv`
does not exist yet and would start empty. The page is designed so that a prospective section can be
added later without restructuring.

## Key constraints discovered during design

### 1. Train-side scores do not exist

`src/train_model.py` scores only the test snapshot and writes `data/pair_scores_2026.csv`. There is
no train-side prediction file and no persisted metric other than the rendered markdown table. A
train-vs-test comparison is therefore impossible without extending the training script first.

### 2. The production queue cannot be literally replayed

`src/backtest.py` already documents this: a literal replay of the 2026 queue as of 2025 is impossible
with this extract, because the lapsed rule requires two prior years and 2023 is not in the data.

**Consequence for §3 of this page.** What can be honestly graded is the *model-driven article
ranking*, not the full production queue:

| Production rule (`contact_queue.py`) | Replicable on the test window? |
| --- | --- |
| Rank articles by `p_model × ca` (expected value) | Yes — `p_model` and `ca_sum` both exist in the test snapshot |
| Restrict to pairs bought in both prior years | Yes — pairs bought in 2024 **and** 2025 |
| Lapsed filter (bought in prior years, absent this year) | **No** — needs a 2023 basis |
| Client status boost (reactivation status) | **No** — needs Jan–Apr of two prior years |
| Top-3 articles per client | Yes |

The page must state this in the UI, not only in this document. Presenting the approximation as the
real queue would misrepresent the result.

### 3. Calibration has never been measured

`progress.md` records that the queue shows confidence as labels (Élevée / Moyenne / Faible) rather
than raw percentages, precisely because calibration was not validated. This page is what would
settle that question in either direction.

## Architecture

Metric computation lives in a module with no Streamlit import. The page is a display layer.

```text
src/train_model.py          (extended) also writes pair_scores_train.csv + model_metrics.json
src/validation.py           (new)      all metric computation — pure pandas/sklearn, no Streamlit
src/pages/1_Validation.py   (new)      display only, calls validation.py behind @st.cache_data
tests/test_validation.py    (new)      pytest against validation.py
tests/test_app.py           (moved)    the existing AppTest scenarios, recovered from scratchpad
```

**Why not compute inside the page.** Doing so would re-implement the ranking logic in UI code, which
would then drift silently from `contact_queue.py`, and it would be untestable without booting a
browser — a risk this project already took once, when a theming bug shipped because no browser tool
was available to verify it.

`src/app.py` stays the entrypoint (`streamlit run src/app.py`) and its content is unchanged. Creating
`src/pages/` is what turns the app into a multipage app: Streamlit renders the entrypoint as the first
page and auto-discovers `pages/*.py` as the rest, adding a nav to the sidebar. The numeric filename
prefix controls nav order only.

### Data contract

`train_model.py` gains two outputs:

- `data/pair_scores_train.csv` — same columns as `pair_scores_2026.csv`, for the 2024→2025 snapshot.
- `reports/model_metrics.json` — train/validation/test AUC, base rates, pair counts, best iteration,
  feature importances, and the decile tables. JSON so the page never re-fits the model.

The model is seeded (`random_seed=42`), so re-running the training script must reproduce the
existing `pair_scores_2026.csv` and `model_catboost.cbm` identically. This is verified as an
implementation step, not assumed.

## Page structure

### §1 — Le protocole

Answers *"what is the difference between training and test?"*

- Timeline: train = features 2024 → label 2025; test = features 2025 → label 2026.
- Table: pair count, positive rate, and AUC for train / internal validation / test.
- The iteration experiment written up as evidence: at 4000 iterations the random validation split
  improved (0.798 → 0.810) while the chronological test degraded (0.762 → 0.748). Kept at 800.
  Conclusion stated plainly: a random validation split is not a proxy for temporal generalization.

### §2 — Performance agrégée

Answers *"did it predict correctly, overall?"*

- AUC comparison: manual score 0.616 / logistic 0.738 / CatBoost 0.762, against the same test set.
- Decile lift (3.0% → 53.0% against an 18.4% base rate).
- **Calibration curve** — predicted probability vs observed frequency, in bins. New measurement.
- Confusion matrix at an adjustable threshold (see decision below).

### §3 — Les recommandations

Answers *"were the recommendations right?"*

- Precision@N for N ∈ {5, 10, 30} per vendor, compared against the manual score and the base rate.
- Money view: TND captured vs TND missed in the top-N.
- A visible caveat box reproducing the replication table from "Key constraints" above.

### §4 — Inspection individuelle

Answers *"show me a specific prediction and whether it came true."*

- Vendor selector → client selector → table of that client's pairs.
- Columns: article, designation, `p_model`, predicted, **reality ✓/✗**, CA at stake.
- Filters: hits only / misses only / **most confident errors** (high `p_model`, label 0).

The "most confident errors" filter is the diagnostically useful one — it is where systematic failure
modes become visible.

## Open decision — confusion matrix threshold

The threshold that converts a probability into a "yes" is a business choice, not a statistical one:

- `0.5` — the default, but against an 18.4% base rate it classifies almost nothing as positive.
- `0.184` (base rate) — balanced, but arbitrary-looking.
- Top-N framing — define positive as "in the vendor's top 30", matching actual use.

**Decision:** implement as a slider defaulting to the base rate, with the top-30-equivalent threshold
marked on it. This makes the trade-off demonstrable live rather than fixed to one defensible-but-
brittle number.

## Testing

`tests/test_validation.py` covers `src/validation.py` with pytest, using a small synthetic frame —
no dependency on the real 51 MB extract:

- Calibration binning returns bins that sum to the input row count.
- Precision@N on a hand-built frame with a known answer.
- Decile lift is monotonic on a perfectly-ranked synthetic input.
- The queue-replay pool applies the both-years restriction correctly.
- Threshold sweep endpoints behave (threshold 0 → all positive; threshold 1 → none).

Page-level smoke tests use Streamlit's `AppTest`, following the pattern already used for `app.py`:
page loads, vendor switch works, drill-down renders for a client with known values.

**Note on the existing tests.** The 7 `AppTest` scenarios cited in `progress.md` as verification for
`app.py` currently live only in a session scratchpad and are not in the repository. They should be
moved into `tests/` as part of this work, so that the claim in `progress.md` is reproducible from a
clone.

## Out of scope

- Prospective tracking of live recommendations and vendor feedback outcomes.
- Retraining, hyperparameter tuning, or any model change. Tuning was deliberately stopped; this page
  measures the existing model and must not become a tuning loop against the single honest test window.
- Any change to the recommendation logic in `contact_queue.py`.

## Success criteria

- The page answers all four questions above without the reader opening a CSV.
- Every number shown is traceable to `validation.py`, which is covered by tests.
- The replication limits of §3 are visible in the UI, not only in this document.
- Re-running `train_model.py` reproduces the existing model artifacts bit-for-bit.
