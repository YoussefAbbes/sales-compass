# Tuning log

Every attempt to tune this model, including the ones that failed. A failed
experiment is only wasted if it is not written down — twice now, the same
mistake has cost a multi-hour run.

Numbers here are reproducible from `reports/v2-baseline-1500iter/`,
`reports/v3-6000iter/` and the curves archived alongside them.

---

## Attempt 1 — v1, 4000 iterations (abandoned)

Recorded in the `src/snapshots.py` docstring. A 4000-iteration run improved a
**random** validation split while degrading the real out-of-time test. Tuning
and calibration were both abandoned as untrustworthy.

Diagnosis at the time: the validation split shared a year with training.
This motivated the whole rolling-snapshot refoundation (v2).

---

## Attempt 2 — v2, 6000 iterations (failed, 2026-08-12)

Ran the refounded design at a 6000-iteration cap with 200 rounds of patience,
against the frozen 1500-iteration baseline. Same data, same seed, same
features; only the cap changed.

| Metric | 1500 iter (baseline) | 6000 iter | Δ |
| --- | --- | --- | --- |
| Train AUC | 0.807 | 0.833 | +0.026 |
| Validation AUC | 0.781 | 0.788 | **+0.006** |
| **Test AUC** | **0.775** | **0.771** | **−0.004** |
| Train − test gap | +0.032 | +0.062 | doubled |
| Precision @ top-30/vendor | 62.2% | 61.7% | −0.5 pt |

Every one of the four test snapshots degraded, so this is not noise in a
single month:

| Snapshot | 1500 iter | 6000 iter |
| --- | --- | --- |
| 2026-01 | 0.7763 | 0.7733 |
| 2026-02 | 0.7736 | 0.7696 |
| 2026-03 | 0.7749 | 0.7705 |
| 2026-04 | 0.7759 | 0.7723 |

Validation AUC never plateaued — it rose monotonically for the whole 6000
iterations (0.734 → 0.776 at 500 → 0.781 at 1500 → 0.787 at 5000 → 0.7877 peak
at 5970) while test AUC moved the other way.

**Verdict: rejected.** The baseline was kept as the model of record.

### Diagnosis

The same failure as attempt 1, for a subtler reason. v2 made the validation
set time-forward, which fixed the *random split* problem — but it did not put
an embargo between train and validation.

With a 3-month label horizon, the last training snapshot (2025-08) labelled
purchases across Aug–Oct 2025. The validation snapshots (2025-09, 2025-10)
labelled purchases across Sep–Dec 2025. **The same physical purchase events
defined labels on both sides of the boundary.** Extra iterations let the model
memorise those events; validation paid for it, test did not.

The val→test embargo was correct from the start, so the headline test AUC of
0.775 was never contaminated. What was broken was only the *model-selection*
signal — precisely the thing the v2 refoundation existed to repair.

Note that the 1500-iteration baseline was itself cap-limited, not converged:
validation AUC was still rising when it stopped. Under a contaminated
validation set, the less-trained model generalised better. That inversion is
the signature of the leak, not a property of the data.

### A second, smaller bug found in the same run

The report claimed *"Early stopping fired… this is a converged result."* It had
not fired: the best iteration was 5970 under a 6000 cap with 200 rounds of
patience, so only 29 iterations followed the best. Training ended by exhausting
the cap.

The predicate was `best < ITERATIONS - 1`, which is true for every outcome
except a best on the very last iteration — it could not detect a cap-limited
run at all. Replaced by `early_stopping_fired()` in `src/train_rolling.py`,
covered by tests at the exact boundary.

### Fixes applied (2026-08-12)

- `EMBARGO_TV` (2025-07, 2025-08) inserted between train and validation.
  Training drops from 8 snapshots to 6, ~4.0M rows to ~3.0M.
- `tests/test_snapshots.py::test_train_to_validation_gap_is_at_least_the_horizon`
  now fails if any future edit reintroduces the leak.
- `early_stopping_fired()` replaces the broken convergence predicate; both the
  Markdown report and the JSON read from the one expression.
- Test suite: 202 passing.

---

## Attempt 3 — v4, corrected splits (SUCCESS, 2026-08-12)

6000-iteration cap, 200 rounds of patience, on the embargoed splits.
**Early stopping fired at iteration 1545, best iteration 1345** — the first
genuinely converged model in the project's history. Archived in
`reports/v4-embargoed/`.

| Metric | v2 baseline (1500, leaky val) | v3 (6000, leaky val) | **v4 (embargoed)** |
| --- | --- | --- | --- |
| Train AUC | 0.8072 | 0.8333 | 0.8110 |
| Validation AUC | 0.7813 | 0.7877 | 0.7709 |
| **Test AUC** | 0.7752 | 0.7714 | **0.7743** |
| Train − test gap | +0.0320 | +0.0620 | +0.0367 |
| **Validation − test gap** | +0.0061 | +0.0163 | **−0.0035** |
| Precision @ top-30/vendor | 62.2% | 61.7% | 61.9% |
| Top-decile lift | ×3.43 | ×3.37 | ×3.45 |
| Converged? | no (cap) | no (cap) | **yes** |

### The prediction, and how it held up

The log predicted early stopping would fire well before 6000, and that test AUC
would land slightly below 0.775. Both held: it stopped at 1345, and test came in
at 0.7743, nine ten-thousandths under the baseline.

(Mid-run, at iteration 1087, patience was still 0 and the call was made that
early stopping probably would NOT fire, on the grounds that CatBoost resets
patience on any improvement however small. That call was wrong — the curve went
genuinely flat rather than creeping, and patience filled 0→200 between
iterations 1345 and 1545.)

### Why this settles the diagnosis

**The validation−test gap flipped sign.** Validation was optimistic about test by
+0.0061 on the baseline and by +0.0163 after 6000 leaky iterations — the
optimism GREW with training, which is the memorisation signature. With the
embargo in place validation is now −0.0035, i.e. very slightly *pessimistic*.

That is the ideal property for a model-selection set: tuning against it cannot
flatter you. It is also not explainable by the smaller training set, which would
lower validation and test together rather than reverse their order.

**The leak was worth roughly 4,600 spurious iterations.** Contaminated
validation asked for 5,970 trees; clean validation asks for 1,345. The v2
baseline's 1500-iteration cap — picked arbitrarily, with no legitimate stopping
rule behind it — happened to sit just above the true optimum, which is why it
performed well despite the broken selection signal. That was luck, and it is now
replaced by a rule.

**The fix cost almost nothing.** Test AUC moved −0.0009 and precision@top-30
moved −0.3 points, both inside noise, despite training on 1M fewer rows.
Per-snapshot test AUC stayed tight (0.7727–0.7751). In exchange the project has,
for the first time, a validation set it can legitimately tune, calibrate and
early-stop against.

### Model of record

`data/model_rolling_v4_embargoed.cbm` (1,345 trees, 93 MB) supersedes
`data/model_rolling_1500.cbm` as the model of record.

### Open follow-up

CatBoost's `early_stopping_rounds` has no minimum-improvement threshold: it
resets on any new maximum, however microscopic. It worked here because the curve
went truly flat, but a curve that creeps by 1e-6 per iteration would run to the
cap while learning nothing. Adding a minimum-delta stopping rule would make the
behaviour robust rather than fortunate.
