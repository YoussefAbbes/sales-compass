# Modèle de prospection par article — design

**Date:** 2026-08-05
**Status:** Approved, pending implementation plan
**Supersedes the ranking logic of:** [2026-08-04-article-prospects-design.md](2026-08-04-article-prospects-design.md) (block 2 only)

## Problem

Block 2 of the prospection page — the clients who have *never* bought the selected article —
is ranked by a hand-written heuristic: count how many of the article's "signature" articles the
client already buys, filtered by two thresholds (`MIN_SHARE = 0.15`, `MIN_LIFT = 1.2`) that were
chosen as defensible starting values and never fitted to the data. The module comments them
"to validate with stakeholders". They never were.

Block 1, by contrast, is scored by the CatBoost model at AUC 0.762 against a 0.616 logistic
baseline, and the call queue was backtested at +49% versus random.

**Block 2 has no accuracy number of any kind.** There is currently no way to answer "does the
prospect list work?"

### Why this matters more than it appears

Measured on `data/ca_monthly.parquet` (896,312 rows, Jan–May of 2024, 2025, 2026):

| Transition | Pairs in Y | Pairs in Y+1 | New pairs | Share of Y+1 |
| --- | --- | --- | --- | --- |
| 2024 → 2025 | 231,434 | 251,342 | 204,985 | **81.6%** |
| 2025 → 2026 | 251,342 | 246,631 | 200,440 | **81.3%** |

Pair counts alone could mislead — many small new pairs could still be little money — so revenue
was measured separately:

| Window | Total CA | CA on new pairs | CA on repeat pairs | Share new |
| --- | --- | --- | --- | --- |
| 2025 | 46,432,146 | 33,107,352 | 13,324,794 | **71.3%** |
| 2026 | 45,160,481 | 31,462,682 | 13,697,799 | **69.7%** |

A newly-adopted pair is individually worth less than a repeat one (median 78 TND against 115, a
0.67× ratio), but there are so many more of them that they dominate.

**The consequence: the validated model governs ~30% of revenue, and the unvalidated heuristic
governs ~70%.**

## Goal

Replace the block-2 ranking with a supervised model whose accuracy is measured on the untouched
2025→2026 window, and which must beat the current heuristic to ship.

Not in scope: changing block 1, changing the page layout, or changing what block 2 *displays*.

## Feasibility gate (measured before design)

The risk was that adoption is too rare to learn. It is not.

| Transition | Candidate pairs | Adoptions | Base rate |
| --- | --- | --- | --- |
| 2024 → 2025 | 20,694,758 | 144,453 | **0.698%** (1 in 143) |
| 2025 → 2026 | 22,852,887 | 141,377 | **0.619%** (1 in 161) |

Candidates = (clients active in both windows) × (articles active in both windows), minus pairs the
client already held in Y.

| Fact | 2024→2025 | 2025→2026 |
| --- | --- | --- |
| Clients adopting ≥1 new article | 99.3% | 99.3% |
| New articles per client (median) | 77 | 69 |
| Articles gaining ≥1 new buyer | 98.4% | 97.9% |
| Articles gaining ≥5 new buyers | 53.7% | 48.2% |

0.6–0.7% is low but ordinary for this class of problem, and 144K positives is a large training
set. The two transitions agree closely on every measure, which is the property that makes a model
trained on 2024→2025 credible when tested on 2025→2026.

## Decisions settled during brainstorming

1. **Supervised gradient boosting, not collaborative filtering.** `project.md` §2 requires that
   every recommendation be explained "using verifiable business facts". A latent-factor dot product
   cannot produce that sentence; basket-affinity features can. Matrix factorisation also sits
   closer to the §3 non-goal on premature sophistication. A CF/CatBoost hybrid is the natural v2
   *after* this version establishes a measured baseline.
2. **Negatives are sampled within each article, never globally.** The page ranks clients *within*
   one article. Drawing negatives across the whole grid would train the model to answer "is this
   article plausible for anyone at all", inflate AUC, and optimise a different question than the
   one asked at inference. Starting ratio: 20 negatives per positive.
3. **Headline metric is precision@5 per article**, because the page shows five clients. AUC is
   reported alongside for comparability with the 0.762 model, but is not the headline.
4. **The heuristic is the incumbent and must be beaten.** It is scored on the identical split.
   Two further floors are reported: random, and rank-by-client-size — the latter because returning
   the same large clients for every part is the specific failure the signature was designed to
   avoid.
5. **No novelty-appetite feature.** The obvious strong predictor is how many new articles a client
   adopted in the *previous* transition. It is unbuildable without breaking the feature symmetry
   `train_model.py` enforces: every feature uses only snapshot year Y, because 2023 does not exist
   for the training split. Including it would give the test window a feature the train window
   cannot have.
6. **Live scoring, not a precomputed score file.** The candidate grid is 20M+ pairs, so a
   `prospect_scores.csv` in the style of `pair_scores_2026.csv` is not viable. Scoring one
   article's candidates is a few thousand rows, comparable to the 0.42 s the heuristic already
   costs, and keeps vendor scoping exact.
7. **Block 2 still displays no percentage.** The model improves the *order* of the top five, which
   is what the vendor acts on. Whether to surface a probability is deferred until calibration is
   measured — consistent with dropping the isotonic calibration and removing the
   "Confiance: Élevée" label.

## The learning problem

**Candidate universe**, for snapshot window Y:

- clients with positive CA in Y;
- articles with **≥ 3 buyers** in Y — the same `MIN_BUYERS` floor block 2 already refuses to go
  below, covering ~70% of the catalogue;
- minus every pair the client already holds in Y.

**No filter may reference window Y+1.** The feasibility table above restricted to clients and
articles present in *both* windows, which is legitimate for estimating a base rate but must not be
carried into training: "still active next year" is not knowable at inference time, and filtering on
it would drop churned clients from the negatives and leak the label window into the candidate set.
Training therefore keeps every client active in Y. The realised training base rate will sit
somewhat below the 0.698% measured above, because clients who churn entirely contribute negatives
only.

**Label:** `1` if the client bought that article in window Y+1, else `0`.

**Split:** train on features 2024 → label 2025; test on features 2025 → label 2026. Unchanged from
`train_model.py`, and the test window stays untouched until the final evaluation.

**Sampling:** for each article in the training sample, its adopters are the positives and its
non-adopters are drawn from that same article's candidate pool.

## Features

The reference tables carry no client attributes beyond a name, and no article attributes beyond
brand and a free-text designation. Every client feature is therefore behavioural, derived from
`ca_monthly.parquet` within window Y.

| Group | Features |
| --- | --- |
| **Client** | `client_ca`, `client_n_articles`, `client_n_months`, `client_n_vendors`, `client_n_families`, `mean_ca_per_article` |
| **Article** | `article_n_buyers`, `article_ca`, `mean_ca_per_buyer`, `article_n_vendors`, `enseigne` (cat.), `family` (cat.) |
| **Affinity** | `n_signature_held`, `share_signature_held`, `lift_weighted_overlap`, `brand_n_held`, `brand_ca_share`, `family_n_held`, `family_ca_share`, `cosine_basket_sim`, `knn_buyers` |

**No vendor features.** An earlier draft listed `vendor_sells_article` and `vendor_client_ca`.
They were dropped once the scoring order was settled: the model scores a client-article couple
**vendor-blind**, and ownership is applied afterwards as a display filter. Feeding the vendor in
would make the model learn each vendor's book rather than the article's appeal, and would force
scoring to be repeated per (vendor, article) instead of once per article. Ownership is a question
about who may be *shown* a prospect, not about who wants the part.

`n_signature_held` is today's block-2 score. It stops being *the* answer and becomes one input
whose weight is learned.

The four affinity features that are not self-evident from their names:

| Feature | Definition |
| --- | --- |
| `lift_weighted_overlap` | Σ of the lift of each signature article the client holds, rather than a plain count — a client holding three high-lift articles scores above one holding three marginal ones |
| `share_signature_held` | `n_signature_held / len(signature)`, so the count is comparable across articles with signatures of different sizes |
| `cosine_basket_sim` | cosine between the client's binary article vector and the article's buyer-profile vector (the mean binary vector of its buyers) |
| `knn_buyers` | of the **K = 20** clients most similar to this one by cosine basket similarity, how many bought the article. K = 20 is a starting value to be tuned on the training window only |

### The `family` feature

Derived from the first whitespace-delimited token of `designation`, upper-cased. Verified as a
real product family rather than noise:

| Measure | Value |
| --- | --- |
| Distinct first tokens | 616 |
| Top 50 cover | 75.0% of named articles |
| Top 200 cover | 97.0% |
| First *two* tokens | 3,260 values, 62.7% singletons — too sparse, rejected |

Coverage against the fact table:

| Measure | Value |
| --- | --- |
| Distinct articles in `ca_monthly` | 40,182 |
| Present in `articles.csv` (named) | 24,659 — **61.4%** by count |
| Revenue carried by named articles | **91.8%** |

Unnamed articles take `family = "INCONNU"`. They are 38.6% of the catalogue but only 8.2% of
revenue.

`family` and `enseigne` (262 values) enter as native CatBoost categoricals, not one-hot: at 616
and 262 levels one-hot would add ~900 sparse columns, and CatBoost's ordered target statistics
handle high-cardinality categoricals without that blowup.

## Architecture

```text
src/prospect_features.py       builds the (article, candidate-client) feature frame
src/prospect_model.py          sampling, metrics, scoring, rank_prospects() entry point
src/train_prospect_model.py    trains, evaluates, writes report + model artifact
src/article_prospects.py       NOT MODIFIED — wrapped, not edited
src/pages/2_Prospection.py     one line changed: calls rank_prospects()
```

`src/article_prospects.py` is left untouched. An earlier draft put the model path inside its
`prospects()` function, but `prospect_features` needs `buyers_of` and `vendor_clients` from it,
so the model path would have closed an import cycle. Putting the new entry point in
`prospect_model.py` keeps the dependency strictly one-directional —
`train → model → features → article_prospects` — and has the side benefit of keeping the feature
additive, so the existing 103 tests continue to cover the heuristic unchanged.

**Training and inference call the same feature builder.** One function serves both paths, so
train/serve skew — features computed differently in production than in training — is structurally
impossible rather than merely tested against.

```text
TRAINING                                INFERENCE (page)
for each article in sample:             for the selected article:
    build_features(article, cands) ─────┴──→ build_features(article, cands)
              ↓                                        ↓
    label + fit CatBoost                      model.predict_proba
```

Artifacts written to `reports/`: `prospect_model.md` (human-readable, matching the existing model
report) and `prospect_metrics.json` (machine-readable, so the validation page and API can consume
it exactly as they consume `model_metrics.json` today). The model itself is saved to
`data/prospect_model.cbm` and is not committed — `data/` is gitignored.

`prospects()` uses the model when the artifact is present and falls back to the heuristic when it
is not, so the page never breaks on a fresh clone that has not run training.

## Evaluation protocol

Reported on the untouched 2025→2026 window, per article, then averaged over **articles that gained
at least one new buyer in 2026** (48.2% of those active in both windows). Articles with zero
adoptions are excluded from the average: precision@5 is necessarily 0 for them regardless of
ranking quality, so including them would measure the catalogue's adoption rate rather than the
model. Their count is reported alongside so the coverage is visible.

| Metric | Purpose |
| --- | --- |
| **precision@5** | headline — matches what the page shows |
| recall@5 | how much of the available opportunity the five capture |
| AUC | comparability with the existing 0.762 model |

Against four references: the current heuristic (incumbent), rank-by-client-CA, rank-by-client-
breadth, and random. Breadth is included alongside CA because the two differ — a client can spend
heavily on few references or lightly across many — and breadth is the more plausible naive
predictor of trying something new, so it is the tougher floor.

**Ship criterion:** the model beats the heuristic on precision@5. If it does not, the heuristic
stays and the finding is recorded — a measured "no improvement" is a valid outcome of this work.

## Edge cases

| Situation | Behaviour |
| --- | --- |
| Article with < 3 buyers | No signature, no candidates — unchanged message ("trop peu d'acheteurs") |
| Model artifact absent | Falls back to the heuristic, page renders normally |
| Article unseen at training time | Features are computed live from window Y, so it scores normally; only the categoricals may be unfamiliar, which CatBoost handles |
| Vendor with a single client | Both blocks degrade cleanly, as today |
| Fewer than 5 candidates after vendor filtering | Shows what exists, no padding |

## Testing

Following the existing convention — unit tests on synthetic frames, no dependency on the 51 MB
extract:

- feature builder produces the expected columns and dtypes on a synthetic frame;
- within-article sampling draws negatives only from the target article's candidate pool;
- the fallback path is exercised when no model artifact exists;
- `prospects()` returns the same typed empty frames as today in every edge case;
- **a leakage test**: a feature frame built for window Y contains no column derived from Y+1.

Reference to hold: **103 tests green** before this work starts.

## Out of scope, deliberately

- **Changing block 1.** Its model, ranking and display stay as they are.
- **Displaying a probability in block 2.** Deferred until calibration is measured.
- **Collaborative-filtering embeddings.** The natural v2, once this version has a baseline.
- **The API and React dashboard.** The module is written so an endpoint can call it unchanged
  later, exactly as `validation.py` is shared today, but no route is added now.
- **Quantity or price suggestions.** The extract has neither.
