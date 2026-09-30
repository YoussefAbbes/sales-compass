# Prospection par article — design

**Date:** 2026-08-04
**Status:** Approved, pending implementation plan

## Problem

The application answers one of the three questions in `project.md` §1: *which clients should this
vendor contact, and for which articles?* The reverse question — *given an article, which clients are
the best prospects?* — is the article-first workflow of `project.md` §4, and nothing in the project
addresses it.

It is not a cosmetic gap. A vendor with a promotion to push, a supplier target to hit, or overstock
to move starts from the part, not from the client. Today they have no tool for that.

## Goal

A third Streamlit page: search for an article, see the five clients most worth calling about it,
scoped to the vendor selected in the sidebar.

**Audience:** vendors, like page 1 — plain language, no metric shown without a caption stating what
it means.

## Key constraint discovered during design

**The CatBoost model cannot score a (client, article) pair that has no purchase history.**

Its pair-level features — `months_n`, `ca_sum`, `regularity`, `share_of_client` — are all computed
from the pair's own past. For a client who has never bought the article, they do not exist. This is
not a limitation of the training run; it is what the model *is*.

Measured consequences:

| Fact | Value |
| --- | --- |
| Scored pairs available | 251,342, covering 25,543 articles |
| Median clients scored per article | **4** |
| Articles with ≥5 scored clients | 11,766 / 25,543 (**46.1%**) |
| Full client × article grid | 2,060 × 40,321 = 83 M couples |
| Couples actually observed | 621,094 — **99.3% have no pair features at all** |

So a "top 5 clients" built only from model scores would be unfillable for the majority of articles,
and would exclude by construction the population the feature is commercially about: clients who do
**not** yet buy the article.

The page therefore has two blocks, ranked by two different signals, each stating its own provenance.

## Decisions settled during brainstorming

1. **Two blocks, not one list.** Mixing a model probability and a similarity rank into a single
   ranking would produce a number that means nothing. They stay separate and visibly labelled.
2. **Prospect ranking = basket similarity, computed live.** Measured: 0.42 s per article, and it
   covers articles with ≥3 buyers — **70.0% of the catalogue** (28,137 / 40,182), against 12.4% for
   the precomputed `cross_sell_pairs.csv`, which `baseline_lists.py` limits to the top 5,000
   articles. Live computation also keeps the signal specific to the article, which a generic
   client-health ranking would not be: that would return the same large clients for every part.
3. **Vendor scoping is kept** (`project.md` §12). Verified affordable: on 300 randomly drawn
   articles, a median of **43 vendors out of 51** can still fill a top 5 after filtering, and there
   is **no article** for which no vendor can. Compliance costs nothing here.
4. **Free-text article search.** 40,321 articles, of which 15,638 carry no designation, make a
   dropdown unusable. Search covers reference, brand and designation, so unnamed articles remain
   reachable by their reference.
5. **New page, not a section of `app.py`.** Matches the existing multipage structure and keeps the
   vendor queue untouched.

## Architecture

Additive only. No existing file is modified — not `app.py`, not the validation page, not the
pipeline, not the API.

```text
src/article_prospects.py          pure computation, no Streamlit import
src/pages/2_Prospection.py        display layer only
tests/test_article_prospects.py   unit tests on synthetic frames
tests/test_prospection_page.py    AppTest
```

The split mirrors `validation.py` / `1_Validation.py`: logic in the module, display in the page.
That is what makes the feature testable without a browser, and it is the reason the existing
validation metrics could be reused by the API later without reimplementation.

Inputs, all already produced by the pipeline:

| File | Used for |
| --- | --- |
| `data/ca_monthly.parquet` | baskets, ownership, client size |
| `data/pair_scores_2026.csv` | model probabilities (block 1) |
| `data/articles.csv` | search index |
| `data/clients.csv` | display names |

## Ownership rule, used identically by both blocks

"A vendor's clients" = clients with **positive CA under that `vendor_code` in
`ca_monthly.parquet`**, over any month.

Stated explicitly because `pair_scores_2026.csv` also carries a `vendor_code` — the *dominant*
vendor of that pair — and the two disagree whenever a client is served by several vendors, which
is the case for 78% of clients. Using the fact table for both blocks keeps the two lists on the
same population; taking the pair's dominant vendor in block 1 and the fact table in block 2 would
silently show two different vendor scopes on one page.

## Block 1 — "Ils l'achetaient" (clients with history on this article)

Population: this vendor's clients that appear in `pair_scores_2026.csv` for this article.

Ranking: **expected value = `p_model` × `ca_sum`**, where `ca_sum` is the pair's 2025 revenue as
written by `train_model.py` — the same rule as the call queue, so that "#1" means the same thing
everywhere in the application.

Display: probability and amount, carrying the existing "estimation optimiste" caveat.

## Block 2 — "À prospecter" (never bought this article)

1. Buyers of X = clients with positive CA on X.
2. **Signature of X** = articles over-represented among those buyers: share ≥ 0.15 among buyers,
   lift > 1.2 versus the catalogue-wide rate, top 30 by lift.
3. Candidates = the vendor's clients who are **not** buyers of X, ranked by how many signature
   articles they already buy. Ties broken by client CA.

Both steps 1 and 3 consider purchases of X **under any vendor**, not only the selected one. A
client who already buys the article through another vendor is not a prospect, and presenting them
as one would produce the exact false alarm the lapsed-reorder rule was designed to avoid
(`baseline_lists.py`: pairs are computed client-wide, then attributed).

Reason text, factual and verifiable against the data:

> Achète 23 des 30 articles que prennent habituellement les acheteurs de cette pièce, mais jamais
> celle-ci.

**No percentage is displayed in block 2.** The model does not score these clients; showing a number
would imply it does. This applies the same rule as `project.md` §9 — a label rather than a false
precision — and the same reasoning that led to dropping the isotonic calibration.

## Wording constraint

The block 1 probability is a propensity for the January–May 2026 window, which has already elapsed.
As on the call queue, it is a **propensity score, not a forecast**. Wording: *« propension au
rachat »*, never *« chance qu'il achète »*.

## Edge cases — each renders a message, never an empty table

| Situation | Frequency | Behaviour |
| --- | --- | --- |
| Article with < 3 buyers | ~30% of catalogue | Block 2: "trop peu d'acheteurs pour calculer une signature" |
| Article absent from `pair_scores_2026.csv` | common | Block 1: "aucun client de ce vendeur ne l'a acheté en 2025" |
| Search returns nothing | — | invites trying a reference |
| Vendor with a single client | exists (min = 1) | both blocks degrade cleanly |
| Signature computed but no candidate left | possible | Block 2: explicit "aucun prospect" message |

## Testing

Unit tests on synthetic frames, with no dependency on the 51 MB extract: search by reference,
designation and brand; signature computation including the lift threshold; exclusion of existing
buyers from block 2; vendor filtering; expected-value ordering in block 1; and each of the five edge
cases above.

AppTest on the page: load, search, selection, vendor switch, article with no buyers.

Reference to hold: **70 tests green** before this work starts.

## Out of scope, deliberately

- **The React dashboard and the API.** The module is written so that an endpoint can call it
  unchanged later, exactly as `validation.py` is shared today — but no route is added now.
- **Quantity or price suggestions.** The extract has neither.
- **Persisting the prospect lists.** They are computed on demand; nothing is written to `data/`.
