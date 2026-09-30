# Prospection on the v4 model

**Date:** 2026-08-12
**Status:** approved, ready for implementation

## Problem

`src/pages/2_Prospection.py` still runs the v1 model. It reads
`pair_scores_2026.csv` for block 1 and `ca_monthly.parquet` for everything else.
Both predate the transactional refoundation.

Two consequences, one of them serious.

**The percentages are from the wrong model.** Block 1 shows v1 probabilities —
the model whose validation set we proved was contaminated (see
`reports/tuning-log.md`). The figure a vendor reads as "16 % de propension au
rachat" is not the number behind the validated 61.9 % precision@top-30.

**Block 2 can pitch a part to someone who already buys it.** `prospects()`
computes `candidates = owned - buyers`, and `buyers_of()` reads `fact`.
`ca_monthly.parquet` holds January-May only, so a client who bought the article
in September is not counted as a buyer. Measured: **400,353 of 1,017,175 buyer
pairs (39 %) are invisible to that exclusion.** The module docstring states the
rule this breaks — "a client already buying the part through a colleague is not
a prospect, and pitching it to them would expose that colleague's business."

## Goal

Prospection ranks on the v4 model, keeps every client visible, and shows a
percentage only where one is earned.

## Non-goals

- **Block 2's adoption model is not touched.** It answers a question v4
  structurally cannot: v4's candidates all require a prior purchase of the
  article, so it has no features for a pair that never traded. Block 2 only
  benefits from the full-ledger fix to its buyer exclusion.
- No calibration. Percentages stay rankings.
- No layout redesign beyond the new block.

## The dormant half

Of 1,017,175 buyer pairs in the ledger, the v4 live universe covers exactly
50 %. The other **508,337 are dormant**: they bought the article somewhere in
the 30-month ledger, but not inside the snapshot's 12-month feature window, so
v4 has no opinion on them.

Without a third block these clients vanish from the page — no v4 score removes
them from block 1, and they are excluded from block 2 by definition because they
*are* buyers. A vendor would silently lose sight of half the buyer base.

## Design: three blocks, distinguished by French tense

| Block | Population | Ranked by | Percentage? |
| --- | --- | --- | --- |
| 1 · Ils l'achètent | bought, and inside v4's window | `p x ca_sum` | **yes** — v4 |
| 2 · Ils l'achetaient | bought, dormant 12+ months | article CA | **no** |
| 3 · À prospecter | never bought | adoption model | **no** |

The present/imperfect distinction carries the meaning without extra words.
Today's block 1 is in fact labelled "Ils l'achetaient" — the imperfect — which
already describes the dormant group better than the active one.

Showing no percentage in block 2 is the same discipline block 3 already applies
("any number here would be invented") and the same reason `score_live` drops its
all-zero label column. **Show a number only where one is earned.** v4 changes
which model earns block 1's number; it does not change the rule.

## Components

### `src/article_prospects.py`

Add one function. `existing_customers` is unchanged — it already takes a scores
frame, so it accepts v4 scores once they are renamed at the boundary.

```
dormant_buyers(fact, scored_pairs, article_id, vendor_code, top=TOP_N)
    -> DataFrame[client_id, article_ca, last_month]
```

- population: `vendor_clients(fact, vendor) & buyers_of(fact, article)` minus
  any client already carrying a v4 score for that article;
- ranked by the client's own CA on that article, descending;
- `last_month` is that client's most recent month with positive CA on the
  article, so the vendor can say when it stopped;
- returns no probability column, ever. A test enforces its absence.

`BLOCK3_COLUMNS = ["client_id", "article_ca", "last_month"]`

### `src/pages/2_Prospection.py`

- `load_all()` reads the full ledger through `ledger.monthly_ca` instead of
  `ca_monthly.parquet`, and prefers `pair_scores_live.csv` — renaming `p` to
  `p_model` and `amount_sum` to `ca_sum` at the boundary — falling back to
  `pair_scores_2026.csv`, then to an empty frame.
- Renders the new block between the existing two, renumbered 1/2/3.
- Replaces the caption "Le pourcentage est une **estimation optimiste** (voir la
  page Validation)" with the honest wording already used on the queue page: the
  percentages are not calibrated and should be read as a ranking.

The rename happens at the page boundary rather than inside
`article_prospects.py` so that module stays independent of which model produced
the scores — the property that lets it be unit-tested without a browser.

## Error handling

| Condition | Behaviour |
| --- | --- |
| `pair_scores_live.csv` missing | fall back to `pair_scores_2026.csv`, then to an empty frame; blocks 2 and 3 still render |
| no dormant buyers for an article | block 2 is omitted rather than shown empty |
| article never bought by anyone | blocks 1 and 2 both empty; block 3 still answers |

## Test plan (TDD, RED first)

`tests/test_article_prospects.py`:

1. a client with a v4 score is NOT in dormant_buyers — it belongs to block 1;
2. a client who never bought the article is NOT in dormant_buyers — block 3;
3. a client outside the vendor's ownership is excluded;
4. the output carries **no** probability column;
5. rows are ordered by article CA, descending;
6. `last_month` is the client's last positive month on that article, not their
   last month overall;
7. `buyers_of` sees a September purchase — the regression guard for the 400,353
   pairs, pinned against a synthetic ledger with a month outside January-May.

Synthetic frames only, matching the existing style in that file.

## Definition of done

- Prospection block 1 percentages come from `pair_scores_live.csv`.
- A client who bought the article in September is never offered as a prospect.
- Dormant buyers appear in their own block with no percentage.
- Full suite green (228 passing before this change, plus the new tests).
