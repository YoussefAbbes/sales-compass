# Phase 0 — Discovery Meeting Checklist

Purpose: leave this meeting able to write the SQL extractor and fix the prediction
horizon. Everything else can be resolved by email; these cannot.

Rule of thumb: **ask for artifacts, not explanations.** A schema export answers forty
questions that you would otherwise have to think to ask.

---

## Send ahead of the meeting

Give them time to prepare, or the answers will be "I'll have to check."

- [ ] Request a DDL export (`CREATE TABLE` statements) for the core tables.
- [ ] Request row counts and the date range of order history.
- [ ] Ask who the technical owner of the database is, and whether they can attend.

---

## Tier 1 — Blocking. Do not leave without these.

### 1. Database access

| Question | Answer |
|---|---|
| Engine and version (SQL Server / PostgreSQL / Oracle / MySQL / other) | |
| Can I get read-only access? To production or a replica? | |
| Who grants it, and by when? | |
| Any VPN, IP allowlist, or on-site-only restriction? | |

Determines the driver, the SQL dialect, and whether the whole plan is even runnable.

### 2. Schema

| Question | Answer |
|---|---|
| Which tables hold: clients, articles, orders, order lines, favorites? | |
| Primary and foreign keys for each | |
| Is there an ER diagram or data dictionary? | |

Ask for the DDL export in writing. Do not try to transcribe a schema verbally.

### 3. Order status vocabulary — the highest-risk item

| Question | Answer |
|---|---|
| What are all possible values of the order status column? | |
| Which values mean a **completed** purchase? | |
| Which mean cancelled, draft, quotation, sample, return, exchange? | |
| Are returns separate rows, negative quantities, or a flag? | |

If this is wrong, every label in the training set is wrong and no model result means
anything. Get the exact string or integer values, written down.

### 4. Volume and history

| Question | Answer |
|---|---|
| Number of active clients | |
| Number of active articles | |
| Number of completed order lines | |
| How far back does usable history go? | |
| Was there a system migration that split or renumbered IDs? | |

The migration question matters: it is the usual reason the same physical product exists
under several article IDs.

### 5. Business objective

| Question | Answer |
|---|---|
| What does success look like — revenue, margin, conversion rate, retention, vendor time saved? | |
| If only one metric could improve, which one? | |
| Is there a current baseline number for it? | |

Ranking weights (`commercial_priority` in project.md §8) cannot be set without this.

### 6. People

| Question | Answer |
|---|---|
| Who actually knows the schema, day to day? | |
| Who owns the vendor-facing application? | |
| Who decides business rules (pricing, eligibility, territories)? | |

---

## Tier 2 — Shapes the model. Get these if time allows.

### 7. Reorder cycles

- What are the main article categories?
- Typical reorder cycle for each — days, weeks, months?
- Are these consumables, spare parts, equipment, or a mix?

Sets the prediction horizon `H`. Fast-moving and slow-moving categories may need
different horizons.

### 8. Vendor ownership

- Is the vendor-client assignment a table, or derived from region/segment logic?
- Can a client have several vendors? Can a client have none?
- Does ownership change over time, and is history kept?

If ownership history is not kept, backtesting "which vendor saw this recommendation"
is impossible — say so early.

### 9. Vendor activity data

- Are client contacts recorded? Where?
- Are quotations recorded, with status and outcome?
- Is there any existing record of a vendor rejecting or postponing an opportunity?

If none of this exists, the feedback loop (project.md §10) starts empty and the
`prior_refusal_90d` / `days_since_last_contact` features are unavailable in v1.

### 10. Favorites

- What creates a favorite? Client action, vendor action, or automatic?
- Is removal recorded, or is the row deleted?
- Is there a timestamp?

Deleted rows mean favorite history cannot be reconstructed — affects §5.2 entirely.

### 11. Commercial rules

- Do clients have specific catalogues or contracts limiting what they can buy?
- Client-specific pricing, or one price list?
- Any compatibility rules between articles and clients?
- Where is credit hold / payment block recorded?

### 12. Stock — the trap

- Is stock a single live value, or is there a history table?
- If live only: can historical availability be reconstructed at all?

**Why this matters:** if stock has no history, you cannot test the stock filter against
past dates. A backtest will happily recommend articles that were out of stock at the
time, and offline metrics will be optimistic. This is worth flagging in the meeting
rather than discovering later.

---

## Tier 3 — Can wait. Ask by email.

- [ ] Should recommendations appear inside the existing vendor application, or is a
      separate interface acceptable?
- [ ] What is the application's technology stack?
- [ ] How fresh must recommendations be — nightly, hourly, real time?
- [ ] Which vendors would form a pilot group?
- [ ] Are there communication opt-outs or contact-frequency policies to respect?

---

## Artifacts to leave with

Worth more than any verbal answer:

- [ ] DDL export of the core tables
- [ ] Row counts and history date range
- [ ] The list of order status values with their meanings
- [ ] An anonymised sample export (1,000 rows per table) if permitted
- [ ] A named contact for schema questions

---

## What each answer unblocks

| Answer | Unblocks |
|---|---|
| Engine + access | `sources/sql_extractor.py`, driver choice, writeback strategy |
| Schema + DDL | Mapping their tables to the canonical contract |
| Status vocabulary | Correct labels — without this, nothing downstream is valid |
| Category reorder cycles | Prediction horizon `H` |
| Business objective | `commercial_priority` weights, confidence thresholds |
| Vendor ownership rules | Permission filters, per-vendor queue construction |
| Stock history | Whether the stock filter can be backtested honestly |

---

## Note on scope

If the meeting produces a wish list larger than the plan, resist expanding scope in the
room. Record everything, then map it against plan.md phases afterwards and come back
with what fits the timeline. "Let me check that against the delivery plan and confirm"
is a complete answer.
