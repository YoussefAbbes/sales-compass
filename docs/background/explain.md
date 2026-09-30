# How to explain this project

Two layers. **Part 1** is what you say. **Part 2** is what you answer when pushed.
Everything here is checked against the code and `reports/` — no claim that the repo can't back.

---

# PART 1 — THE CORE

## The 30-second pitch

> A distributor of auto spare parts has 54 vendors and 2,060 clients. Nobody knew who to call
> first — vendors worked from memory. I built a system that ranks, for each vendor, which
> clients to call today and which parts to talk about, with a reason for each one.
> It's validated on a year the model never saw: among the pairs it puts in its top 10%,
> 53% really did buy again, versus 18% for an average pair — and 60% precision on a vendor's
> top-30 list, against 42% for the rule the business would have used by hand.

If you only get one more sentence:

> The whole project was shaped by what the data actually was — monthly revenue totals, not
> orders. So I did not build "when will this client buy". I built "who is worth calling first",
> which is the question the data can honestly answer.

## The problem in one line

Three questions a vendor has every morning: *which articles will this client need next?*,
*which clients do I call today?*, *who are the best prospects for this article?* (`project.md` §1)

## The constraint that shaped everything

I asked for transactional data. What existed was **one Excel file**: monthly revenue per
vendor × client × article, January–May of 2024, 2025 and 2026.

| What that data has | What it does NOT have |
| --- | --- |
| Money per month, per pair | Dates → "reorders every 32 days" is impossible |
| 896,312 rows, 15 months | Quantities → "usually orders 8 units" is impossible |
| 54 vendors, 2,060 clients, 40,321 articles | Order status → a sale, a return and a quote are indistinguishable except by sign |
| Monthly revenue only, no dates | June–December → a 7-month blind spot every year |
| | Stock, favorites, prices, contact history |

**This is the single most important thing to say early.** Every "why didn't you do X" question
in Part 2 traces back to this table. Say it before the jury finds it.

## The pipeline in 5 steps

Each step is one script, and each one exists for a reason.

**1. Build a clean dataset** — `src/build_dataset.py`
Excel is pivoted (12 month-columns wide). I unpivot it into one long fact table:
`(vendor, client, article, month) → CA`, saved as Parquet.
*Why:* every later step is a groupby on that grain. Parquet loads ~896K rows instantly, Excel doesn't.
*Proof it's right:* per-month totals of my table match the source file exactly.

**2. Three rule-based signals** — `src/baseline_lists.py`
- **Lapsed reorders**: pairs bought in 2024 *and* 2025, nothing in 2026 → 28,119 pairs.
- **Client reactivation**: clients inactive in 2026 or down ≥30% → 141 + 348 clients.
- **Cross-sell**: articles appearing in the same (client, month) basket, ranked by *lift*.

*Why rules before ML:* they are explainable on day one, they gave the business something usable
immediately, and they became the **benchmark the model had to beat**. Without them, "AUC 0.762"
would have nothing to compare against.

**3. A CatBoost model** — `src/train_model.py`
Question: *a pair bought in year Y — will it be bought again in Y+1?*
Train on 2024→2025. Test on 2025→2026, **untouched**.
*Why CatBoost:* tabular data, mixed numeric + categorical (brand, vendor code), 231K rows.
Gradient boosting is the right tool at this size; deep learning is not.

**4. The call queue** — `src/contact_queue.py`
One row per client per vendor, combining the three signals.
Articles are ranked by **expected value = model probability × last year's revenue**.
*Why both terms:* probability alone puts a 1,896 TND antifreeze above a 15,590 TND oil.
"#1" is meant to be *where the most money is*, not *the surest bet* — and the interface says so.

**5. Delivery — three formats, on purpose**
- **38 Excel workbooks**, one per vendor (a vendor only sees their own clients) — works with zero IT.
- **Streamlit app** — the interactive prototype, plus a validation page.
- **React + FastAPI dashboard** — four views, the version to demo.

## The 6 numbers to know cold

| # | Number | What it means |
| --- | --- | --- |
| 1 | **896,312 rows** — 54 vendors, 2,060 clients, 40,321 articles, 15 months | The scale of the data |
| 2 | **AUC 0.762** vs **0.616** for the manual rule (0.5 = chance) | The model beats the business rule |
| 3 | **Top decile: 53% repurchase** vs a **18.4% base rate** (×2.9) | Ranking works where it matters |
| 4 | **Top-30 per vendor: 60.4% precision** vs 42.3% manual | The number a vendor actually feels |
| 5 | **Rule backtest: 52% of a top-30 list built on 2024 alone bought again in 2025** (×2.6) | Even without ML the signals were real |
| 6 | **1,204 clients queued across 38 vendors — 67 automated tests green** | It's a working system, not a notebook |

## The 3 limits to say yourself, before you're asked

1. **The Jun–December blind spot.** Labels only observe Jan–May. A client who rebuys in
   September is scored as a failure. Every rate I report is therefore a **lower bound** —
   the true performance is at least this good, never worse.
2. **2023 doesn't exist**, so I have exactly one honest test window. That's why I stopped
   tuning, and why calibration could not be fixed (Part 2).
3. **No causal proof.** I show that recommended clients buy more often. I have *not* shown
   that the vendor's call caused it — that needs a controlled pilot with a control group
   (`plan.md` Phase 6). Never claim revenue impact from these numbers.

Saying these three first is not a weakness. It's the difference between a student who ran a
script and one who knows what the script proves.

---

# PART 2 — THE HARD QUESTIONS

## About the data

**Q. Why is a monthly revenue extract enough to do anything at all?**
It isn't enough for the original goal (predicting *when*). It is enough to rank *who and what*,
because repurchase behavior is visible even at month granularity: a pair bought in 5 different
months has an ~80% chance of coming back next year, one bought in a single month has ~15%.
That gradient is the whole signal (`reports/backtest.md`, T1).

**Q. Why did you keep negative revenue in the table instead of cleaning it?**
Negative CA (3.9% of rows) is a return netted in by the source system. Deleting it would silently
change every total, and my conservation check against the Excel would have failed. So I kept it
and made each analysis decide: a "purchase" is defined as a **month with net CA > 0**. Cleaning
is destructive; a definition is auditable.

**Q. May 2026 looks lower than other Mays. Did you handle it?**
Yes — 21% fewer rows and 29% less revenue than May 2025, while April 2026 was the second-highest
month ever. That pattern reads as a mid-month extract, not a bad month. It matters because it
would make every client look more lapsed than they are. So the reactivation logic compares
**January–April only**, across all three years, which is symmetric and needs no assumption.
Confirming the extract date is an open question for the company.

**Q. What are the `***` client names?**
56 clients are named like `***STE X*** 09-25` — almost certainly closed or blocked accounts,
with a closure date. I flagged them in a column and excluded them from the vendor lists rather
than deleting them, and listed it as a question for the business. Recommending a call to a closed
account is the fastest way to lose a vendor's trust in the tool.

**Q. A client can appear under several vendors. Whose client is it?**
78% of clients appear under more than one vendor code — the data cannot answer it, it's a
business rule. My rule: the vendor with the **largest historical revenue** on that pair owns
the row. Critically, I compute lapsed pairs at *client* level first, then attribute — otherwise
a client who simply switched vendors would look like a lost customer to both of them.

## About the method

**Q. Why start with rules instead of going straight to ML?**
Three reasons. It gave the business a usable deliverable in week one. It produced the benchmark
the model had to beat — 0.616 AUC is not "chance", it's a real business heuristic and beating it
means something. And it forced me to understand the data before modeling it.

**Q. Why CatBoost and not a neural network / a recommender like ALS?**
231K training rows, 20 features, two of them categorical with many levels (brand, vendor code).
That's the region where gradient boosting on trees wins, and CatBoost handles categorical
features natively without me hand-encoding 262 brands. A deep model needs far more interaction
data than 15 months of monthly aggregates. I did run **logistic regression as a second benchmark**
(0.738) precisely so I could show that the gain from CatBoost is real but modest — most of the
signal is linear, and being honest about that is stronger than pretending the model is magic.

**Q. Why not a random train/test split?**
Because the deployment situation is "predict next year from this year". A random split would let
the model see 2026 rows while predicting 2026 — leakage — and would report a number the system
could never reproduce in production. So: **train 2024→2025, test 2025→2026, and the test window
is never touched** (`project.md` §11 required this from the start).

**Q. What exactly is one training row, and what's the label?**
One row = one (client, article) pair active in year Y. Features are built from **year Y only**.
Label = 1 if that client bought that article again in Jan–May of Y+1.

**Q. Why "year Y only"? Couldn't you give the model more history?**
That's the subtle one. The 2025 snapshot *could* see 2024, but the 2024 snapshot has no 2023.
If I gave the test rows two years of history and the training rows one, train and test would
have different feature distributions and the comparison would be meaningless. Strict symmetry
was worth more than extra features.

**Q. Your model uses `vendor_code` as a feature. Isn't that circular?**
It's a legitimate categorical — some vendors genuinely sell more repeat business than others —
and it ranks only 7th in importance (3.9). The dominant features are client health
(`client_n_articles` 16.9, `client_ca` 15.0, `client_late_share` 14.3) and brand (14.6).
The interesting finding is exactly that: **who the client is** predicts repurchase better than
how regular the pair was.

## About the results

**Q. Is AUC 0.762 good?**
For this data, yes, and the honest framing is comparative: 0.5 is chance, 0.616 is the business
rule, 0.738 is logistic regression, 0.762 is CatBoost. But AUC is a ranking metric over all
251,342 pairs, and a vendor doesn't call 251,342 pairs — so the number I'd actually defend is
**precision on the top 30 per vendor: 60.4% versus 42.3%**. That's the shape of the real
decision.

**Q. Your train AUC is 0.815 and test is 0.762. Isn't that overfitting?**
A gap of +0.053 across a full year boundary is small and expected — some of it isn't overfitting
at all, it's the year changing. I show all three numbers (train 0.815 / validation 0.798 /
test 0.762) on the validation page rather than reporting only the flattering one.

**Q. You trained for 800 iterations and stopped. Why not tune the model?**
I tried. A 4,000-iteration run improved the validation split (0.798 → 0.810) but **degraded the
real test** (0.762 → 0.748). Cause: the validation split is a random slice of the *same 2024
snapshot* as training, so improving on it rewards memorizing 2024 patterns that don't transfer.
I reverted, and stopped tuning entirely — with a single honest test window, any further
configuration search is just slowly fitting the test set. **A random split inside one year is
not a proxy for generalizing to the next year.**

**Q. Then why do you show percentages to vendors if the model is over-confident?**
Measured over-confidence grows with confidence: −0.2% error in the 0–10% band, **+10.7%** in the
70–80% band. So I tried to fix it properly — an isotonic curve fitted on the validation split,
never on the test. **It made calibration worse** (max gap 10.7% → 13.9%). Reason: on that split
the model is well calibrated, so the curve learns to push probabilities *up*; across a year they
don't come true at that rate. Root cause, again: no dataset represents a *fresh year*, because
2023 is missing. So the number is shown raw, labelled as an optimistic estimate, next to the
warning — and the correction is computed and stored but deliberately not applied.

**Q. Wait — that's the same missing year as the tuning problem?**
Yes, and that's the point worth making. **One gap in the data produced three separate limits**:
no tuning, no calibration, and no literal replay of the queue (the "lapsed" rule needs two prior
years, so replaying 2025's queue would need 2023). Three findings, one root cause, reached
independently.

**Q. How do you know the queue itself works, not just the model?**
I replayed what could be replayed: the ranking by expected value, the restriction to pairs bought
in both years, top-3 per client → 3,056 pairs, 1,099 clients, **48.8% repurchased against a 18.4%
base rate (×2.65)**. About 79 % of the announced expected value was actually realized.
What is *not* replayed (the lapsed filter, the status boost) is written in a box on the validation
page itself, not hidden in a report.

## About the judgment calls

**Q. Your priority formula uses `max(...)`, not a sum. Why?**
`priority = max(expected value, revenue gap) × status boost`. A client's revenue drop already
partly *contains* the lapsed articles — summing them would count the same missed money twice.

**Q. You display a TND amount. Is that a prediction?**
No, and it used to be labelled in a way that implied it was. I checked: for **39% of the queue**
(the declining and inactive clients) the displayed amount comes from history, not the model,
because the revenue gap wins the `max`. Worse, `priority_score` is a blend — nothing in the world
costs 50,608 TND. So I split them: `priority_score` still drives the **order**, but the vendor
screens now show a real figure they can say on the phone — *"42,174 TND bought last year, nothing
this year"*. Coverage: 100% of clients.

**Q. Why is the #1 article sometimes not the most likely one?**
Because ranking is probability × revenue. Measured on the live queue: article #1 is the most
*probable* 60.6% of the time and the most *expensive* 74.0% of the time; the correlation between
the two is only 0.23, so they're largely independent criteria. That's a deliberate choice —
"#1" means most money at stake — and it's now written in the interface instead of being implied.

**Q. Why explanations from rules and not from an LLM?**
An explanation shown to a vendor has to be verifiable against the database. Reason codes are
deterministic and auditable; a generated sentence can be fluent and wrong, and one wrong reason
in front of a client destroys trust in the whole tool. `project.md` §13 keeps a language model
for later, for rewriting *already verified* facts — never for choosing them.

## About the engineering

**Q. Why an API instead of exporting JSON for the React app?**
The extract contains **real people's names and their revenue** — that's why `data/` is gitignored.
A static export would put those names inside `web/`, then inside every `dist/` build, trivially
committed by accident. The FastAPI backend reads `data/` at runtime, binds to `127.0.0.1` only,
and returns aggregates or bounded slices.

**Q. Did you verify that, or just assume it?**
Verified — and it caught a real leak. A scan of `web/` against `clients.csv` found **one real
client name**, used as an example in a docstring. It would have gone to GitHub. I fixed it and
turned the manual check into a permanent test (`tests/test_no_data_leak.py`), so the guarantee
survives the next person who edits that file.

**Q. What did your tests actually catch?**
Real bugs, which is the only answer that means anything: a crash when a vendor search returned
no clients; feedback being saved against the wrong client; and the best one — **the API server
did not start**, while the whole suite passed, because `conftest.py` was fixing the import path
that the real process didn't have. Lesson: a green suite doesn't prove the program boots. I now
also start the binary. 67 tests today.

**Q. Anything the tests don't cover?**
Appearance. Tests prove the numbers are right; they don't prove a page is readable. That exact
class of bug hit twice — invisible dark-on-dark text from a Streamlit theme value reused for
widget internals, and a contrast regression. I flag visual review as a manual step rather than
claiming coverage I don't have.

## About what's next

**Q. What would you do with three more months?**
In order: (1) get order-level data — dates, quantities, statuses — which unlocks the real
question, "will this client buy within 30 days", and finally makes reorder timing possible;
(2) run a controlled pilot with a control group, because that's the only way to prove revenue
impact rather than correlation; (3) close the feedback loop — the vendor actions collected in
the app and the Excel sheets are the first *real* labels this system has ever had, since
everything today is inferred from revenue alone.

**Q. What would you do differently?**
Ask for the data schema before writing the specification. `project.md` describes favorites,
stock, quotations and contact history — none of it exists. The spec was right about *what a good
system needs*; it just wasn't yet informed about what was available, and half the project became
adapting to that.

---

## Three answers only you can give

Fill these in yourself before the defense — they're the questions where a rehearsed answer
sounds rehearsed and an honest one lands.

**What was the hardest moment?**
> _(one candidate: the 4,000-iteration run that looked like an improvement and wasn't — what it
> felt like to delete a better-looking number.)_

**What did you learn that you didn't know before?**
> _(…)_

**What are you actually proud of here?**
> _(…)_
