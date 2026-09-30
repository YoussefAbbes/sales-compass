Profiled it end to end. Here's what's actually there.

The source
One file: CA client vendeur article Janv-mai 24-25-26.xlsx, 51 MB, two sheets.

Janv-fev-mars-avril — 581K rows, codes only, 12 month columns (Jan–Apr × 2024/25/26)
Mai — 173K rows, 3 month columns, and it's the only sheet carrying names (vendor names, client names, article brand + designation)
build_dataset.py unpivots both into one tidy fact table.

What it fundamentally is
A pivoted monthly revenue aggregate — not a transaction log. One cell = "this client bought this much TND of this article from this vendor, this month." There is no such thing as an order in this data.

The fact table — ca_monthly.parquet
896,312 rows × 5 columns. Grain: (vendor_code, client_id, article_id, month) → ca

Vendors	54
Clients	2,060
Articles	40,321
Months	15 — Jan–May of 2024, 2025, 2026
Total CA	withheld (company-confidential)
Median line	72 TND (mean 146, max 107,730)
Negative rows	35,111 (3.9%) — returns/credit notes netted in, not stored separately
Reference tables — all scraped from the Mai sheet
File	Rows	Coverage of what appears in transactions
vendors.csv	44	44/54 — 10 vendors unnamed (18.5%)
clients.csv	1,778	1,778/2,060 — 282 unnamed (13.7%)
articles.csv	24,683	24,683/40,321 — 15,638 unnamed (38.8%)
262 brands (TIVY, LPR, SASI, VALE, FEBI, BOSH…). Designations look like TENDEUR COURROIE ALT-1340267-CORSA D-ASTRA G-H-J/CRUZE — part type, OEM reference, compatible vehicles, all crammed into one string.

★ Insight ─────────────────────────────────────

The naming gap isn't random, and that's why it's tolerable. Names exist only where a client bought that article in a May. So an unnamed article is one that never traded in May 2024/25/26 — structurally the dormant long tail. It also means the gap will never close by itself: no amount of reprocessing recovers a name the Excel doesn't contain. This is the reference-table request to the company, not a code fix.
─────────────────────────────────────────────────

What is NOT in it
This list matters more than what is, because it defines the ceiling on what any model here can do:

No dates — only months. "Reorders every 32 days" is unanswerable.
No quantities — only money. Can't say "usually orders 8 units."
No order status — a sale, a cancellation, a quote, and a return are indistinguishable except by sign.
No June–December. A 7-month hole every year. A client who buys every October is invisible.
No stock, no favorites, no contact history, no prices/margins.
Structural facts that shape everything downstream
Extreme sparsity. 621,094 distinct (client, article) pairs out of a possible 83M grid — 0.75% fill. And 74.3% of pairs appear in exactly one month. Most client-article relationships are one-offs, not habits.

Heavy concentration. Top 10% of articles = 73.2% of CA. Top 10% of clients = 56.2%.

Ownership is many-to-many. 1,616 of 2,060 clients (78.4%) appear under more than one vendor code — up to 4. "Whose client is this?" needs a business rule; the data won't answer it. Vendor size ranges from 1 client to 948.

One thing the profiling flags that you should confirm
May 2026 looks incomplete:

Month	Rows	CA vs May 2025
2024-05	62,900	−3%
2025-05	65,956	baseline
2026-05	51,981	−29%
That's 21% fewer rows and 29% less revenue than May 2025 — the lowest of all 15 months, while 2026-04 was the second highest. That pattern reads like a mid-month extract, not a bad month.

This is listed as an open question in progress.md, and it's not cosmetic: your lapsed/reactivation logic compares Jan–May 2026 against prior Jan–May windows. If May 2026 is partial, every client looks more lapsed than they are, and the queue is systematically biased toward false alarms.

There are two defensible ways to handle it and the choice is a business call, not a technical one:

Drop May 2026 entirely and compare Jan–Apr across all three years — clean, symmetric, loses the freshest month.
Keep it but scale — treat it as partial and prorate, which requires assuming a completeness ratio you can't verify.
I'd lean toward the first until the company confirms, since it's honest without needing an assumption. Do you know when that extract was pulled? If it was taken mid-May, that settles it


"With the data available — monthly revenue only, no dates, no quantities, five months a year — the model doesn't predict when a client will buy. It predicts who is worth calling first, and it does that 1.5× better than the current rule and 3× better than chance, validated on a year it never saw."