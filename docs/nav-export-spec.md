# NAV export specification

What to export from Dynamics NAV, how often, and in exactly what shape, so that the
recommendation pipeline runs on it without manual fixing.

Hand this document to whoever runs the NAV exports. It assumes no knowledge of the
project. Written 2026-08-20 against the pipeline as it stands.

For what happens to the files afterwards, see [model-lifecycle.md](model-lifecycle.md).

---

## 1. Why exports are needed at all

The system predicts, for each client, which parts they are likely to buy **next**. It
learned how to do that from 30 months of history, and that learning is finished — no more
data is needed to train it.

What it needs is a **current reading**. Its single most important input is *when each
client last bought*. Between exports that reading goes out of date, and the system starts
recommending clients who have already been served.

So an export is not a data-collection exercise. It is re-running a report that already
exists, with a later date filter.

---

## 2. When to run one

The two models advance on different clocks.

| | Repurchase model | Adoption model |
| --- | --- | --- |
| Advances | every new month of data | only when a **half-year closes** |
| Needs data through | any month end | 30 June or 31 December |
| How fast its output ages | half-life about 2 weeks | slowly, in half-year steps |

**Recommended cadence: calendar-quarterly.** Export covering through **31 March**,
**30 June**, **30 September**, **31 December**.

Two of those four dates are exactly the half-year boundaries the adoption model needs, so
a single quarterly schedule keeps both models correct.

| Cadence | Verdict |
| --- | --- |
| Monthly | Best, if the export is cheap to run. Repurchase lists stay fresh; adoption ignores the extras. |
| **Quarterly** | **Recommended.** Serves both models, four runs a year. |
| Twice a year (Jun + Dec) | Bare minimum. Adoption stays correct, but repurchase lists arrive mostly spent. |

> **Always end on a closed month.** Never export a partial month. The last month is used to
> compute how recently each client bought, and a half-recorded month makes every client
> look quieter than they are.

---

## 3. File 1 — the sales ledger

The main file. One row per sales line.

| | |
| --- | --- |
| NAV report | Écritures comptables article (item ledger entries) |
| Sheet name | **`Écritures comptables article`** — exactly, accent included |
| File name | must start with `vente` and end with `.xlsx` |
| Example | `vente 01-07-2026 -- 30-09-2026.xlsx` |
| Date filter | starts the day after the previous export ended; ends on a closed month |

### Columns

Thirteen columns are imported and one more is checked. Extra columns are ignored, so it is
safe to leave the report as it is.

| NAV column | Used for |
| --- | --- |
| `Date comptabilisation` | the transaction date |
| `Type document` | identifying returns |
| `N° document` | counting distinct orders |
| `N° article` | the article identifier |
| `Désignation article` | the article name |
| `Enseigne` | the brand |
| `N° origine` | **the client identifier** |
| `Code vendeur` | the salesperson |
| `Code magasin` | the warehouse |
| `Quantité` | quantity |
| `Montant vente (réel)` | revenue |
| `Coût total (réel)` | cost, for margin |
| `Discount Amount` | discount |
| `Type écriture` | *checked, not imported* — see below |

### Two hard rules

1. **Filter the export to sales only.** Every row must have `Type écriture` = `Vente`. The
   loader stops with an error if any other value appears, rather than silently mixing in
   movements that are not sales.

2. **Do not rename document types.** Returns are recognised by the exact string
   `Réception retour vente` in `Type document`. If that label changes, returns get counted
   as ordinary sales.

### Size limit

Excel holds 1,048,576 rows per sheet. The existing history needed two files for that
reason. If a single export would exceed the limit, split it into two files — both starting
with `vente` — and make sure the split is a clean date cut with **no overlap**.

---

## 4. Files 2 and 3 — the master lists

Current-state snapshots of articles and clients. Unlike the sales ledger, these are always
exported **in full** and replace the previous version.

| File name | Sheet name |
| --- | --- |
| `deatils articles nav.xlsx` | `Liste des articles` |
| `deatils clients nav.xlsx` | `Liste des clients` |

> **Keep the misspelling.** The file names are read literally by the code, including
> `deatils` rather than `details`. Correcting the spelling will stop the pipeline finding
> the files.

### Articles — columns

**Required.** The script stops with an error if any is missing:

`N°`, `Stocks`, `Plage de Stock`

**Also used when present:** `Enseigne`, `Désignation`, `Code catégorie article`,
`Marque par défaut`, `Marque`, `N° origine parent`, `Prix unitaire`, `Coût unitaire`,
`Pareto`

### Clients — columns

**Required.** The script stops with an error if any is missing:

`N°`, `Bloqué`

**Also used when present:** `Nom`, `Client parent`, `Cause dernier blocage`, `Type client`,
`Ville`, `Gouvernerat`, `Code circuit`, `N° téléphone`, `N° téléphone 2`, `N° téléphone 3`,
`Contact`, `Adresse`, `N° identif. intracomm.`, `Solde`, `Crédit autorisé DS`,
`Code vendeur`, `<Code Vendeur 2>`, `<Code Vendeur 3>`

> The last two column headers contain **literal angle brackets**, exactly as written above.
> They are not placeholders and must not be "cleaned up".

Why `Bloqué` and the stock columns are mandatory while the rest are optional: they drive
safety filters. A missing phone number blanks a display field. A missing `Bloqué` would
make every client look un-blocked and switch off the rule that stops vendors calling
non-paying accounts — silently. So those fail loudly instead.

---

## 5. Where the files go

All three go in the `exported-data/` folder of the project, alongside the previous exports.

**Keep the old `vente*.xlsx` files.** The pipeline reads every file matching that pattern
and joins them into one history. Deleting them shortens the history the model can see.

The two master files are simply overwritten each time.

---

## 6. Checklist before handing the files over

- [ ] Sales export filtered to `Type écriture` = `Vente` only
- [ ] Sales date range starts the day **after** the previous export ended
- [ ] Sales date range ends on the **last day of a closed month**
- [ ] Sheet named `Écritures comptables article`
- [ ] File name starts with `vente`
- [ ] Under 1,048,576 rows, or split cleanly into two non-overlapping files
- [ ] Both master files exported in full, with the `deatils` spelling kept
- [ ] Master sheets named `Liste des articles` and `Liste des clients`
- [ ] All three files placed in `exported-data/`, old `vente*.xlsx` files left in place

---

## 7. The five things that go wrong

| Mistake | What happens |
| --- | --- |
| Sheet renamed, or accents removed | Pipeline cannot find the sheet, stops immediately |
| Master file renamed to fix `deatils` | File not found, stops immediately |
| New sales file **overlaps** the previous date range | Sales counted twice — **no error, wrong numbers** |
| Export not filtered to `Vente` | Hard error on `Type écriture` |
| Export ends mid-month | Runs fine, but every client looks less active than they are |

Only the third one fails quietly. It is the one worth double-checking.

---

## 8. What happens next

Once the files are in `exported-data/`, refreshing takes about 15 minutes and no code
changes. The commands are in [model-lifecycle.md §3](model-lifecycle.md). In short: rebuild
the ledger, re-score, rebuild the queue, run the tests.

The new prediction window follows automatically from the export's end date:

| Export covers through | New predictions cover | Valid until |
| --- | --- | --- |
| 30 September 2026 | Oct – Dec 2026 | 31 December 2026 |
| 31 December 2026 | Jan – Mar 2027 | 31 March 2027 |
| 31 March 2027 | Apr – Jun 2027 | 30 June 2027 |

Nothing needs configuring for this — the pipeline reads the last transaction date and works
out the window itself.

> **Note.** Salesperson *names* are not in the NAV export, only their codes. `vendors.csv`
> comes from an older workbook and should be preserved rather than regenerated — see known
> gap 1 in [model-lifecycle.md §5](model-lifecycle.md).

---

## 9. Request template

Copy and adapt when asking for an export.

> **Subject:** NAV export request — item ledger entries from &lt;START DATE&gt;
>
> Could you please run three exports:
>
> **1. Item ledger entries (Écritures comptables article)**
> - Date range: **&lt;START DATE&gt;** to **&lt;END OF LAST CLOSED MONTH&gt;**
> - Filter: `Type écriture` = `Vente` only
> - Keep the sheet named `Écritures comptables article`
> - Save as `vente <START> -- <END>.xlsx`
> - If it exceeds ~1,000,000 rows, split into two files with no overlapping dates
>
> **2. Article list** — full export, sheet `Liste des articles`, saved as
> `deatils articles nav.xlsx` (please keep this exact spelling)
>
> **3. Client list** — full export, sheet `Liste des clients`, saved as
> `deatils clients nav.xlsx` (please keep this exact spelling)
>
> These are the same three reports as the previous export — only the date filter on the
> first one changes. Please keep all columns as they are.
