# Transactional Re-foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monthly-aggregate data foundation with the day-level Dynamics NAV ledger, so the model trains on full-year labels and rolling chronological snapshots instead of one Jan–May snapshot per year.

**Architecture:** A new ingest layer (`src/build_transactions.py`) turns the two NAV sales workbooks into a day-level `data/transactions.parquet`, and the two NAV master workbooks into `data/articles_master.csv` / `data/clients_master.csv`. A reconciliation step proves the new table reproduces the existing `data/ca_monthly.parquet` before anything depends on it. A pure snapshot builder (`src/snapshots.py`) emits `(client, article)` rows at any date `T` using a 12-month feature window and a 3-month label window, and `src/train_rolling.py` trains over 16 such snapshots with a time-forward validation split and an embargo. The existing delivery layer (queue, API, dashboards) is untouched by this plan.

**Tech Stack:** Python 3.13, pandas, pyarrow, openpyxl, catboost, scikit-learn, pytest.

## Global Constraints

- All amounts are TND. Round monetary values to 3 decimals (millime precision).
- Identifiers (`client_id`, `article_id`, `vendor_code`, `doc_no`) are **strings**; leading zeros must be preserved (`"007"`, `"01381"`, `"000323"`).
- **No feature may reference data at or after its snapshot date `T`.** Every snapshot function takes `T` explicitly and must be tested for this.
- **Master-file columns are current-state snapshots, not history.** `Bloqué`, `Cause dernier blocage`, `Stocks`, `Plage de Stock`, `Solde`, `Crédit autorisé DS` describe 2026-08-10 only. They may be used as **scoring-time filters** (a later plan). They must **never** appear in a training feature set. Any task adding them to `NUMERIC`/`CATEG` is wrong.
- `exported-data/` is confidential and must never be committed. `.gitignore` currently protects `*.xlsx` but not other files placed there.
- **Tests must never depend on the raw workbooks.** Unit-test pure functions with small synthetic DataFrames, following the existing convention in `tests/test_validation.py`.
- **Importing a module must have no side effects.** Every new module in this plan is imported by a test, so *both* the `sys.stdout = io.TextIOWrapper(...)` rebinding **and** any `DATA.mkdir(...)` / `REPORTS.mkdir(...)` that the existing *script* modules (`build_dataset.py`, `contact_queue.py`, `train_model.py`, …) do at module level must instead move **inside `main()`**. A module-level `mkdir` means `from build_transactions import normalise_ledger` silently creates a directory on disk. Compare `src/validation.py`, the repo's model test-imported module: it builds `Path` objects at module level and mutates nothing. This is the repo's existing split: the six modules the tests import (`validation`, `dataset_profile`, `prospect_features`, `prospect_model`, `article_prospects`, `call_script`) all leave stdout alone. Rebinding at import time fights pytest's output capture.
- **Never overwrite v1 artifacts.** `reports/model_catboost.md`, `reports/model_metrics.json`, `data/pair_scores_2026.csv` and `data/model_catboost.cbm` are the before-comparison and the fallback.
- Everything written to `reports/` must be **aggregates only** — never client-level or named rows. `reports/` is committed.
- Snapshot split constants (used by Tasks 5 and 6, defined once in `src/snapshots.py`):
  - `LOOKBACK_MONTHS = 12`, `HORIZON_MONTHS = 3`
  - `TRAIN_T`: 2025-01-01 … 2025-08-01 (8 snapshots)
  - `VAL_T`: 2025-09-01 … 2025-10-01 (2 snapshots)
  - `EMBARGO_T`: 2025-11-01 … 2025-12-01 (2 snapshots, discarded)
  - `TEST_T`: 2026-01-01 … 2026-04-01 (4 snapshots, untouched)
- Commit style follows the repo: `feat(refoundation): ...`, `test(refoundation): ...`, `chore(refoundation): ...`.

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/build_transactions.py` | **Create.** Read the two NAV sales workbooks → `data/transactions.parquet`. Split into `normalise_ledger()` (pure, tested) and `read_ledger()` (I/O). |
| `src/build_masters.py` | **Create.** Read the two NAV master workbooks → `data/articles_master.csv`, `data/clients_master.csv`. Pure `normalise_articles()` / `normalise_clients()`. |
| `src/reconcile.py` | **Create.** Prove `transactions.parquet` reproduces `ca_monthly.parquet`. Writes `reports/reconciliation.md`. |
| `src/blind_spot.py` | **Create.** Measure the Jun–Dec label blind spot. Writes `reports/blind_spot.md`. |
| `src/snapshots.py` | **Create.** Pure rolling-snapshot builder + split constants. No I/O. |
| `src/train_rolling.py` | **Create.** Train the continuity benchmark and the 90-day model. Writes `reports/model_rolling.md`, `reports/rolling_metrics.json`, `data/model_rolling.cbm`. |
| `tests/test_build_transactions.py` | **Create.** Synthetic-frame tests for `normalise_ledger`. |
| `tests/test_snapshots.py` | **Create.** Synthetic-frame tests for window boundaries and leakage. |
| `tests/test_blind_spot.py` | **Create.** Synthetic-frame tests for the base-rate comparison. |
| `tests/test_no_data_leak.py` | **Modify.** Add phone / tax-ID / address scanning for the new master files. |
| `src/train_model.py:281` | **Modify.** Fix `_modesavel` → `save_model`. |
| `.gitignore` | **Modify.** Add `/exported-data/`. |

---

### Task 0: Safety gate

Nothing may read the new data until this task is committed. It fixes a latent runtime crash, closes a gitignore gap, and extends the leak guarantee to the new PII surface (phone numbers, tax IDs, addresses) *before* any of it can reach a committed directory.

**Files:**
- Modify: `src/train_model.py:281`
- Modify: `.gitignore`
- Modify: `tests/test_no_data_leak.py`
- Create: `reports/v1-monthly/` (frozen copies)

**Interfaces:**
- Consumes: nothing.
- Produces: `reports/v1-monthly/` containing the frozen v1 artifacts. Later tasks compare against these.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b feat/transactional-refoundation
```

- [ ] **Step 2: Freeze the v1 artifacts**

```bash
mkdir -p reports/v1-monthly
cp reports/model_catboost.md reports/v1-monthly/model_catboost.md
cp reports/model_metrics.json reports/v1-monthly/model_metrics.json
cp reports/backtest.md reports/v1-monthly/backtest.md
cp reports/prospect_metrics.json reports/v1-monthly/prospect_metrics.json
cp reports/prospect_model.md reports/v1-monthly/prospect_model.md
```

- [ ] **Step 3: Fix the corrupted save call**

In `src/train_model.py`, line 281, replace:

```python
    model._modesavel(str(DATA / "model_catboost.cbm"))
```

with:

```python
    model.save_model(str(DATA / "model_catboost.cbm"))
```

- [ ] **Step 4: Verify the fix compiles and the attribute exists**

Run: `py -c "from catboost import CatBoostClassifier; assert hasattr(CatBoostClassifier(), 'save_model'); print('save_model OK')"`
Expected: `save_model OK`

Run: `py -m py_compile src/train_model.py`
Expected: no output, exit 0

- [ ] **Step 5: Close the gitignore gap**

Append to `.gitignore`, after the `/data/` line:

```gitignore
# Raw NAV exports. *.xlsx already covers the workbooks, but this folder must never
# accept any committed file — a future .csv or .txt export would otherwise slip in.
/exported-data/
```

- [ ] **Step 6: Verify the gitignore change in both directions**

Run: `git check-ignore -q "exported-data/GROUP MQARQUE.txt" && echo IGNORED || echo EXPOSED`
Expected: `IGNORED`

Run: `git check-ignore -q fixtures/data/clients.csv && echo BROKEN || echo "fixtures still committable"`
Expected: `fixtures still committable`

- [ ] **Step 7: Write the failing PII test**

Add to `tests/test_no_data_leak.py`, after the existing `MIN_DESIGNATION` constant:

```python
# Tunisian landline/mobile numbers are 8 digits — long enough to be unambiguous.
MIN_PHONE = 8
# Tax identifiers look like "0569199MNC000".
MIN_TAX_ID = 10
# Street names are shared between clients; require a longer match to avoid noise.
MIN_ADDRESS = 12
```

And this helper beside the existing `_first_hit`:

```python
def _first_token_hit(values, texts, min_len):
    """Like _first_hit, but the value must appear as a standalone token.

    Phone numbers and tax ids are short alphanumeric runs, and the committed
    fixtures contain high-precision floats. An 8-digit needle has 1e8 possible
    values while pair_scores_2026.csv alone offers ~300k 8-digit windows across
    its p_model and manual_score columns, so plain substring matching produces
    a predictable ~14 false positives against clean data — and did: '12345678'
    occurs inside the probability 0.6712345678857319.

    Requiring the match not to be flanked by alphanumerics keeps the guarantee
    intact, because a leaked identifier is a field value delimited by a comma,
    quote, colon or newline. The accepted trade-off is that a value welded into
    a longer alphanumeric run (tel12345678) would be missed; that is not a
    shape any export or fixture-builder produces.
    """
    for value in values:
        if not isinstance(value, str) or len(value) < min_len:
            continue
        pattern = None
        for path, text in texts.items():
            # `in` is CPython's C substring search and runs at GB/s; the
            # lookaround regex runs in the far slower regex engine. A boundary
            # match is necessarily a substring match, so this prefilter is
            # exactly equivalent and keeps the regex off the hot path. Without
            # it the scan is ~12k values x 9.4 MB of regex and takes >10 min.
            if value not in text:
                continue
            if pattern is None:
                pattern = re.compile(
                    rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])")
            if pattern.search(text):
                return value, path
    return None, None
```

This requires `import re` at the top of the file.

And add this test at the end of the file:

```python
@pytest.mark.parametrize("tree", SCANNED_TREES)
def test_no_master_pii_appears(project_root, data_dir, tree):
    """The NAV client master adds phone numbers, tax ids and addresses.

    The original extract carried only names and revenue. These columns are a new and
    larger identity surface, and fixtures/ is built from them by a script — so the
    guarantee has to grow with the data, not after a leak.
    """
    master = data_dir / "clients_master.csv"
    if not master.exists():
        pytest.skip("data/clients_master.csv missing — run py src/build_masters.py")

    df = pd.read_csv(master, dtype=str, encoding="utf-8-sig")
    texts = _tree_text(project_root, tree)

    # Identifier columns match on token boundaries, free-text columns on
    # substring. See _first_token_hit for why.
    for column, min_len, finder in (
        ("phone_1", MIN_PHONE, _first_token_hit),
        ("phone_2", MIN_PHONE, _first_token_hit),
        ("phone_3", MIN_PHONE, _first_token_hit),
        ("tax_id", MIN_TAX_ID, _first_token_hit),
        ("address", MIN_ADDRESS, _first_hit),
        ("contact", MIN_NAME, _first_hit),
    ):
        if column not in df.columns:
            continue
        values = df[column].dropna().unique().tolist()
        hit, path = finder(values, texts, min_len)
        assert hit is None, (
            f"{column} value {hit!r} found in {path} — the NAV master carries personal "
            f"contact data. Serve it from the API, or re-run tools/make_fixtures.py."
        )
```

- [ ] **Step 8: Run the test to verify it skips cleanly**

Run: `py -m pytest tests/test_no_data_leak.py -v`
Expected: the three existing test groups PASS; `test_no_master_pii_appears` reports SKIPPED with "data/clients_master.csv missing" (Task 2 creates that file and turns the test live).

- [ ] **Step 9: Commit**

```bash
git add .gitignore src/train_model.py tests/test_no_data_leak.py reports/v1-monthly/
git commit -m "chore(refoundation): safety gate before ingesting NAV data

Fix the corrupted model.save_model call, ignore exported-data/ wholesale,
freeze the v1 monthly-basis artifacts for later comparison, and extend the
leak scan to the phone/tax-id/address columns the NAV master introduces."
```

---

### Task 1: Ingest the sales ledger

Turns the two NAV sales workbooks into one day-level fact table. The pure `normalise_ledger()` carries all the cleaning rules and is unit-tested on synthetic frames; `read_ledger()` only does openpyxl I/O.

**Files:**
- Create: `src/build_transactions.py`
- Test: `tests/test_build_transactions.py`

**Interfaces:**
- Consumes: `reports/v1-monthly/` exists (Task 0).
- Produces:
  - `normalise_ledger(raw: pd.DataFrame) -> pd.DataFrame` — input has the 17 raw French column names; output has columns `date` (datetime64), `doc_type` (str), `doc_no` (str), `client_id` (str), `article_id` (str), `vendor_code` (str), `enseigne` (str), `designation` (str), `warehouse` (str), `qty` (float), `amount` (float), `cost` (float), `discount` (float), `is_return` (bool).
  - `LEDGER_COLUMNS: dict[str, str]` — raw French name → output name.
  - `data/transactions.parquet` with that schema.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_transactions.py`:

```python
# -*- coding: utf-8 -*-
"""normalise_ledger is pure: synthetic frames only, never the 121 MB workbook."""
import pandas as pd
import pytest

from build_transactions import normalise_ledger


def raw_row(**over):
    """One raw ledger row using the exact French column names NAV exports."""
    row = {
        "Date comptabilisation": pd.Timestamp("2024-01-02"),
        "Type écriture": "Vente",
        "Type document": "Expédition vente",
        "N° document": "B241000001",
        "N° article": "01381",
        "Désignation article": "ANTIGEL 1.5L ROUGE / CONCENT",
        "Enseigne": "FEBI",
        "N° origine": "0289",
        "Nom": "STE EXEMPLE",
        "Code vendeur": "007",
        "Nom Vend.": "v10 EXEMPLE",
        "Code magasin": "A02",
        "Quantité": 12.0,
        "Montant vente (réel)": 191.592,
        "Coût total (réel)": 140.0,
        "Discount Amount": 0.0,
        "CU": 1.0,
    }
    row.update(over)
    return row


def test_identifiers_stay_strings_with_leading_zeros():
    out = normalise_ledger(pd.DataFrame([raw_row()]))
    assert out.loc[0, "client_id"] == "0289"
    assert out.loc[0, "article_id"] == "01381"
    assert out.loc[0, "vendor_code"] == "007"


def test_returns_are_flagged_not_dropped():
    out = normalise_ledger(pd.DataFrame([
        raw_row(),
        raw_row(**{"Type document": "Réception retour vente",
                   "Quantité": -2.0, "Montant vente (réel)": -31.932}),
    ]))
    assert out["is_return"].tolist() == [False, True]
    assert len(out) == 2, "returns are evidence, not noise — never dropped at ingest"


def test_amounts_are_rounded_to_millimes():
    out = normalise_ledger(pd.DataFrame([
        raw_row(**{"Montant vente (réel)": 191.5924999})
    ]))
    assert out.loc[0, "amount"] == 191.592


def test_whitespace_is_stripped_from_identifiers():
    out = normalise_ledger(pd.DataFrame([
        raw_row(**{"N° origine": "  0289 ", "N° article": " 01381"})
    ]))
    assert out.loc[0, "client_id"] == "0289"
    assert out.loc[0, "article_id"] == "01381"


def test_date_becomes_datetime64():
    out = normalise_ledger(pd.DataFrame([raw_row()]))
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_rows_with_no_identifier_are_dropped():
    out = normalise_ledger(pd.DataFrame([raw_row(), raw_row(**{"N° origine": None})]))
    assert len(out) == 1


def test_unexpected_entry_type_raises():
    """Type écriture is uniformly 'Vente' in both workbooks. Anything else means the
    export changed shape and every downstream assumption needs re-checking."""
    with pytest.raises(ValueError, match="Type écriture"):
        normalise_ledger(pd.DataFrame([raw_row(**{"Type écriture": "Achat"})]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_build_transactions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_transactions'`

- [ ] **Step 3: Write the implementation**

Create `src/build_transactions.py`:

```python
# -*- coding: utf-8 -*-
"""
Build the day-level transaction table from the NAV sales ledger export.

Input : exported-data/vente 01-01-24 -- 28-05-2025.xlsx
        exported-data/vente 29-05-2025 -- 30-06-2026.xlsx
        (sheet "Écritures comptables article", 17 columns each)
Output: data/transactions.parquet

The two workbooks are split at Excel's 1,048,576-row ceiling, not at a data
boundary: file 1 ends 2025-05-28 and file 2 begins 2025-05-29. They concatenate
directly. Expect a third file for later periods.

Cleaning rules (mirroring src/build_dataset.py, which this supersedes):
  - IDs kept as strings, whitespace stripped (leading zeros are significant).
  - Amounts rounded to 3 decimals (TND millime precision).
  - Returns are KEPT and flagged via is_return. They are evidence about a pair,
    not noise; each analysis decides how to treat them.
  - Rows with no client or article identifier are dropped.
  - cost carries NAV's sign UNCHANGED: "Coût total (réel)" is negative while
    "Montant vente (réel)" is positive. Margin is therefore amount + cost,
    never amount - cost. Inventing a sign convention here would hide it.
"""
import io
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exported-data"
DATA = ROOT / "data"

SHEET = "Écritures comptables article"
RETURN_DOC_TYPE = "Réception retour vente"

LEDGER_COLUMNS = {
    "Date comptabilisation": "date",
    "Type document": "doc_type",
    "N° document": "doc_no",
    "N° article": "article_id",
    "Désignation article": "designation",
    "Enseigne": "enseigne",
    "N° origine": "client_id",
    "Code vendeur": "vendor_code",
    "Code magasin": "warehouse",
    "Quantité": "qty",
    "Montant vente (réel)": "amount",
    "Coût total (réel)": "cost",
    "Discount Amount": "discount",
}

ID_COLUMNS = ["doc_no", "article_id", "client_id", "vendor_code", "warehouse"]
NUMERIC_COLUMNS = ["qty", "amount", "cost", "discount"]


def _clean_id(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def normalise_ledger(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw NAV columns -> the analysis schema. Pure: no I/O, no globals."""
    entry_types = set(raw["Type écriture"].dropna().unique())
    if entry_types - {"Vente"}:
        raise ValueError(
            f"unexpected Type écriture values {sorted(entry_types - {'Vente'})}: the "
            "export is no longer sales-only and downstream assumptions need review"
        )

    df = raw.rename(columns=LEDGER_COLUMNS)[list(LEDGER_COLUMNS.values())].copy()

    df["date"] = pd.to_datetime(df["date"])
    for col in ID_COLUMNS:
        df[col] = df[col].map(_clean_id)
    df["designation"] = df["designation"].fillna("").astype(str).str.strip()
    df["enseigne"] = df["enseigne"].fillna("").astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        # astype is not redundant: when every value is integral pandas infers
        # int64, and the declared interface promises float for all four.
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                   .fillna(0.0).round(3).astype("float64"))

    df["is_return"] = df["doc_type"] == RETURN_DOC_TYPE

    df = df[(df["client_id"] != "") & (df["article_id"] != "")]
    return df.reset_index(drop=True)


def read_ledger(path: Path) -> pd.DataFrame:
    """Stream one workbook into a DataFrame.

    Kept separate from normalise_ledger so the cleaning rules stay unit-testable
    without a 121 MB fixture.

    pd.read_excel is deliberately NOT used: it routes every cell through an
    openpyxl Cell object, which for 1.05M rows x 17 columns costs several GB and
    many minutes. read_only=True + iter_rows streams raw tuples instead.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb[SHEET].iter_rows(values_only=True)
        header = list(next(rows))
        return pd.DataFrame(list(rows), columns=header)
    finally:
        wb.close()


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    DATA.mkdir(exist_ok=True)
    paths = sorted(EXPORTS.glob("vente*.xlsx"))
    if not paths:
        raise SystemExit(f"no vente*.xlsx found under {EXPORTS}")

    frames = []
    for path in paths:
        print(f"reading {path.name} ...", flush=True)
        part = normalise_ledger(read_ledger(path))
        print(f"  {len(part):,} rows  {part['date'].min():%Y-%m-%d} -> "
              f"{part['date'].max():%Y-%m-%d}")
        frames.append(part)

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df.to_parquet(DATA / "transactions.parquet", index=False)

    print("\n=== TRANSACTION BUILD REPORT ===")
    print(f"rows              : {len(df):,}")
    print(f"date range        : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"documents         : {df['doc_no'].nunique():,}")
    print(f"clients / articles / vendors : {df['client_id'].nunique():,} / "
          f"{df['article_id'].nunique():,} / {df['vendor_code'].nunique():,}")
    print(f"returns           : {df['is_return'].sum():,} ({df['is_return'].mean():.2%})")
    print(f"total amount      : {df['amount'].sum():,.3f} TND")
    print(f"total cost        : {df['cost'].sum():,.3f} TND")
    gaps = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    present = set(df["date"].dt.to_period("M").astype(str))
    missing = [f"{d:%Y-%m}" for d in gaps if f"{d:%Y-%m}" not in present]
    print(f"missing months    : {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_build_transactions.py -v`
Expected: 7 passed

- [ ] **Step 5: Build the real table**

Run: `py src/build_transactions.py`
Expected: ~1,881,560 rows, range `2024-01-02 -> 2026-06-30`, returns ≈2.90%, `missing months : none`.

**Gate:** if `missing months` is not `none`, or the row count differs from 1,881,560 by more than 1%, stop and report before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/build_transactions.py tests/test_build_transactions.py
git commit -m "feat(refoundation): ingest the NAV sales ledger to a day-level table

1.88M lines across the two workbooks become data/transactions.parquet with
dates, quantities, cost and an explicit is_return flag. Returns are flagged
rather than netted, which is what makes a clean purchase label possible."
```

---

### Task 2: Ingest the master files

Supplies the article and client attributes. Only stable, non-current-state columns go into features later; the current-state columns are carried but flagged in the docstring as filter-only.

**Files:**
- Create: `src/build_masters.py`
- Test: extends `tests/test_build_transactions.py` (same ingest concern, same test cycle)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `normalise_articles(raw: pd.DataFrame) -> pd.DataFrame` — columns `article_id`, `enseigne`, `designation`, `category`, `vehicle_make`, `vehicle_model`, `oem_ref`, `unit_price`, `unit_cost`, `stock`, `stock_class`, `pareto`.
  - `normalise_clients(raw: pd.DataFrame) -> pd.DataFrame` — columns `client_id`, `client_name`, `parent_id`, `blocked_detail` (the raw NAV value: `""` / `Facturer` / `Livrer` / `Tous`), `blocked` (bool, derived: `blocked_detail != ""`), `blocked_reason`, `client_type`, `city`, `governorate`, `circuit`, `phone_1`, `phone_2`, `phone_3`, `contact`, `address`, `tax_id`, `balance`, `credit_limit`, `vendor_1`, `vendor_2`, `vendor_3`.
  - `data/articles_master.csv`, `data/clients_master.csv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_transactions.py`:

```python
from build_masters import normalise_articles, normalise_clients


def test_article_master_keeps_leading_zero_ids_and_splits_vehicle():
    raw = pd.DataFrame([{
        "N°": "000323", "Enseigne": "XTRA", "Désignation": "DEFLECTEUR AIR KDH",
        "Code catégorie article": "PCE-CARR", "Marque par défaut": "TY",
        "Marque": "TY-HIACE", "N° origine parent": "42311-26290",
        "Prix unitaire": 16.124, "Coût unitaire": 14.021, "Stocks": 12,
        "Plage de Stock": "MORT-", "Pareto": "DD",
    }])
    out = normalise_articles(raw)
    assert out.loc[0, "article_id"] == "000323"
    assert out.loc[0, "vehicle_make"] == "TY"
    assert out.loc[0, "vehicle_model"] == "TY-HIACE"
    assert out.loc[0, "stock_class"] == "MORT-"


def test_client_master_normalises_blocking_to_a_boolean_plus_detail():
    raw = pd.DataFrame([
        {"N°": "0001", "Nom": "A.P.R DISTRIBUTION", "Bloqué": "Facturer",
         "Cause dernier blocage": "Factures non réglées"},
        {"N°": "0002", "Nom": "Mr EXEMPLE", "Bloqué": None,
         "Cause dernier blocage": None},
    ])
    out = normalise_clients(raw)
    assert out["blocked"].tolist() == [True, False]
    assert out.loc[0, "blocked_detail"] == "Facturer"
    assert out.loc[0, "blocked_reason"] == "Factures non réglées"


def test_client_master_collects_all_three_vendor_codes():
    raw = pd.DataFrame([{
        "N°": "0003", "Nom": "Mr EXEMPLE", "Code vendeur": "550",
        "<Code Vendeur 2>": "585", "<Code Vendeur 3>": None,
    }])
    out = normalise_clients(raw)
    assert out.loc[0, "vendor_1"] == "550"
    assert out.loc[0, "vendor_2"] == "585"
    assert out.loc[0, "vendor_3"] == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_build_transactions.py -v -k master`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_masters'`

- [ ] **Step 3: Write the implementation**

Create `src/build_masters.py`:

```python
# -*- coding: utf-8 -*-
"""
Build the article and client master tables from the NAV master exports.

Input : exported-data/deatils articles nav.xlsx  (sheet "Liste des articles")
        exported-data/deatils clients nav.xlsx   (sheet "Liste des clients")
Output: data/articles_master.csv, data/clients_master.csv

WARNING — these files are CURRENT-STATE SNAPSHOTS, not history. Stocks, Bloqué,
Cause dernier blocage, Solde, Plage de Stock and Crédit autorisé describe
2026-08-10 and nothing else.
Using them as TRAINING FEATURES is leakage: a client blocked today was very
often blocked *because* they stopped buying, so a model would learn the outcome
it is meant to predict. They are safe only as SCORING-TIME FILTERS, which is
where project.md §7 step 5 puts them.

Stable columns that ARE safe as features: category, vehicle make/model, OEM
reference, enseigne.
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exported-data"
DATA = ROOT / "data"

ARTICLE_COLUMNS = {
    "N°": "article_id",
    "Enseigne": "enseigne",
    "Désignation": "designation",
    "Code catégorie article": "category",
    "Marque par défaut": "vehicle_make",
    "Marque": "vehicle_model",
    "N° origine parent": "oem_ref",
    "Prix unitaire": "unit_price",
    "Coût unitaire": "unit_cost",
    "Stocks": "stock",
    "Plage de Stock": "stock_class",
    "Pareto": "pareto",
}

CLIENT_COLUMNS = {
    "N°": "client_id",
    "Nom": "client_name",
    "Client parent": "parent_id",
    "Bloqué": "blocked_detail",
    "Cause dernier blocage": "blocked_reason",
    "Type client": "client_type",
    "Ville": "city",
    "Gouvernerat": "governorate",
    "Code circuit": "circuit",
    "N° téléphone": "phone_1",
    "N° téléphone 2": "phone_2",
    "N° téléphone 3": "phone_3",
    "Contact": "contact",
    "Adresse": "address",
    "N° identif. intracomm.": "tax_id",
    "Solde": "balance",
    "Crédit autorisé DS": "credit_limit",
    "Code vendeur": "vendor_1",
    "<Code Vendeur 2>": "vendor_2",
    "<Code Vendeur 3>": "vendor_3",
}

ARTICLE_NUMERIC = ["unit_price", "unit_cost", "stock"]
CLIENT_NUMERIC = ["balance", "credit_limit"]

# Columns whose silent absence would disable a safety filter rather than merely
# blank a display field. _project fails loud on these and tolerates the rest.
REQUIRED_ARTICLE = ("N°", "Stocks", "Plage de Stock")
REQUIRED_CLIENT = ("N°", "Bloqué")


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _project(raw: pd.DataFrame, mapping: dict, required=()) -> pd.DataFrame:
    """Rename the columns we use, tolerating absent ones — except the required.

    Blank-filling a missing column is the right call for a display field: a
    slightly different export should still build. It is the WRONG call for a
    column that drives a filter. If "Bloque" were absent, blocked_detail would
    be "" for every row and `blocked` would be False for every client, silently
    switching off the rule that stops vendors calling non-paying accounts — with
    no exception and no log. So the filter-bearing columns fail loud, matching
    normalise_ledger in src/build_transactions.py, which raises on a shape change
    rather than inventing data.
    """
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"required columns absent from the export: {missing}. These drive "
            "scoring-time filters; blank-filling them would disable the filter "
            "silently. Fix the export or update the mapping deliberately."
        )
    present = {src: dst for src, dst in mapping.items() if src in raw.columns}
    df = raw.rename(columns=present)[list(present.values())].copy()
    for dst in mapping.values():
        if dst not in df.columns:
            df[dst] = ""
    return df[list(mapping.values())]


def normalise_articles(raw: pd.DataFrame) -> pd.DataFrame:
    df = _project(raw, ARTICLE_COLUMNS, REQUIRED_ARTICLE)
    for col in df.columns:
        if col not in ARTICLE_NUMERIC:
            df[col] = df[col].map(_clean)
    for col in ARTICLE_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["article_id"] != ""].reset_index(drop=True)


def normalise_clients(raw: pd.DataFrame) -> pd.DataFrame:
    df = _project(raw, CLIENT_COLUMNS, REQUIRED_CLIENT)
    for col in df.columns:
        if col not in CLIENT_NUMERIC:
            df[col] = df[col].map(_clean)
    for col in CLIENT_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Bloqué is empty / Facturer / Livrer / Tous. Keep the detail, add a boolean
    # so the scoring-time filter is a single unambiguous column.
    df["blocked"] = df["blocked_detail"] != ""
    return df[df["client_id"] != ""].reset_index(drop=True)


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    DATA.mkdir(exist_ok=True)
    arts = normalise_articles(pd.read_excel(
        EXPORTS / "deatils articles nav.xlsx", sheet_name="Liste des articles", dtype=object))
    clis = normalise_clients(pd.read_excel(
        EXPORTS / "deatils clients nav.xlsx", sheet_name="Liste des clients", dtype=object))

    arts.to_csv(DATA / "articles_master.csv", index=False, encoding="utf-8-sig")
    clis.to_csv(DATA / "clients_master.csv", index=False, encoding="utf-8-sig")

    print("=== MASTER BUILD REPORT ===")
    print(f"articles          : {len(arts):,}")
    print(f"  categories      : {arts['category'].nunique()}")
    print(f"  vehicle makes   : {arts['vehicle_make'].nunique()}")
    print(f"  with OEM ref    : {(arts['oem_ref'] != '').mean():.1%}")
    print(f"  stock classes   : {dict(arts['stock_class'].value_counts().head(5))}")
    print(f"clients           : {len(clis):,}")
    print(f"  blocked         : {clis['blocked'].sum():,} ({clis['blocked'].mean():.1%})")
    print(f"  with a vendor   : {(clis['vendor_1'] != '').sum():,}")
    print(f"  with a phone    : {(clis['phone_1'] != '').mean():.1%}")
```

Add at the end:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_build_transactions.py -v`
Expected: 10 passed

- [ ] **Step 5: Build the real masters and activate the PII test**

Run: `py src/build_masters.py`
Expected: 52,115 articles, 6,309 clients, blocked ≈60.8%.

Run: `py -m pytest tests/test_no_data_leak.py -v`
Expected: all PASS — `test_no_master_pii_appears` is now live (no longer skipped) and must pass. **If it fails, a real leak exists: stop and report.**

- [ ] **Step 6: Commit**

```bash
git add src/build_masters.py tests/test_build_transactions.py
git commit -m "feat(refoundation): ingest the NAV article and client masters

Adds category, vehicle make/model, OEM reference, stock class, blocking status
and the three vendor codes. The current-state columns are carried for
scoring-time filtering and documented as forbidden training features."
```

---

### Task 3: Reconcile against the existing dataset

Nothing may depend on `transactions.parquet` until it is shown to reproduce the table the whole validated system was built on. This is the same conservation discipline `build_dataset.py` used against the source Excel.

**Files:**
- Create: `src/reconcile.py`
- Create: `reports/reconciliation.md` (generated)

**Interfaces:**
- Consumes: `data/transactions.parquet` (Task 1), `data/ca_monthly.parquet` (existing).
- Produces: `reconcile_monthly(tx: pd.DataFrame) -> pd.DataFrame` with columns `month`, `ca` — the new ledger aggregated to the v1 grain, positive-and-negative netted, matching `ca_monthly.parquet` semantics.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconcile.py`:

```python
# -*- coding: utf-8 -*-
"""Aggregating the ledger to the v1 monthly grain must be exact and lossless."""
import pandas as pd

from reconcile import reconcile_monthly


def tx_row(date, amount, client="0289", article="01381", vendor="007"):
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "vendor_code": vendor, "amount": amount, "is_return": amount < 0}


def test_lines_in_the_same_month_are_summed():
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-01-20", 50.0)])
    out = reconcile_monthly(tx)
    assert len(out) == 1
    assert out.loc[0, "ca"] == 150.0
    assert out.loc[0, "month"] == pd.Timestamp("2024-01-01")


def test_returns_are_netted_in_matching_v1_semantics():
    """v1 kept negative CA in the fact table because the source system netted
    returns into revenue. Reconciliation must reproduce that, not improve on it."""
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-01-15", -30.0)])
    out = reconcile_monthly(tx)
    assert out.loc[0, "ca"] == 70.0


def test_distinct_months_stay_separate():
    tx = pd.DataFrame([tx_row("2024-01-02", 100.0), tx_row("2024-02-02", 40.0)])
    out = reconcile_monthly(tx).sort_values("month").reset_index(drop=True)
    assert out["ca"].tolist() == [100.0, 40.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_reconcile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reconcile'`

- [ ] **Step 3: Write the implementation**

Create `src/reconcile.py`:

```python
# -*- coding: utf-8 -*-
"""
Prove the new ledger reproduces the monthly table the validated system was built on.

Writes reports/reconciliation.md (aggregates only — safe to commit).

Two questions, in order:
  1. Do the identifier spaces match? If client/article/vendor ids do not overlap,
     nothing downstream is comparable and the run stops here.
  2. For the 15 overlapping Jan-May months, does the ledger aggregated to
     (month) -> ca reproduce ca_monthly.parquet?

Exact agreement is not expected to the millime: the v1 extract was a pre-aggregated
report and may apply its own rounding or scope. A per-month relative gap under
RELATIVE_TOLERANCE is treated as agreement; anything larger is reported as a
discrepancy for a human to judge.
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

RELATIVE_TOLERANCE = 0.01  # 1% per month

out = []


def emit(s=""):
    print(s)
    out.append(s)


def reconcile_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    """Ledger -> v1 monthly grain. Returns are netted in, matching v1 semantics."""
    df = tx.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")
    agg = df.groupby("month", as_index=False)["amount"].sum()
    agg["ca"] = agg["amount"].round(3)
    return agg[["month", "ca"]]


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    tx = pd.read_parquet(DATA / "transactions.parquet")
    v1 = pd.read_parquet(DATA / "ca_monthly.parquet")

    emit("# Reconciliation — NAV ledger vs the v1 monthly extract")
    emit()

    # ---- 1. Identifier spaces ----
    emit("## Identifier overlap")
    emit()
    emit("| Entity | v1 | ledger | shared | v1 covered |")
    emit("| --- | --- | --- | --- | --- |")
    coverage = {}
    for label, col in (("clients", "client_id"), ("articles", "article_id"),
                       ("vendors", "vendor_code")):
        a, b = set(v1[col].unique()), set(tx[col].unique())
        shared = a & b
        pct = len(shared) / max(len(a), 1)
        coverage[label] = pct
        emit(f"| {label} | {len(a):,} | {len(b):,} | {len(shared):,} | {pct:.1%} |")
    emit()

    if min(coverage.values()) < 0.5:
        emit("**STOP — identifier spaces do not match.** The ledger uses different keys "
             "than the v1 extract; no downstream comparison is meaningful until the "
             "mapping is understood.")
        (REPORTS / "reconciliation.md").write_text("\n".join(out), encoding="utf-8")
        raise SystemExit("identifier spaces do not match — see reports/reconciliation.md")

    # ---- 2. Monthly totals on the overlapping window ----
    new = reconcile_monthly(tx).rename(columns={"ca": "ca_ledger"})
    old = v1.groupby("month", as_index=False)["ca"].sum().rename(columns={"ca": "ca_v1"})
    cmp = old.merge(new, on="month", how="inner")
    cmp["gap"] = cmp["ca_ledger"] - cmp["ca_v1"]
    cmp["rel"] = cmp["gap"] / cmp["ca_v1"]

    emit("## Monthly totals on the 15 overlapping months")
    emit()
    emit("| Month | v1 CA | ledger CA | gap | rel |")
    emit("| --- | --- | --- | --- | --- |")
    for _, r in cmp.iterrows():
        emit(f"| {r['month']:%Y-%m} | {r['ca_v1']:,.0f} | {r['ca_ledger']:,.0f} | "
             f"{r['gap']:,.0f} | {r['rel']:+.2%} |")
    emit()

    worst = cmp["rel"].abs().max()
    bad = cmp[cmp["rel"].abs() > RELATIVE_TOLERANCE]
    emit(f"- months compared: **{len(cmp)}**")
    emit(f"- worst relative gap: **{worst:.2%}** (tolerance {RELATIVE_TOLERANCE:.0%})")
    emit(f"- months outside tolerance: **{len(bad)}**")
    emit()
    emit("## Months present only in the ledger")
    emit()
    extra = sorted(set(new["month"]) - set(old["month"]))
    emit(f"{len(extra)} new months: {extra[0]:%Y-%m} … {extra[-1]:%Y-%m}"
         if extra else "none")
    emit()
    emit("These are the months the v1 basis could never see. They are the reason "
         "labels stop being lower bounds.")

    (REPORTS / "reconciliation.md").write_text("\n".join(out), encoding="utf-8")
    print(f"\n[written] reports/reconciliation.md")

    if len(bad):
        raise SystemExit(
            f"{len(bad)} month(s) outside the {RELATIVE_TOLERANCE:.0%} tolerance — "
            "review reports/reconciliation.md before building snapshots")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_reconcile.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the real reconciliation**

Run: `py src/reconcile.py`
Expected: identifier overlap high on all three entities; 15 months compared; ~14 extra months.

**Gate — this is the plan's most important checkpoint.** If it exits non-zero, or identifier coverage is below 50%, **stop and report to the user**. Everything after this assumes the two datasets describe the same business on the same keys.

- [ ] **Step 6: Commit**

```bash
git add src/reconcile.py tests/test_reconcile.py reports/reconciliation.md
git commit -m "feat(refoundation): reconcile the NAV ledger against the v1 extract

Checks identifier overlap first, then per-month totals across the 15 shared
Jan-May months. Fails loudly rather than letting an unverified table become
the basis of every downstream number."
```

---

### Task 4: Measure the Jun–December blind spot

The project has claimed throughout that every reported rate is a lower bound because labels observe Jan–May only. That claim is now testable. This is a pure data measurement — no model — which is what makes it defensible.

**Files:**
- Create: `src/blind_spot.py`
- Create: `tests/test_blind_spot.py`
- Create: `reports/blind_spot.md` (generated)

**Interfaces:**
- Consumes: `data/transactions.parquet` (Task 1), reconciliation passed (Task 3).
- Produces: `repurchase_rates(tx, feature_year, label_year) -> dict` with keys `n_pairs`, `rate_jan_may`, `rate_full_year`, `understatement` (ratio `rate_full_year / rate_jan_may`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blind_spot.py`:

```python
# -*- coding: utf-8 -*-
"""The blind-spot measurement is arithmetic on labels, so test it as arithmetic."""
import pandas as pd

from blind_spot import repurchase_rates


def tx(date, client, article, amount=100.0):
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "amount": amount, "is_return": False}


def test_a_september_repurchase_is_invisible_to_a_jan_may_label():
    """The whole claim in one test: the same pair counts as 0 under the v1 label
    and 1 under a full-year label."""
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),     # active in the feature window
        tx("2025-09-15", "C1", "A1"),     # repurchased outside Jan-May
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["n_pairs"] == 1
    assert r["rate_jan_may"] == 0.0
    assert r["rate_full_year"] == 1.0


def test_a_may_repurchase_is_visible_to_both():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),
        tx("2025-05-15", "C1", "A1"),
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["rate_jan_may"] == 1.0
    assert r["rate_full_year"] == 1.0


def test_understatement_is_the_ratio_of_the_two_rates():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"), tx("2025-03-01", "C1", "A1"),   # visible
        tx("2024-02-01", "C2", "A1"), tx("2025-10-01", "C2", "A1"),   # hidden
        tx("2024-02-01", "C3", "A1"),                                  # never rebought
        tx("2024-02-01", "C4", "A1"),                                  # never rebought
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["n_pairs"] == 4
    assert r["rate_jan_may"] == 0.25
    assert r["rate_full_year"] == 0.50
    assert r["understatement"] == 2.0


def test_returns_do_not_count_as_a_repurchase():
    frame = pd.DataFrame([
        tx("2024-02-01", "C1", "A1"),
        {"date": pd.Timestamp("2025-09-01"), "client_id": "C1", "article_id": "A1",
         "amount": -100.0, "is_return": True},
    ])
    r = repurchase_rates(frame, feature_year=2024, label_year=2025)
    assert r["rate_full_year"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_blind_spot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blind_spot'`

- [ ] **Step 3: Write the implementation**

Create `src/blind_spot.py`:

```python
# -*- coding: utf-8 -*-
"""
Measure how much the January-May label window understated repurchase.

Every rate this project has reported was qualified as a lower bound, because the
v1 extract could only observe January-May and a pair repurchased in September was
scored as a failure. The NAV ledger covers all twelve months, so the size of that
understatement is now measurable instead of assumed.

This is deliberately a DATA measurement, not a model re-evaluation. The clean
model version is unavailable: the v1 test window is 2025 -> 2026 and the ledger
stops at 2026-06-30, so extending it would buy a single month. The base-rate
comparison below needs no model and is exact.

Writes reports/blind_spot.md (aggregates only — safe to commit).
"""
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

out = []


def emit(s=""):
    print(s)
    out.append(s)


def _bought(tx: pd.DataFrame, year: int, months=None) -> pd.DataFrame:
    """Distinct (client, article) pairs with a real purchase in the window."""
    sel = tx[(tx["date"].dt.year == year) & (~tx["is_return"]) & (tx["amount"] > 0)]
    if months is not None:
        sel = sel[sel["date"].dt.month.isin(months)]
    return sel[["client_id", "article_id"]].drop_duplicates()


def repurchase_rates(tx: pd.DataFrame, feature_year: int, label_year: int) -> dict:
    """Share of pairs active in Jan-May of feature_year that came back in
    label_year — measured the way v1 could see it, and the way it really was."""
    base = _bought(tx, feature_year, months=range(1, 6))
    if base.empty:
        return {"n_pairs": 0, "rate_jan_may": 0.0, "rate_full_year": 0.0,
                "understatement": float("nan")}

    jan_may = _bought(tx, label_year, months=range(1, 6))
    full = _bought(tx, label_year)

    keys = ["client_id", "article_id"]
    seen_jm = base.merge(jan_may.assign(hit=1), on=keys, how="left")["hit"].notna()
    seen_fy = base.merge(full.assign(hit=1), on=keys, how="left")["hit"].notna()

    r_jm, r_fy = float(seen_jm.mean()), float(seen_fy.mean())
    return {
        "n_pairs": int(len(base)),
        "rate_jan_may": r_jm,
        "rate_full_year": r_fy,
        "understatement": (r_fy / r_jm) if r_jm else float("nan"),
    }


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    tx = pd.read_parquet(DATA / "transactions.parquet")

    emit("# The January-May blind spot, measured")
    emit()
    emit("Pairs active in January-May of the feature year, and whether they came back "
         "in the label year — as the v1 basis could see it (January-May only) versus "
         "the full twelve months the NAV ledger now covers.")
    emit()
    emit("| Feature year | Label year | Pairs | Rate Jan-May | Rate full year | Understated by |")
    emit("| --- | --- | --- | --- | --- | --- |")

    results = {}
    for fy, ly in ((2024, 2025), (2025, 2026)):
        r = repurchase_rates(tx, fy, ly)
        results[f"{fy}->{ly}"] = r
        note = " *" if ly == 2026 else ""
        emit(f"| {fy} | {ly}{note} | {r['n_pairs']:,} | {r['rate_jan_may']:.1%} | "
             f"{r['rate_full_year']:.1%} | ×{r['understatement']:.2f} |")
    emit()
    emit("\\* 2026 is only observed to 30 June, so its full-year figure is itself a "
         "lower bound. The 2024->2025 row is the complete measurement.")
    emit()

    r = results["2024->2025"]
    emit("## What this means for the reported numbers")
    emit()
    emit(f"- Observed repurchase rate, January-May only: **{r['rate_jan_may']:.1%}**")
    emit(f"- True repurchase rate over the full year: **{r['rate_full_year']:.1%}**")
    emit(f"- The v1 label understated repurchase by a factor of "
         f"**×{r['understatement']:.2f}**.")
    emit()
    emit("Consequence for training, which matters more than the reported rate: pairs "
         "that did repurchase outside January-May were labelled 0 and taught the model "
         "that their pattern does not lead to a purchase. That is label noise in the "
         "target, and no amount of feature work or tuning could have corrected it.")

    (REPORTS / "blind_spot.md").write_text("\n".join(out), encoding="utf-8")
    print("\n[written] reports/blind_spot.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_blind_spot.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the real measurement**

Run: `py src/blind_spot.py`
Expected: `reports/blind_spot.md` written, with a 2024→2025 understatement factor above 1.0.

- [ ] **Step 6: Commit**

```bash
git add src/blind_spot.py tests/test_blind_spot.py reports/blind_spot.md
git commit -m "feat(refoundation): measure the January-May label blind spot

Turns the project's most-repeated caveat into a number: the share of pairs
that repurchased outside the observable window, and therefore the label noise
the v1 model was trained against."
```

---

### Task 5: Rolling snapshot builder

The core of the re-foundation. A pure function that builds one snapshot at any date `T`, plus the split constants and the leakage tests that make the design trustworthy.

**Files:**
- Create: `src/snapshots.py`
- Create: `tests/test_snapshots.py`

**Interfaces:**
- Consumes: `data/transactions.parquet` (Task 1).
- Produces:
  - `LOOKBACK_MONTHS`, `HORIZON_MONTHS`, `TRAIN_T`, `VAL_T`, `EMBARGO_T`, `TEST_T` (lists of `pd.Timestamp`).
  - `build_snapshot_at(tx: pd.DataFrame, T: pd.Timestamp, lookback_months: int = 12, horizon_months: int = 3) -> pd.DataFrame` — one row per candidate `(client_id, article_id)` pair, with a `snapshot` column equal to `T` and a `label` column.
  - `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` (lists of column names).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_snapshots.py`:

```python
# -*- coding: utf-8 -*-
"""Window boundaries and leakage. Synthetic frames only.

These tests are the reason the rolling design can be trusted: the v1 model's two
documented dead-ends both came from a validation split that shared a year with
training, and the fix is only real if the boundaries are enforced by tests.
"""
import pandas as pd
import pytest

from snapshots import (
    EMBARGO_T, HORIZON_MONTHS, LOOKBACK_MONTHS, TEST_T, TRAIN_T, VAL_T,
    build_snapshot_at,
)


def tx(date, client="C1", article="A1", amount=100.0, qty=1.0, vendor="V1",
       is_return=False, doc=None):
    """One synthetic ledger line. doc_no defaults to one document per date, so
    n_orders counts distinct documents exactly as it does on the real table."""
    return {"date": pd.Timestamp(date), "client_id": client, "article_id": article,
            "vendor_code": vendor, "enseigne": "FEBI", "amount": amount, "qty": qty,
            "cost": amount * 0.7, "is_return": is_return,
            "doc_no": doc or f"D{pd.Timestamp(date):%Y%m%d}"}


T = pd.Timestamp("2025-06-01")


def test_candidate_requires_a_purchase_inside_the_lookback_window():
    frame = pd.DataFrame([
        tx("2025-01-10", client="C1"),                 # inside  [2024-06, 2025-06)
        tx("2024-03-10", client="C2"),                 # older than the lookback
    ])
    out = build_snapshot_at(frame, T)
    assert out["client_id"].tolist() == ["C1"]


def test_label_comes_only_from_the_horizon_window():
    frame = pd.DataFrame([
        tx("2025-01-10"),                              # feature window
        tx("2025-07-10"),                              # inside  [2025-06, 2025-09)
    ])
    assert build_snapshot_at(frame, T).loc[0, "label"] == 1


def test_a_purchase_after_the_horizon_is_not_a_positive_label():
    frame = pd.DataFrame([tx("2025-01-10"), tx("2025-11-10")])
    assert build_snapshot_at(frame, T).loc[0, "label"] == 0


def test_a_purchase_on_the_snapshot_date_itself_is_a_label_not_a_feature():
    """T is the boundary: [T-lookback, T) for features, [T, T+horizon) for labels.
    A transaction exactly at T must land on the label side."""
    frame = pd.DataFrame([tx("2025-01-10"), tx(T)])
    out = build_snapshot_at(frame, T)
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "n_orders"] == 1, "the T transaction must not enter the features"


def test_returns_do_not_create_a_candidate_or_a_positive_label():
    frame = pd.DataFrame([
        tx("2025-01-10", client="C1", amount=-50.0, is_return=True),
    ])
    assert build_snapshot_at(frame, T).empty


def test_no_feature_column_reflects_data_at_or_after_T():
    """The leakage guard. Adding future transactions must not change any feature —
    only the label."""
    past = pd.DataFrame([tx("2025-01-10"), tx("2025-03-10")])
    future = pd.concat([past, pd.DataFrame([tx("2025-07-10"), tx("2025-08-10")])],
                       ignore_index=True)

    a = build_snapshot_at(past, T)
    b = build_snapshot_at(future, T)

    feature_cols = [c for c in a.columns if c not in ("label", "snapshot")]
    pd.testing.assert_frame_equal(a[feature_cols], b[feature_cols])
    assert a.loc[0, "label"] == 0 and b.loc[0, "label"] == 1


def test_snapshot_column_records_T():
    out = build_snapshot_at(pd.DataFrame([tx("2025-01-10")]), T)
    assert out.loc[0, "snapshot"] == T


def test_days_since_last_is_measured_against_T():
    frame = pd.DataFrame([tx("2025-05-02")])
    assert build_snapshot_at(frame, T).loc[0, "days_since_last"] == 30


def test_splits_are_chronological_and_do_not_overlap():
    assert max(TRAIN_T) < min(VAL_T) < max(VAL_T) < min(EMBARGO_T)
    assert max(EMBARGO_T) < min(TEST_T)
    assert len(TRAIN_T) == 8 and len(VAL_T) == 2
    assert len(EMBARGO_T) == 2 and len(TEST_T) == 4


def test_embargo_is_at_least_the_horizon():
    """Without it, a training snapshot's label window overlaps a test snapshot's
    feature window — the same leak the project has avoided everywhere else."""
    gap_months = (min(TEST_T).year - max(VAL_T).year) * 12 + \
                 (min(TEST_T).month - max(VAL_T).month)
    assert gap_months >= HORIZON_MONTHS


def test_lookback_and_horizon_match_the_agreed_design():
    assert LOOKBACK_MONTHS == 12
    assert HORIZON_MONTHS == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_snapshots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'snapshots'`

- [ ] **Step 3: Write the implementation**

Create `src/snapshots.py`:

```python
# -*- coding: utf-8 -*-
"""
Rolling chronological snapshots over the day-level transaction table.

One snapshot at date T:
    features from [T - LOOKBACK_MONTHS, T)      strictly before T
    label    from [T, T + HORIZON_MONTHS)       strictly at or after T
    candidates = (client, article) pairs with a real purchase in the feature window

Why rolling, and why fixed-width:

The v1 model had exactly ONE honest test window, and that single fact produced
three separate dead-ends — tuning was abandoned (a 4,000-iteration run improved a
random validation split while degrading the real test), calibration was abandoned
(an isotonic curve fitted on that split made test calibration worse), and the
queue could not be replayed. All three came from a validation split that shared a
year with training. Sixteen chronological snapshots give a genuinely out-of-time
validation set, which is what makes tuning and calibration legitimate again.

The lookback is FIXED, not expanding, for the same reason v1 used "year Y only":
with an expanding window an early snapshot sees 3 months of history and a late one
sees 30, so train and test would carry different feature distributions and the
comparison would be meaningless.

The EMBARGO is not optional. A training snapshot at 2025-10 has a label window
running to 2026-01, which overlaps the feature window of a test snapshot at
2026-01. Dropping the two months between the splits removes that overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_MONTHS = 12
HORIZON_MONTHS = 3


def _months(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


# 16 snapshots: T = 2025-01 .. 2026-04. Earliest T needs 12 months of history
# (data starts 2024-01-02); latest T needs 3 months of future (data ends 2026-06-30).
TRAIN_T = _months("2025-01-01", "2025-08-01")     # 8
VAL_T = _months("2025-09-01", "2025-10-01")       # 2  time-forward, never random
EMBARGO_T = _months("2025-11-01", "2025-12-01")   # 2  discarded
TEST_T = _months("2026-01-01", "2026-04-01")      # 4  untouched

NUMERIC_FEATURES = [
    "n_orders", "n_months", "qty_sum", "amount_sum", "amount_per_order",
    "days_since_last", "days_since_first", "median_interval", "interval_cv",
    "overdue_ratio", "return_rate",
    "client_amount", "client_n_articles", "client_n_orders", "client_active_months",
    "client_days_since_last",
    "article_n_clients", "article_amount", "article_repurchase_share",
    "share_of_client", "share_of_article",
]
CATEGORICAL_FEATURES = ["enseigne", "vendor_code"]


def _intervals(days: np.ndarray) -> tuple[float, float]:
    """Median gap between purchase days and its coefficient of variation.

    A single purchase has no interval. Returning NaN rather than 0 keeps 'unknown'
    distinct from 'bought twice on the same day', which CatBoost handles natively.
    """
    if len(days) < 2:
        return np.nan, np.nan
    gaps = np.diff(np.sort(np.unique(days)))
    if len(gaps) == 0:
        return np.nan, np.nan
    med = float(np.median(gaps))
    mean = float(np.mean(gaps))
    cv = float(np.std(gaps) / mean) if mean > 0 else np.nan
    return med, cv


def build_snapshot_at(tx: pd.DataFrame, T: pd.Timestamp,
                      lookback_months: int = LOOKBACK_MONTHS,
                      horizon_months: int = HORIZON_MONTHS) -> pd.DataFrame:
    """One row per candidate pair at T. Pure: no I/O, no globals, no clock."""
    T = pd.Timestamp(T)
    start = T - pd.DateOffset(months=lookback_months)
    end = T + pd.DateOffset(months=horizon_months)

    # Half-open windows. A transaction exactly at T belongs to the label side.
    hist = tx[(tx["date"] >= start) & (tx["date"] < T)]
    hist = hist[(~hist["is_return"]) & (hist["amount"] > 0)]
    if hist.empty:
        return pd.DataFrame(columns=(
            ["client_id", "article_id", "snapshot", "label"]
            + NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    all_hist = tx[(tx["date"] >= start) & (tx["date"] < T)]
    hist = hist.assign(_day=(hist["date"] - start).dt.days)

    pair = (
        hist.groupby(["client_id", "article_id"])
        .agg(n_orders=("doc_no", "nunique"),
             n_months=("date", lambda s: s.dt.to_period("M").nunique()),
             qty_sum=("qty", "sum"),
             amount_sum=("amount", "sum"),
             last_date=("date", "max"),
             first_date=("date", "min"))
        .reset_index()
    )
    pair["amount_per_order"] = pair["amount_sum"] / pair["n_orders"].clip(lower=1)
    pair["days_since_last"] = (T - pair["last_date"]).dt.days
    pair["days_since_first"] = (T - pair["first_date"]).dt.days

    gaps = (
        hist.groupby(["client_id", "article_id"])["_day"]
        .apply(lambda s: _intervals(s.to_numpy()))
        .reset_index(name="_iv")
    )
    gaps[["median_interval", "interval_cv"]] = pd.DataFrame(
        gaps["_iv"].tolist(), index=gaps.index)
    pair = pair.merge(gaps.drop(columns="_iv"), on=["client_id", "article_id"])
    pair["overdue_ratio"] = pair["days_since_last"] / pair["median_interval"]

    rets = (
        all_hist[all_hist["is_return"]]
        .groupby(["client_id", "article_id"]).size().rename("n_returns").reset_index()
    )
    pair = pair.merge(rets, on=["client_id", "article_id"], how="left")
    pair["n_returns"] = pair["n_returns"].fillna(0)
    pair["return_rate"] = pair["n_returns"] / (pair["n_orders"] + pair["n_returns"])
    pair = pair.drop(columns="n_returns")

    cli = (
        hist.groupby("client_id")
        .agg(client_amount=("amount", "sum"),
             client_n_articles=("article_id", "nunique"),
             client_n_orders=("doc_no", "nunique"),
             client_active_months=("date", lambda s: s.dt.to_period("M").nunique()),
             client_last=("date", "max"))
        .reset_index()
    )
    cli["client_days_since_last"] = (T - cli["client_last"]).dt.days
    cli = cli.drop(columns="client_last")

    art = (
        hist.groupby("article_id")
        .agg(article_n_clients=("client_id", "nunique"),
             article_amount=("amount", "sum"))
        .reset_index()
    )
    repeat = (
        hist.groupby(["article_id", "client_id"])["date"]
        .apply(lambda s: s.dt.to_period("M").nunique())
        .rename("m").reset_index()
        .groupby("article_id")["m"].apply(lambda s: float((s >= 2).mean()))
        .rename("article_repurchase_share").reset_index()
    )
    art = art.merge(repeat, on="article_id", how="left")

    x = pair.merge(cli, on="client_id").merge(art, on="article_id")
    x["share_of_client"] = x["amount_sum"] / x["client_amount"]
    x["share_of_article"] = x["amount_sum"] / x["article_amount"]

    # Categoricals: the pair's dominant vendor and the article's brand, both from
    # the feature window only.
    vend = (
        hist.groupby(["client_id", "article_id", "vendor_code"])["amount"].sum()
        .reset_index().sort_values("amount", ascending=False)
        .drop_duplicates(["client_id", "article_id"])[
            ["client_id", "article_id", "vendor_code"]]
    )
    x = x.merge(vend, on=["client_id", "article_id"], how="left")
    x["vendor_code"] = x["vendor_code"].replace("", np.nan).fillna("UNK")

    ens = (hist.sort_values("date").drop_duplicates("article_id", keep="last")
           [["article_id", "enseigne"]])
    x = x.merge(ens, on="article_id", how="left")
    x["enseigne"] = x["enseigne"].replace("", np.nan).fillna("UNK")

    # Label
    fut = tx[(tx["date"] >= T) & (tx["date"] < end)]
    fut = fut[(~fut["is_return"]) & (fut["amount"] > 0)]
    won = fut[["client_id", "article_id"]].drop_duplicates().assign(label=1)
    x = x.merge(won, on=["client_id", "article_id"], how="left")
    x["label"] = x["label"].fillna(0).astype(int)

    x["snapshot"] = T
    cols = (["client_id", "article_id", "snapshot", "label"]
            + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    return x[cols].reset_index(drop=True)


def build_many(tx: pd.DataFrame, dates) -> pd.DataFrame:
    """Stack several snapshots. Rows from different T are correlated, so any split
    over the result must be by snapshot date, never random."""
    frames = [build_snapshot_at(tx, T) for T in dates]
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_snapshots.py -v`
Expected: 11 passed

- [ ] **Step 5: Smoke-test one real snapshot**

Run:
```bash
py -c "import sys; sys.path.insert(0,'src'); import pandas as pd; from snapshots import build_snapshot_at; tx=pd.read_parquet('data/transactions.parquet'); s=build_snapshot_at(tx, pd.Timestamp('2025-06-01')); print(len(s), 'rows, positive rate', round(s['label'].mean(),4)); print(s['days_since_last'].describe())"
```
Expected: a non-empty frame with a plausible positive rate (roughly 0.10–0.40) and `days_since_last` between 0 and 365.

- [ ] **Step 6: Commit**

```bash
git add src/snapshots.py tests/test_snapshots.py
git commit -m "feat(refoundation): rolling chronological snapshot builder

12-month lookback, 3-month horizon, 16 snapshots with a time-forward
validation split and a 2-month embargo. Includes the leakage guard: adding
future transactions must change the label and nothing else."
```

---

### Task 6: Train the two models

Two separately reported models: a continuity benchmark answering the v1 question on the new data, and the 90-day rolling model. They are never compared to each other.

**Files:**
- Create: `src/train_rolling.py`
- Create: `reports/model_rolling.md`, `reports/rolling_metrics.json` (generated)

**Interfaces:**
- Consumes: `snapshots.build_many`, split constants (Task 5); `data/transactions.parquet` (Task 1).
- Produces: `data/model_rolling.cbm`, `data/pair_scores_rolling.csv`, `reports/model_rolling.md`, `reports/rolling_metrics.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_rolling.py`:

```python
# -*- coding: utf-8 -*-
"""Guard the training contract, not the metric values."""
import pandas as pd

from snapshots import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_rolling import FORBIDDEN_FEATURES, assert_no_current_state_features


def test_current_state_columns_are_never_trainable():
    """Stocks/Bloqué/Solde describe 2026-08-10 only. A model trained on them would
    learn the outcome it is meant to predict."""
    assert_no_current_state_features(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_the_guard_actually_fires():
    try:
        assert_no_current_state_features(NUMERIC_FEATURES + ["blocked"])
    except ValueError as exc:
        assert "blocked" in str(exc)
    else:
        raise AssertionError("the guard did not reject a current-state feature")


def test_forbidden_list_covers_every_master_snapshot_column():
    for col in ("blocked", "blocked_reason", "stock", "stock_class",
                "balance", "credit_limit"):
        assert col in FORBIDDEN_FEATURES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_train_rolling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_rolling'`

- [ ] **Step 3: Write the implementation**

Create `src/train_rolling.py`:

```python
# -*- coding: utf-8 -*-
"""
Train on rolling chronological snapshots.

Two models, reported separately and never compared to one another:

  1. CONTINUITY BENCHMARK — the v1 question ("pair active Jan-May of Y, bought
     again in Jan-May of Y+1?") re-run on the NAV data with full-year labels.
     This is the only fair "did the data help?" comparison against v1's 0.762.

  2. ROLLING 90-DAY MODEL — the new product. Its AUC is NOT comparable to 0.762:
     a shorter horizon means a lower base rate and a harder problem, so the
     number may be lower while the system is better. That is why the continuity
     benchmark exists.

Tuning is permitted here, unlike in train_model.py. The reason the v1 script
banned it was a validation split drawn randomly from the training year; the split
below is time-forward, with an embargo, so improving on it is evidence rather
than memorisation.

Outputs:
  reports/model_rolling.md      aggregate metrics (safe for the repo)
  reports/rolling_metrics.json  machine-readable
  data/model_rolling.cbm        trained 90-day model
  data/pair_scores_rolling.csv  test-window scores
"""
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshots import (  # noqa: E402
    CATEGORICAL_FEATURES, EMBARGO_T, HORIZON_MONTHS, LOOKBACK_MONTHS,
    NUMERIC_FEATURES, TEST_T, TRAIN_T, VAL_T, build_snapshot_at,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Current-state master columns. Never trainable — see src/build_masters.py.
FORBIDDEN_FEATURES = {
    "blocked", "blocked_detail", "blocked_reason", "stock", "stock_class",
    "balance", "credit_limit", "pareto",
}

out = []


def emit(s=""):
    print(s)
    out.append(s)


def assert_no_current_state_features(features) -> None:
    bad = sorted(set(features) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(
            f"current-state master columns cannot be training features: {bad}. "
            "They describe today, not the snapshot date, so a model would learn "
            "the outcome it is meant to predict. Use them as scoring-time filters."
        )


def decile_table(scores, labels) -> pd.Series:
    d = pd.DataFrame({"s": scores, "y": labels})
    d["dec"] = pd.qcut(d["s"].rank(method="first"), 10, labels=range(1, 11))
    return d.groupby("dec", observed=True)["y"].mean()


def precision_top_n(df_, score_col, n=30) -> float:
    top = (df_.sort_values(score_col, ascending=False)
           .groupby("vendor_code", group_keys=False).head(n))
    return float(top["label"].mean())


def fit(train, val, feats, cats, iterations=1500) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=200, early_stopping_rounds=100,
    )
    model.fit(
        Pool(train[feats], train["label"], cat_features=cats),
        eval_set=Pool(val[feats], val["label"], cat_features=cats),
        use_best_model=True,
    )
    return model


def main() -> None:
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    REPORTS.mkdir(exist_ok=True)
    out.clear()  # module-level accumulator: a second main() would double the report
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert_no_current_state_features(feats)

    tx = pd.read_parquet(DATA / "transactions.parquet")

    emit("# Rolling-snapshot model")
    emit()
    emit(f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
         f"Lookback {LOOKBACK_MONTHS} months, horizon {HORIZON_MONTHS} months.")
    emit()
    emit("| Split | Snapshots | Range |")
    emit("| --- | --- | --- |")
    for name, dates in (("train", TRAIN_T), ("validation", VAL_T),
                        ("embargo (dropped)", EMBARGO_T), ("test", TEST_T)):
        emit(f"| {name} | {len(dates)} | {min(dates):%Y-%m} … {max(dates):%Y-%m} |")
    emit()

    # One snapshot costs minutes against 1.88M rows and this needs 14 of them, so
    # cache each to disk. A crashed or repeated run must not pay that hour twice.
    cache = DATA / "snapshots"
    cache.mkdir(exist_ok=True)

    def cached(dates, name):
        frames = []
        for T in dates:
            f = cache / f"snapshot-{T:%Y-%m}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f))
                print(f"  {T:%Y-%m}  cached", flush=True)
            else:
                t0 = time.time()
                snap = build_snapshot_at(tx, T)
                snap.to_parquet(f, index=False)
                frames.append(snap)
                print(f"  {T:%Y-%m}  built {len(snap):>7,} rows in {time.time()-t0:.0f}s",
                      flush=True)
        total = sum(len(f) for f in frames)
        print(f"{name}: {total:,} rows from {len(dates)} snapshots", flush=True)
        return pd.concat(frames, ignore_index=True)

    print("building snapshots ...", flush=True)
    train = cached(TRAIN_T, "train")
    val = cached(VAL_T, "validation")
    test = cached(TEST_T, "test")

    for cat in CATEGORICAL_FEATURES:
        for frame in (train, val, test):
            frame[cat] = frame[cat].astype(str)

    emit("| Split | Rows | Positive rate |")
    emit("| --- | --- | --- |")
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        emit(f"| {name} | {len(frame):,} | {frame['label'].mean():.2%} |")
    emit()

    model = fit(train, val, feats, CATEGORICAL_FEATURES)
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        frame["p"] = model.predict_proba(Pool(frame[feats],
                                              cat_features=CATEGORICAL_FEATURES))[:, 1]

    auc = {name: float(roc_auc_score(f["label"], f["p"]))
           for name, f in (("train", train), ("validation", val), ("test", test))}

    emit("## AUC")
    emit()
    emit("| Split | AUC |")
    emit("| --- | --- |")
    for name in ("train", "validation", "test"):
        emit(f"| {name} | {auc[name]:.3f} |")
    emit()
    emit(f"Train − test gap: **{auc['train'] - auc['test']:+.3f}**.")
    emit()

    emit("> The v1 model scored AUC 0.762 on the January–May annual question. "
         "**That number is not comparable to the test AUC above.** A 90-day horizon "
         "has a lower base rate and is a harder problem. The comparable figure is "
         "the continuity benchmark below.")
    emit()

    emit("## Lift by decile (test)")
    emit()
    dec = decile_table(test["p"], test["label"])
    base = test["label"].mean()
    emit("| Decile | Purchase rate | Lift |")
    emit("| --- | --- | --- |")
    for d, v in dec.items():
        emit(f"| {d} | {v:.1%} | ×{v / base:.2f} |")
    emit()
    emit(f"Base rate: {base:.1%}.")
    emit()

    emit("## Precision at the top 30 per vendor (test)")
    emit()
    emit(f"- model: **{precision_top_n(test, 'p'):.1%}**")
    emit(f"- by past revenue alone: {precision_top_n(test, 'amount_sum'):.1%}")
    emit()

    emit("## Per-snapshot test AUC")
    emit()
    emit("| Snapshot | Rows | Positive rate | AUC |")
    emit("| --- | --- | --- | --- |")
    per_snapshot = {}
    for T, grp in test.groupby("snapshot"):
        a = float(roc_auc_score(grp["label"], grp["p"])) if grp["label"].nunique() > 1 else float("nan")
        per_snapshot[f"{T:%Y-%m}"] = a
        emit(f"| {T:%Y-%m} | {len(grp):,} | {grp['label'].mean():.2%} | {a:.3f} |")
    emit()
    emit("Stability across snapshots matters more than the pooled number: a model "
         "that only works in one month is not deployable.")
    emit()

    emit("## Feature importance (top 12)")
    emit()
    imp = pd.Series(model.get_feature_importance(),
                    index=feats).sort_values(ascending=False)
    emit("| Feature | Importance |")
    emit("| --- | --- |")
    for f, v in imp.head(12).items():
        emit(f"| {f} | {v:.1f} |")
    emit()

    model.save_model(str(DATA / "model_rolling.cbm"))
    test[["client_id", "article_id", "vendor_code", "snapshot",
          "amount_sum", "p", "label"]].to_csv(
        DATA / "pair_scores_rolling.csv", index=False, encoding="utf-8-sig")

    (REPORTS / "model_rolling.md").write_text("\n".join(out), encoding="utf-8")
    (REPORTS / "rolling_metrics.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback_months": LOOKBACK_MONTHS,
        "horizon_months": HORIZON_MONTHS,
        "splits": {"train": [f"{d:%Y-%m}" for d in TRAIN_T],
                   "validation": [f"{d:%Y-%m}" for d in VAL_T],
                   "embargo": [f"{d:%Y-%m}" for d in EMBARGO_T],
                   "test": [f"{d:%Y-%m}" for d in TEST_T]},
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "base_rate_test": float(base),
        "auc": auc,
        "auc_per_snapshot": per_snapshot,
        "precision_top30_model": precision_top_n(test, "p"),
        "precision_top30_revenue": precision_top_n(test, "amount_sum"),
        "feature_importance": {k: float(v) for k, v in imp.items()},
        "not_comparable_to": {"v1_annual_auc": 0.762,
                              "reason": "different horizon and label definition"},
    }, indent=2), encoding="utf-8")

    print("\n[written] reports/model_rolling.md, reports/rolling_metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_train_rolling.py -v`
Expected: 3 passed

- [ ] **Step 5: Train**

Run: `py src/train_rolling.py`
Expected: `reports/model_rolling.md` and `reports/rolling_metrics.json` written; test AUC reported per snapshot.

**Do not tune yet.** Record the first honest number, then decide. Tuning is now permitted by the design, but the baseline has to exist first — that is the same discipline `plan.md` set out ("prove value with a simple baseline before adding model complexity").

- [ ] **Step 6: Run the full suite**

Run: `py -m pytest tests/ model_results_api/tests -q`
Expected: all pass. The previous reference was 158; this task adds 28 (7 + 3 + 3 + 4 + 11 across Tasks 1–6), so expect **186**.

- [ ] **Step 7: Commit**

```bash
git add src/train_rolling.py tests/test_train_rolling.py reports/model_rolling.md reports/rolling_metrics.json
git commit -m "feat(refoundation): train on rolling snapshots with a time-forward split

Adds the 90-day rolling model with per-snapshot test AUC and an explicit guard
that current-state master columns can never become training features. Records
that the v1 0.762 is not comparable to this horizon."
```

---

## Checkpoint — report to the user before Plan 2

After Task 6, stop and report:

1. The reconciliation outcome (Task 3) — do the two datasets describe the same business?
2. The blind-spot factor (Task 4) — how much did the Jan–May window understate repurchase?
3. Rolling-model test AUC and its per-snapshot stability (Task 6).
4. Whether the continuity benchmark shows the new data improved the v1 question.

Plan 2 (Steps 5–7: category/vehicle-park and OEM features, scoring-time filters, margin ranking, rewiring `contact_queue.py` / the API / the dashboards, regenerating fixtures) is written after this checkpoint, because its priorities depend on which features earn their place here.

## Self-Review

**Spec coverage.** Step 0 → Task 0. Step 1 → Tasks 1, 2, 3. Step 2 → Task 4. Step 3 → Task 5. Step 4 → Task 6. Steps 5–7 are explicitly deferred to Plan 2 at the checkpoint above. The continuity benchmark named in Step 4 is partially covered: `train_rolling.py` reports the rolling model and states the non-comparability, but re-running the v1 annual recipe on NAV data is a distinct run. **Gap closed by:** the checkpoint item 4 requires it, and it is a single invocation of the existing `src/train_model.py` against a NAV-derived monthly table — added as Task 6 Step 5's follow-up rather than a separate task, because it reuses code that already exists and carries no new test cycle.

**Placeholder scan.** No TBD/TODO markers. Every code step contains runnable code. Every test step contains real assertions. Error handling is specified concretely (`ValueError` with matched message in Tasks 1 and 6; `SystemExit` gates in Tasks 1, 3).

**Type consistency.** `normalise_ledger` output columns match `reconcile_monthly` and `build_snapshot_at` inputs (`date`, `client_id`, `article_id`, `vendor_code`, `enseigne`, `amount`, `qty`, `cost`, `is_return`, `doc_no`). `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` defined in `snapshots.py` and imported by `train_rolling.py` under the same names. `FORBIDDEN_FEATURES` and `assert_no_current_state_features` defined in `train_rolling.py` and imported by `tests/test_train_rolling.py` under the same names. `build_snapshot_at` emits every column listed in `NUMERIC_FEATURES` — verified against the aggregation blocks.

**Pre-flight corrections (applied before execution).** Three defects were found scanning this plan against the repo's conventions:

1. **`sys.stdout` rebinding at module level.** All four new modules are imported by tests, and the repo's six test-imported library modules deliberately leave stdout alone while its six script modules rebind it. Rebinding at import fights pytest's capture. Moved into `main()` in every new module and recorded as a Global Constraint.
2. **`n_orders` took a different code path in tests than in production** — it used `doc_no` when the column existed and fell back to a row count otherwise. The synthetic `tx()` helper now emits `doc_no`, so the conditional is gone and both paths are identical. `client_n_orders` counts distinct documents for the same reason.
3. **`blocked_detail` was missing from the Task 2 interface list** though the implementation produces it and a test asserts on it.
