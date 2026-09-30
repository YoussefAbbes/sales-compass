# Prospection par article — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third Streamlit page where a vendor searches for an article and sees the five clients most worth calling about it, split into those who already bought it (model-scored) and those who never did (basket-similarity).

**Architecture:** All computation lives in `src/article_prospects.py`, which imports no Streamlit and is unit-tested on synthetic frames. `src/pages/2_Prospection.py` is a display layer only. This mirrors the existing `validation.py` / `1_Validation.py` split. Nothing existing is modified.

**Tech Stack:** Python 3.13, pandas, Streamlit (`AppTest` for page tests), pytest.

**Spec:** [docs/superpowers/specs/2026-08-04-article-prospects-design.md](../specs/2026-08-04-article-prospects-design.md)

## Global Constraints

- **Additive only.** No existing file is modified. `70 tests` pass before this work starts and must still pass after every task.
- **No Streamlit import in `src/article_prospects.py`** — it must be importable under pytest with no browser.
- **Ownership rule, both blocks:** a vendor's clients = clients with positive CA under that `vendor_code` in `ca_monthly.parquet`. Never `pair_scores_2026.csv`'s `vendor_code`, which is the pair's *dominant* vendor and disagrees for 78% of clients.
- **"Has bought article X" is evaluated across all vendors**, never restricted to the selected one.
- **Block 2 never displays a probability or percentage.** The model does not score those clients.
- **Wording:** *« propension au rachat »*, never *« chance qu'il achète »* — the probability is for an elapsed window (Jan–May 2026).
- **Signature thresholds:** `MIN_SHARE = 0.15`, `MIN_LIFT = 1.2`, `TOP_SIGNATURE = 30`, `MIN_BUYERS = 3`, `TOP_N = 5`. Module-level constants, matching the project's convention of naming business thresholds at the top of a file.
- `DATA` resolves through `DATA_ROOT` exactly as in `src/validation.py`.
- Run tests with `py -m pytest`, never bare `pytest`.

---

### Task 1: Module skeleton and article search

**Files:**
- Create: `src/article_prospects.py`
- Create: `tests/test_article_prospects.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `search_articles(articles: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame` returning columns `article_id, enseigne, designation, label`, where `label` is `"{article_id} {enseigne} — {designation}"`. Also `DATA: Path` and the module constants listed in Global Constraints.

- [ ] **Step 1: Write the failing test**

Create `tests/test_article_prospects.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for src/article_prospects.py. Synthetic frames only — no dependency on data/."""
import pandas as pd
import pytest

import article_prospects as ap


def make_articles():
    return pd.DataFrame([
        {"article_id": "X1", "enseigne": "MRQ01", "designation": "HUILE MOTEUR 5W40 5L"},
        {"article_id": "X2", "enseigne": "MRQ02", "designation": "HUILE ATF 1L"},
        {"article_id": "9931", "enseigne": "MRQ01", "designation": "PLAQUETTE FREIN AV"},
        {"article_id": "7712", "enseigne": "MRQ03", "designation": None},
    ])


def test_search_matches_designation_case_insensitively():
    out = ap.search_articles(make_articles(), "huile")
    assert set(out["article_id"]) == {"X1", "X2"}


def test_search_matches_reference_so_unnamed_articles_stay_reachable():
    out = ap.search_articles(make_articles(), "7712")
    assert list(out["article_id"]) == ["7712"]
    assert "(désignation inconnue)" in out.iloc[0]["label"]


def test_search_matches_brand():
    out = ap.search_articles(make_articles(), "mrq01")
    assert set(out["article_id"]) == {"X1", "9931"}


def test_search_respects_the_limit():
    out = ap.search_articles(make_articles(), "", limit=2)
    assert len(out) == 2


def test_blank_query_returns_the_expected_columns():
    out = ap.search_articles(make_articles(), "   ")
    assert list(out.columns) == ["article_id", "enseigne", "designation", "label"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'article_prospects'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/article_prospects.py`:

```python
# -*- coding: utf-8 -*-
"""Article-first prospection: given an article, which clients are worth calling?

The reverse of the call queue. Deliberately free of any Streamlit import so it can be
unit-tested without a browser — src/pages/2_Prospection.py is only a display layer.

Two populations, two signals, never mixed:
  block 1  clients who already bought the article -> the CatBoost model scores them
  block 2  clients who never did -> the model has NO pair features for them, so they
           are ranked by basket similarity instead, and carry no probability.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"

# ---- Business thresholds (to validate with stakeholders) ----
MIN_SHARE = 0.15        # an article is in the signature if >= 15% of buyers take it
MIN_LIFT = 1.2          # ... and it is 1.2x more common among buyers than catalogue-wide
TOP_SIGNATURE = 30      # signature articles kept, by lift
MIN_BUYERS = 3          # below this, a signature is noise rather than a pattern
TOP_N = 5               # clients shown per block

UNKNOWN_DESIGNATION = "(désignation inconnue)"


def search_articles(articles: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame:
    """Free-text search over reference, brand and designation.

    Reference is searched too: 15,638 of 40,321 articles carry no designation and would
    otherwise be unreachable.
    """
    out = articles.copy()
    out["designation"] = out["designation"].fillna(UNKNOWN_DESIGNATION)
    out["enseigne"] = out["enseigne"].fillna("")
    q = (query or "").strip().lower()
    if q:
        hay = (out["article_id"].str.lower() + " " + out["enseigne"].str.lower()
               + " " + out["designation"].str.lower())
        out = out[hay.str.contains(q, regex=False, na=False)]
    out["label"] = (out["article_id"] + " " + out["enseigne"]
                    + " — " + out["designation"]).str.replace("  ", " ", regex=False)
    return out[["article_id", "enseigne", "designation", "label"]].head(limit)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Stage the change**

```bash
git add src/article_prospects.py tests/test_article_prospects.py
```

---

### Task 2: Ownership primitives

**Files:**
- Modify: `src/article_prospects.py` (append)
- Modify: `tests/test_article_prospects.py` (append)

**Interfaces:**
- Consumes: the module from Task 1.
- Produces: `vendor_clients(fact: pd.DataFrame, vendor_code: str) -> set[str]` and `buyers_of(fact: pd.DataFrame, article_id: str) -> set[str]`. Both consider only rows with `ca > 0`. `buyers_of` ignores `vendor_code` entirely.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_article_prospects.py`:

```python
def make_fact():
    """Synthetic monthly-CA table.

    Articles : X (target), S1 S2 S3 (its signature), N (noise)
    Clients  : B1..B4 buy X under V1  |  P1 P2 P3 are V1 clients who never bought X
               P4 is a V1 client who buys X under ANOTHER vendor (V2)
               O1 is a V2-only client  |  F1..F6 buy only N
    """
    rows = []

    def add(vendor, client, article, ca=100.0, month="2025-03"):
        rows.append({"vendor_code": vendor, "client_id": client, "article_id": article,
                     "month": pd.Timestamp(month), "ca": ca})

    for c in ["B1", "B2", "B3", "B4"]:
        add("V1", c, "X")
    add("V2", "P4", "X")                       # P4 buys X elsewhere
    for c in ["B1", "B2", "B3", "B4", "P1", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S1")
    for c in ["B1", "B2", "B3", "P1", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S2")
    for c in ["B1", "B2", "P1", "P2", "O1", "P4"]:
        add("V1" if c != "O1" else "V2", c, "S3")
    for c in ["B1", "F1", "F2", "F3", "F4", "F5", "F6"]:
        add("V1", c, "N")
    add("V1", "P3", "Z")                       # V1 client with no signature overlap
    add("V1", "B1", "X", ca=-50.0)             # a return: must not create a buyer
    return pd.DataFrame(rows)


def test_vendor_clients_uses_the_fact_table_not_the_scores():
    v1 = ap.vendor_clients(make_fact(), "V1")
    assert {"B1", "B2", "B3", "B4", "P1", "P2", "P3", "P4"} <= v1
    assert "O1" not in v1


def test_buyers_of_ignores_vendor_so_a_client_buying_elsewhere_still_counts():
    buyers = ap.buyers_of(make_fact(), "X")
    assert buyers == {"B1", "B2", "B3", "B4", "P4"}


def test_negative_ca_alone_does_not_make_a_buyer():
    fact = pd.DataFrame([{"vendor_code": "V1", "client_id": "R1", "article_id": "X",
                          "month": pd.Timestamp("2025-03"), "ca": -80.0}])
    assert ap.buyers_of(fact, "X") == set()
    assert ap.vendor_clients(fact, "V1") == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_article_prospects.py -k "vendor_clients or buyers_of or negative_ca" -v`
Expected: FAIL — `AttributeError: module 'article_prospects' has no attribute 'vendor_clients'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/article_prospects.py`:

```python
def _positive(fact: pd.DataFrame) -> pd.DataFrame:
    """A purchase is a month with net CA > 0 — the project-wide definition.

    Returns are netted into CA by the source system, so a negative row is not a purchase
    and must not make someone a buyer.
    """
    return fact[fact["ca"] > 0]


def vendor_clients(fact: pd.DataFrame, vendor_code: str) -> set:
    """Clients this vendor has actually traded with.

    Ownership comes from the fact table, never from pair_scores_2026.csv: that file's
    vendor_code is the pair's DOMINANT vendor, and 78% of clients are served by several
    vendors, so the two disagree constantly.
    """
    pos = _positive(fact)
    return set(pos.loc[pos["vendor_code"] == vendor_code, "client_id"])


def buyers_of(fact: pd.DataFrame, article_id: str) -> set:
    """Clients who bought this article under ANY vendor.

    Vendor-blind on purpose: a client already buying the part through a colleague is not
    a prospect, and pitching it to them would expose that colleague's business.
    """
    pos = _positive(fact)
    return set(pos.loc[pos["article_id"] == article_id, "client_id"])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Stage the change**

```bash
git add src/article_prospects.py tests/test_article_prospects.py
```

---

### Task 3: Block 1 — clients who already bought the article

**Files:**
- Modify: `src/article_prospects.py` (append)
- Modify: `tests/test_article_prospects.py` (append)

**Interfaces:**
- Consumes: `vendor_clients` from Task 2.
- Produces: `existing_customers(scores: pd.DataFrame, fact: pd.DataFrame, article_id: str, vendor_code: str, top: int = TOP_N) -> pd.DataFrame` with columns `client_id, p_model, ca_sum, expected_value`, sorted by `expected_value` descending. Returns an empty frame **with those columns** when nothing matches.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_article_prospects.py`:

```python
def make_scores():
    """pair_scores_2026.csv shape. Note B4's vendor_code says V2 while the fact table
    says V1 owns B4 — the ownership rule must follow the fact table."""
    return pd.DataFrame([
        {"client_id": "B1", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 100.0, "p_model": 0.90},
        {"client_id": "B2", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 1000.0, "p_model": 0.50},
        {"client_id": "B3", "article_id": "X", "vendor_code": "V1",
         "ca_sum": 10.0, "p_model": 0.99},
        {"client_id": "B4", "article_id": "X", "vendor_code": "V2",
         "ca_sum": 50.0, "p_model": 0.20},
        {"client_id": "O1", "article_id": "X", "vendor_code": "V2",
         "ca_sum": 9999.0, "p_model": 0.99},
        {"client_id": "B1", "article_id": "S1", "vendor_code": "V1",
         "ca_sum": 700.0, "p_model": 0.80},
    ])


def test_block1_ranks_by_expected_value_not_by_probability():
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1")
    assert list(out["client_id"]) == ["B2", "B1", "B4", "B3"]
    assert out.iloc[0]["expected_value"] == pytest.approx(500.0)


def test_block1_ownership_follows_the_fact_table_not_the_scores_vendor():
    """B4's row says V2, but V1 trades with B4 — B4 belongs in V1's list."""
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1")
    assert "B4" in set(out["client_id"])
    assert "O1" not in set(out["client_id"])


def test_block1_is_restricted_to_the_requested_article():
    out = ap.existing_customers(make_scores(), make_fact(), "S1", "V1")
    assert list(out["client_id"]) == ["B1"]


def test_block1_returns_typed_empty_frame_when_no_client_matches():
    out = ap.existing_customers(make_scores(), make_fact(), "UNKNOWN", "V1")
    assert out.empty
    assert list(out.columns) == ["client_id", "p_model", "ca_sum", "expected_value"]


def test_block1_honours_the_top_limit():
    out = ap.existing_customers(make_scores(), make_fact(), "X", "V1", top=2)
    assert list(out["client_id"]) == ["B2", "B1"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_article_prospects.py -k block1 -v`
Expected: FAIL — `AttributeError: module 'article_prospects' has no attribute 'existing_customers'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/article_prospects.py`:

```python
BLOCK1_COLUMNS = ["client_id", "p_model", "ca_sum", "expected_value"]


def existing_customers(scores: pd.DataFrame, fact: pd.DataFrame, article_id: str,
                       vendor_code: str, top: int = TOP_N) -> pd.DataFrame:
    """Block 1 — this vendor's clients who already bought the article, model-ranked.

    Ranked by expected value (p x prior-year CA), the same rule as the call queue, so
    that "#1" means the same thing on every screen: the most money at stake, not the
    surest bet.
    """
    owned = vendor_clients(fact, vendor_code)
    sub = scores[(scores["article_id"] == article_id)
                 & (scores["client_id"].isin(owned))].copy()
    if sub.empty:
        return pd.DataFrame(columns=BLOCK1_COLUMNS)
    sub["expected_value"] = sub["p_model"] * sub["ca_sum"]
    return (sub.sort_values("expected_value", ascending=False)
               .head(top)[BLOCK1_COLUMNS]
               .reset_index(drop=True))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Stage the change**

```bash
git add src/article_prospects.py tests/test_article_prospects.py
```

---

### Task 4: The signature of an article

**Files:**
- Modify: `src/article_prospects.py` (append)
- Modify: `tests/test_article_prospects.py` (append)

**Interfaces:**
- Consumes: `_positive`, `buyers_of` from Task 2.
- Produces: `signature_articles(fact: pd.DataFrame, article_id: str, min_share: float = MIN_SHARE, min_lift: float = MIN_LIFT, top: int = TOP_SIGNATURE) -> list[str]`, ordered by lift descending. Returns `[]` when the article has fewer than `MIN_BUYERS` buyers. The target article is never in its own signature.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_article_prospects.py`:

```python
def test_signature_finds_the_over_represented_articles():
    sig = ap.signature_articles(make_fact(), "X")
    assert set(sig) == {"S1", "S2", "S3"}


def test_signature_excludes_the_article_itself():
    assert "X" not in ap.signature_articles(make_fact(), "X")


def test_signature_excludes_a_popular_but_unrelated_article():
    """N is bought by 7 clients but by only 1 of X's 5 buyers — lift below 1."""
    assert "N" not in ap.signature_articles(make_fact(), "X")


def test_signature_is_ordered_by_lift():
    sig = ap.signature_articles(make_fact(), "X")
    assert sig[0] == "S1"


def test_signature_is_empty_below_the_minimum_buyer_count():
    fact = make_fact()
    rare = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "B1", "article_id": "RARE",
         "month": pd.Timestamp("2025-03"), "ca": 10.0},
        {"vendor_code": "V1", "client_id": "B2", "article_id": "RARE",
         "month": pd.Timestamp("2025-03"), "ca": 10.0},
    ])
    assert ap.signature_articles(pd.concat([fact, rare]), "RARE") == []


def test_signature_honours_the_top_limit():
    assert len(ap.signature_articles(make_fact(), "X", top=2)) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_article_prospects.py -k signature -v`
Expected: FAIL — `AttributeError: module 'article_prospects' has no attribute 'signature_articles'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/article_prospects.py`:

```python
def signature_articles(fact: pd.DataFrame, article_id: str,
                       min_share: float = MIN_SHARE, min_lift: float = MIN_LIFT,
                       top: int = TOP_SIGNATURE) -> list:
    """Articles over-represented among the buyers of `article_id`.

    Lift, not raw co-occurrence: without dividing by the catalogue-wide rate, the answer
    is always the best-selling articles, which say nothing about this one in particular.
    Below MIN_BUYERS the ratio is arithmetic noise, so the signature is refused outright
    rather than returned weak — the page then says so instead of showing a bad list.
    """
    pos = _positive(fact)
    buyers = buyers_of(fact, article_id)
    if len(buyers) < MIN_BUYERS:
        return []

    pairs = pos[["client_id", "article_id"]].drop_duplicates()
    among = pairs[pairs["client_id"].isin(buyers)]
    share_buyers = among["article_id"].value_counts() / len(buyers)
    n_clients = pairs["client_id"].nunique()
    share_all = pairs["article_id"].value_counts() / n_clients

    lift = (share_buyers / share_all.reindex(share_buyers.index)).dropna()
    keep = lift[(share_buyers.reindex(lift.index) >= min_share) & (lift > min_lift)]
    keep = keep.drop(labels=[article_id], errors="ignore")
    return list(keep.sort_values(ascending=False).head(top).index)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: PASS, 19 tests.

- [ ] **Step 5: Stage the change**

```bash
git add src/article_prospects.py tests/test_article_prospects.py
```

---

### Task 5: Block 2 — prospects

**Files:**
- Modify: `src/article_prospects.py` (append)
- Modify: `tests/test_article_prospects.py` (append)

**Interfaces:**
- Consumes: `vendor_clients`, `buyers_of`, `signature_articles`.
- Produces: `prospects(fact: pd.DataFrame, article_id: str, vendor_code: str, top: int = TOP_N) -> tuple[pd.DataFrame, list[str]]` — the frame has columns `client_id, n_signature, client_ca` sorted by `n_signature` then `client_ca`, both descending; the list is the signature used, so the page can say "23 des 30". Clients with `n_signature == 0` are excluded.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_article_prospects.py`:

```python
def test_block2_ranks_prospects_by_signature_overlap():
    out, sig = ap.prospects(make_fact(), "X", "V1")
    assert len(sig) == 3
    assert list(out["client_id"]) == ["P1", "P2"]
    assert out.iloc[0]["n_signature"] == 3
    assert out.iloc[1]["n_signature"] == 1


def test_block2_excludes_clients_who_already_buy_it_under_another_vendor():
    """P4 is a V1 client and buys the whole signature — but buys X through V2."""
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "P4" not in set(out["client_id"])


def test_block2_excludes_clients_of_other_vendors():
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "O1" not in set(out["client_id"])


def test_block2_excludes_clients_with_no_signature_overlap():
    """P3 is a V1 client who has never bought any signature article."""
    out, _ = ap.prospects(make_fact(), "X", "V1")
    assert "P3" not in set(out["client_id"])


def test_block2_returns_typed_empty_frame_when_the_signature_is_refused():
    fact = make_fact()
    rare = pd.DataFrame([{"vendor_code": "V1", "client_id": "B1", "article_id": "RARE",
                          "month": pd.Timestamp("2025-03"), "ca": 10.0}])
    out, sig = ap.prospects(pd.concat([fact, rare]), "RARE", "V1")
    assert sig == []
    assert out.empty
    assert list(out.columns) == ["client_id", "n_signature", "client_ca"]


def test_block2_honours_the_top_limit():
    out, _ = ap.prospects(make_fact(), "X", "V1", top=1)
    assert list(out["client_id"]) == ["P1"]


def test_block2_is_empty_when_every_client_already_buys_the_article():
    """Signature exists, but this vendor has no non-buyer left to propose it to.

    V3's only client is B1, who already buys X — so `candidates` is empty while the
    signature is perfectly valid. The page must say "aucun prospect", not "trop peu
    d'acheteurs": the two messages have different meanings.
    """
    fact = pd.concat([make_fact(), pd.DataFrame([
        {"vendor_code": "V3", "client_id": "B1", "article_id": "S1",
         "month": pd.Timestamp("2025-04"), "ca": 25.0}])])
    out, sig = ap.prospects(fact, "X", "V3")
    assert sig != []
    assert out.empty
    assert list(out.columns) == ["client_id", "n_signature", "client_ca"]


def test_a_vendor_with_a_single_client_degrades_cleanly():
    """Vendor size runs from 1 to 944 clients; the smallest must not crash either block."""
    fact = pd.concat([make_fact(), pd.DataFrame([
        {"vendor_code": "V9", "client_id": "SOLO", "article_id": "S1",
         "month": pd.Timestamp("2025-03"), "ca": 40.0}])])
    out, sig = ap.prospects(fact, "X", "V9")
    assert sig != []
    assert list(out["client_id"]) == ["SOLO"]
    assert ap.existing_customers(make_scores(), fact, "X", "V9").empty
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_article_prospects.py -k block2 -v`
Expected: FAIL — `AttributeError: module 'article_prospects' has no attribute 'prospects'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/article_prospects.py`:

```python
BLOCK2_COLUMNS = ["client_id", "n_signature", "client_ca"]


def prospects(fact: pd.DataFrame, article_id: str, vendor_code: str,
              top: int = TOP_N) -> tuple:
    """Block 2 — this vendor's clients who never bought the article, by basket similarity.

    Returns (frame, signature) so the page can phrase the reason as "23 of the 30
    articles that buyers of this part usually take".

    No probability is produced, and none must be displayed: the model has no pair
    features for a couple that has never traded, so any number here would be invented.
    """
    signature = signature_articles(fact, article_id)
    if not signature:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), []

    owned = vendor_clients(fact, vendor_code)
    buyers = buyers_of(fact, article_id)
    candidates = owned - buyers
    if not candidates:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), signature

    pos = _positive(fact)
    held = (pos[pos["client_id"].isin(candidates) & pos["article_id"].isin(signature)]
            [["client_id", "article_id"]].drop_duplicates())
    if held.empty:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), signature

    out = held.groupby("client_id").size().rename("n_signature").reset_index()
    ca = pos.groupby("client_id")["ca"].sum().rename("client_ca")
    out = out.merge(ca, on="client_id", how="left")
    return (out.sort_values(["n_signature", "client_ca"], ascending=False)
               .head(top)[BLOCK2_COLUMNS]
               .reset_index(drop=True),
            signature)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_article_prospects.py -v`
Expected: PASS, 27 tests.

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `py -m pytest tests/ -q`
Expected: PASS, 97 tests (70 baseline + 27 new).

- [ ] **Step 6: Stage the change**

```bash
git add src/article_prospects.py tests/test_article_prospects.py
```

---

### Task 6: The Streamlit page

**Files:**
- Create: `src/pages/2_Prospection.py`

**Interfaces:**
- Consumes: every function from Tasks 1–5, plus `fmt_tnd` and `inject_css` from `ui_theme`.
- Produces: a page rendering the labels `"Chercher une pièce"` (text input), `"Vendeur"` (selectbox), `"Article"` (selectbox), and the two section headings `"1 · Ils l'achetaient"` and `"2 · À prospecter"`, which Task 7 asserts against.

- [ ] **Step 1: Write the page**

Create `src/pages/2_Prospection.py`:

```python
# -*- coding: utf-8 -*-
"""Prospection par article — l'inverse de la file d'appels.

Run:  streamlit run src/app.py   (cette page apparaît dans la navigation)

Le vendeur part de la pièce, pas du client : il a une promotion à pousser, un objectif
fournisseur, ou du stock à écouler. Deux blocs, deux signaux, jamais mélangés — le
modèle ne sait noter que les clients ayant déjà acheté l'article.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import article_prospects as ap
from ui_theme import fmt_tnd, inject_css

st.set_page_config(page_title="Prospection par article", page_icon="🔎", layout="wide")
inject_css()


@st.cache_data
def load_all():
    fact = pd.read_parquet(ap.DATA / "ca_monthly.parquet")
    articles = pd.read_csv(ap.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")
    clients = pd.read_csv(ap.DATA / "clients.csv", dtype=str, encoding="utf-8-sig")
    vendors = pd.read_csv(ap.DATA / "vendors.csv", dtype=str, encoding="utf-8-sig")
    scores_path = ap.DATA / "pair_scores_2026.csv"
    scores = (pd.read_csv(scores_path, dtype={"client_id": str, "article_id": str,
                                              "vendor_code": str}, encoding="utf-8-sig")
              if scores_path.exists() else
              pd.DataFrame(columns=["client_id", "article_id", "vendor_code",
                                    "ca_sum", "p_model"]))
    return fact, articles, clients, vendors, scores


try:
    fact, articles, clients, vendors, scores = load_all()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nLancez d'abord le pipeline (`python src/build_dataset.py`).")
    st.stop()

name_of = dict(zip(clients["client_id"], clients["client_name"]))

st.markdown(
    '<div class="masthead"><div class="eyebrow">Atelier — prospection</div>'
    '<h1>Qui appeler pour cette pièce ?</h1>'
    '<div class="sub">Choisissez un article : voici les clients de votre secteur qui '
    'l\'achetaient et ceux qui ne l\'ont jamais pris.</div></div>',
    unsafe_allow_html=True,
)

# ---------------- Sélection ----------------
vendor_codes = sorted(set(fact["vendor_code"]))
labels = dict(zip(vendors["vendor_code"], vendors["vendor_name"]))
vendor_code = st.sidebar.selectbox(
    "Vendeur", vendor_codes,
    format_func=lambda c: f"{labels.get(c, c)} ({c})")

query = st.text_input("Chercher une pièce", placeholder="huile atf, plaquette, 21345131…")
found = ap.search_articles(articles, query, limit=25)

if found.empty:
    st.info("Aucune pièce ne correspond. Essayez une référence, une marque, "
            "ou un mot de la désignation.")
    st.stop()

choice = st.selectbox("Article", found["label"].tolist())
article_id = found.loc[found["label"] == choice, "article_id"].iloc[0]

owned = ap.vendor_clients(fact, vendor_code)
buyers = ap.buyers_of(fact, article_id)
c1, c2 = st.columns(2)
c1.metric("Acheteurs, tous vendeurs", f"{len(buyers):,}".replace(",", " "))
c2.metric("Dont vos clients", f"{len(buyers & owned):,}".replace(",", " "))

# ---------------- Bloc 1 ----------------
st.markdown('<div class="section-label">1 · Ils l\'achetaient</div>',
            unsafe_allow_html=True)

block1 = ap.existing_customers(scores, fact, article_id, vendor_code)
if block1.empty:
    st.info("Aucun de vos clients n'a acheté cette pièce sur la période notée par le "
            "modèle. Regardez le bloc 2.")
else:
    st.caption("Classés par argent en jeu — propension au rachat × CA de l'an dernier. "
               "Le pourcentage est une **estimation optimiste** (voir la page Validation).")
    for r in block1.itertuples():
        st.markdown(
            f"**{name_of.get(r.client_id, r.client_id)}** · `{r.client_id}` — "
            f"**{r.p_model * 100:.0f} %** de propension au rachat · "
            f"{fmt_tnd(r.ca_sum)} l'an dernier")

# ---------------- Bloc 2 ----------------
st.markdown('<div class="section-label">2 · À prospecter</div>', unsafe_allow_html=True)

block2, signature = ap.prospects(fact, article_id, vendor_code)
if not signature:
    st.info(f"Trop peu d'acheteurs ({len(buyers)}) pour établir un profil fiable : "
            f"il en faut au moins {ap.MIN_BUYERS}.")
elif block2.empty:
    st.info("Aucun prospect : tous vos clients au profil proche achètent déjà cette pièce.")
else:
    st.caption(f"Clients qui n'ont **jamais** pris cette pièce, classés par ressemblance "
               f"avec le panier de ses acheteurs ({len(signature)} articles-repères).")
    for r in block2.itertuples():
        st.markdown(
            f"**{name_of.get(r.client_id, r.client_id)}** · `{r.client_id}` — "
            f"achète **{r.n_signature} des {len(signature)}** articles que prennent "
            f"habituellement les acheteurs de cette pièce, mais jamais celle-ci "
            f"· {fmt_tnd(r.client_ca)} de CA total")
    st.warning("Le modèle ne note pas ces clients : il n'a aucun historique sur ce "
               "couple client-article. Ce classement vient de la ressemblance des "
               "paniers, pas d'une probabilité.")
```

- [ ] **Step 2: Boot the app and confirm the page renders**

```bash
py -m streamlit run src/app.py --server.headless true --server.port 8512
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8512
```

Expected: `200`, and no traceback in the console. Stop the server afterwards.

- [ ] **Step 3: Stage the change**

```bash
git add src/pages/2_Prospection.py
```

---

### Task 7: AppTest coverage and final verification

**Files:**
- Create: `tests/test_prospection_page.py`

**Interfaces:**
- Consumes: the page from Task 6 and the `data_dir` / `project_root` fixtures already in `tests/conftest.py`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prospection_page.py`:

```python
# -*- coding: utf-8 -*-
"""AppTest coverage for the article prospection page."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def page_path():
    p = Path(__file__).resolve().parents[1] / "src" / "pages" / "2_Prospection.py"
    if not p.exists():
        pytest.skip("prospection page not built yet")
    return str(p)


@pytest.fixture
def at(page_path, data_dir):
    app = AppTest.from_file(page_path, default_timeout=300)
    app.run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    return app


def test_page_loads_with_both_blocks(at):
    text = " ".join(m.value for m in at.markdown)
    assert "Ils l'achetaient" in text
    assert "À prospecter" in text


def test_searching_narrows_the_article_list(at):
    box = [t for t in at.text_input if t.label == "Chercher une pièce"][0]
    before = len([sb for sb in at.selectbox if sb.label == "Article"][0].options)
    box.set_value("huile")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    after = [sb for sb in at.selectbox if sb.label == "Article"][0].options
    assert all("HUILE" in o.upper() for o in after)
    assert len(after) <= before


def test_block2_never_shows_a_probability(at):
    """The model cannot score these clients; a percentage there would be invented."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    assert "propension" not in tail
    assert "%" not in tail


def test_changing_vendor_keeps_the_page_working(at):
    box = [sb for sb in at.selectbox if sb.label == "Vendeur"][0]
    box.set_value(box.options[1])
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_unmatched_search_shows_guidance_not_a_crash(at):
    box = [t for t in at.text_input if t.label == "Chercher une pièce"][0]
    box.set_value("zzzzzznotanarticle")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    text = " ".join(i.value for i in at.info)
    assert "Aucune pièce ne correspond" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospection_page.py -v`
Expected: FAIL before Task 6 exists; after Task 6, this step confirms the labels match.

- [ ] **Step 3: Fix any label mismatch found**

If a test fails on a label (`"Chercher une pièce"`, `"Article"`, `"Vendeur"`), correct the **page** to match the spec's wording, not the test — the labels are part of the interface contract stated in Task 6's Interfaces block.

- [ ] **Step 4: Run the full suite**

Run: `py -m pytest tests/ -q`
Expected: PASS, **102 tests** (70 baseline + 27 module + 5 page).

- [ ] **Step 5: Confirm the untouched surfaces still work**

```bash
py -m pytest tests/test_app.py tests/test_validation_page.py tests/test_api.py -q
```

Expected: PASS. These cover the vendor queue, the validation page and the API — none of which this feature modifies. A failure here means the plan's "additive only" constraint was broken.

- [ ] **Step 6: Stage the change**

```bash
git add tests/test_prospection_page.py
```

---

## Verification checklist

- [ ] `py -m pytest tests/ -q` → 102 passed
- [ ] `src/article_prospects.py` contains no `import streamlit`
- [ ] No existing file was modified: `git status --short` shows only additions plus the two new test files
- [ ] The page boots inside a real Streamlit server (HTTP 200, clean logs)
- [ ] Block 2 renders no percentage anywhere
