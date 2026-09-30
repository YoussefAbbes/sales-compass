# Modèle de prospection par article — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace block 2's hand-tuned ranking heuristic with a CatBoost model trained on whether clients actually adopted articles they had never bought, measured on the untouched 2025→2026 window and required to beat the heuristic to ship.

**Architecture:** Three new modules. `src/prospect_features.py` builds features from one snapshot window; `src/prospect_model.py` handles sampling, metrics, scoring and the ranking entry point; `src/train_prospect_model.py` orchestrates training. Training and inference call the same feature builder, so train/serve skew is structurally impossible. `src/article_prospects.py` is **not modified** — the new entry point wraps it, which keeps the dependency one-directional and avoids a circular import.

**Tech Stack:** Python 3.13, pandas 2.2, numpy 2.2, scipy 1.15 (sparse), scikit-learn 1.6 (cosine), CatBoost 1.2, pytest. All already in `requirements.txt`.

**Spec:** [docs/superpowers/specs/2026-08-05-prospect-model-design.md](../specs/2026-08-05-prospect-model-design.md)

## Global Constraints

- **Run tests with `py -m pytest`, never bare `pytest`.** Bare `pytest` resolves to a different interpreter on this machine.
- **`src/article_prospects.py` and `src/train_model.py` are never modified.** 103 tests pass before this work starts and must still pass after every task.
- **No Streamlit import in `prospect_features.py` or `prospect_model.py`** — both must be importable under pytest with no browser.
- **No feature may reference the label window.** Every feature is computed from snapshot window Y only. This mirrors the symmetry `src/train_model.py` enforces because 2023 does not exist for the training split.
- **No candidate filter may reference the label window.** Training keeps every client active in Y. Filtering on "still active in Y+1" is not knowable at inference time and leaks the label.
- **Negatives are sampled within each article**, never across the whole grid.
- **Block 2 displays no probability.** The model changes the order of the top five, nothing else on screen.
- **Tests use synthetic frames only** — no dependency on `data/`, matching `tests/test_article_prospects.py`.
- `DATA` resolves through `DATA_ROOT` exactly as in `src/validation.py` and `src/article_prospects.py`.
- Business thresholds live as module-level constants at the top of the file, matching project convention.
- Model artifact `data/prospect_model.cbm` is **not committed** — `data/` is gitignored.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/prospect_features.py` | Feature engineering from one snapshot window. Knows nothing about labels, training or CatBoost. |
| `src/prospect_model.py` | Training-set sampling, evaluation metrics, model load/score, `rank_prospects()` entry point. |
| `src/train_prospect_model.py` | CLI script: build snapshots, fit, evaluate, write reports. |
| `tests/test_prospect_features.py` | Unit tests for the feature builder. |
| `tests/test_prospect_model.py` | Unit tests for sampling, metrics and the fallback path. |
| `src/pages/2_Prospection.py` | Modified in Task 10 only: calls `rank_prospects()` instead of `ap.prospects()`. |

Dependency direction, strictly one-way:

```text
train_prospect_model.py -> prospect_model.py -> prospect_features.py -> article_prospects.py
```

---

### Task 1: Module skeleton, family extraction and window slicing

**Files:**
- Create: `src/prospect_features.py`
- Create: `tests/test_prospect_features.py`

**Interfaces:**
- Consumes: `article_prospects` (for `MIN_BUYERS`, `MIN_SHARE`, `MIN_LIFT`, `TOP_SIGNATURE`).
- Produces: `family_of(designation) -> str`, `family_map(articles: pd.DataFrame) -> dict`, `window(fact: pd.DataFrame, year: int) -> pd.DataFrame`, and the constants `FAMILY_UNKNOWN`, `KNN_K`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prospect_features.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for src/prospect_features.py. Synthetic frames only — no dependency on data/."""
import numpy as np
import pandas as pd
import pytest

import prospect_features as pf


def make_articles():
    return pd.DataFrame([
        {"article_id": "X", "enseigne": "MRQ01", "designation": "PLAQUETTE FREIN AV"},
        {"article_id": "S1", "enseigne": "MRQ01", "designation": "DISQUE FREIN AV"},
        {"article_id": "S2", "enseigne": "MRQ02", "designation": "PLAQUETTE FREIN AR"},
        {"article_id": "S3", "enseigne": "MRQ02", "designation": "  filtre  huile  "},
        {"article_id": "N", "enseigne": "MRQ03", "designation": None},
    ])


def test_family_is_the_upper_cased_first_token():
    assert pf.family_of("PLAQUETTE FREIN AV") == "PLAQUETTE"
    assert pf.family_of("  filtre  huile  ") == "FILTRE"


def test_family_of_missing_designation_is_unknown():
    assert pf.family_of(None) == pf.FAMILY_UNKNOWN
    assert pf.family_of(np.nan) == pf.FAMILY_UNKNOWN
    assert pf.family_of("   ") == pf.FAMILY_UNKNOWN


def test_family_map_covers_every_reference_row():
    fam = pf.family_map(make_articles())
    assert fam["X"] == "PLAQUETTE"
    assert fam["N"] == pf.FAMILY_UNKNOWN
    assert len(fam) == 5


def test_window_keeps_only_positive_rows_of_that_year():
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "C1", "article_id": "X",
         "month": pd.Timestamp("2024-03"), "ca": 100.0},
        {"vendor_code": "V1", "client_id": "C1", "article_id": "X",
         "month": pd.Timestamp("2025-03"), "ca": 100.0},
        {"vendor_code": "V1", "client_id": "C2", "article_id": "X",
         "month": pd.Timestamp("2024-04"), "ca": -50.0},
    ])
    out = pf.window(fact, 2024)
    assert list(out["client_id"]) == ["C1"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prospect_features'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/prospect_features.py`:

```python
# -*- coding: utf-8 -*-
"""Features for the block-2 prospect model — one builder shared by training and inference.

The page and the training script call the SAME functions, so a feature can never be computed
one way when fitting and another way when serving. That failure (train/serve skew) is silent:
the model scores well offline and disappoints in the field.

No feature may look outside the snapshot window it is given. src/train_model.py enforces the
same rule, because 2023 does not exist for the training split and the two halves of the split
must see feature distributions built the same way.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_prospects as ap

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"

FAMILY_UNKNOWN = "INCONNU"
KNN_K = 20               # neighbours considered by the knn_buyers feature


def family_of(designation) -> str:
    """Product family = first token of the designation, upper-cased.

    Verified on the catalogue: 616 distinct first tokens, the top 200 covering 97% of named
    articles. Two tokens is far too sparse (3,260 values, 63% of them singletons). 38.6% of
    articles carry no designation at all, but they are only 8.2% of revenue.
    """
    if designation is None or (isinstance(designation, float) and np.isnan(designation)):
        return FAMILY_UNKNOWN
    tokens = str(designation).strip().upper().split()
    return tokens[0] if tokens else FAMILY_UNKNOWN


def family_map(articles: pd.DataFrame) -> dict:
    """article_id -> family, for the 61.4% of articles that appear in articles.csv.

    The rest resolve to FAMILY_UNKNOWN at lookup time via .get(), so callers never need to
    know which articles are named.
    """
    return {a: family_of(d)
            for a, d in zip(articles["article_id"], articles["designation"])}


def window(fact: pd.DataFrame, year: int) -> pd.DataFrame:
    """The positive rows of one Jan–May window.

    `ca > 0` is the project-wide definition of a purchase: returns are netted into CA by the
    source system, so a negative row is not a purchase.
    """
    pos = fact[fact["ca"] > 0]
    return pos[pos["month"].dt.year == year]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): add family extraction and window slicing"
```

---

### Task 2: Sparse basket matrix and client similarity

**Files:**
- Modify: `src/prospect_features.py` (append)
- Modify: `tests/test_prospect_features.py` (append)

**Interfaces:**
- Consumes: `window` from Task 1.
- Produces: `basket_matrix(win: pd.DataFrame) -> tuple[csr_matrix, pd.Index, pd.Index]` returning `(matrix, clients, articles)` where `matrix[i, j] == 1` if client `i` bought article `j`; and `client_similarity(matrix) -> np.ndarray`, a dense square cosine matrix with a zero diagonal.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_features.py`:

```python
def make_fact():
    """Synthetic two-window fact table: 2024 is the feature window, 2025 the label window.

    Articles : X (target), S1 S2 S3 (its signature), N (popular but unrelated)
    Clients  : B1..B4 bought X in 2024        -> already customers, not prospects
               P1 P2 P3 are V1 clients who never bought X in 2024
               P1 ADOPTS X in 2025 (the positive), P2 and P3 do not
               O1 is a V2-only client         -> out of V1's scope
               F1..F6 buy only N              -> the trap: popular article, no relation to X
    """
    rows = []

    def add(vendor, client, article, year, ca=100.0):
        rows.append({"vendor_code": vendor, "client_id": client, "article_id": article,
                     "month": pd.Timestamp(f"{year}-03"), "ca": ca})

    for c in ["B1", "B2", "B3", "B4"]:
        add("V1", c, "X", 2024)
    for c in ["B1", "B2", "B3", "B4", "P1", "P4"]:
        add("V1", c, "S1", 2024)
    add("V2", "O1", "S1", 2024)
    for c in ["B1", "B2", "B3", "P1", "P4"]:
        add("V1", c, "S2", 2024)
    for c in ["B1", "B2", "P1", "P2", "P4"]:
        add("V1", c, "S3", 2024)
    for c in ["B1", "F1", "F2", "F3", "F4", "F5", "F6"]:
        add("V1", c, "N", 2024)
    add("V1", "P3", "Z", 2024)
    add("V2", "P4", "X", 2024)              # P4 buys X through another vendor
    add("V1", "P1", "X", 2025)              # the adoption we want to predict
    add("V1", "B1", "X", 2025)              # a repeat, not an adoption
    return pd.DataFrame(rows)


def test_basket_matrix_marks_bought_pairs_only():
    win = pf.window(make_fact(), 2024)
    m, clients, articles = pf.basket_matrix(win)
    assert m.shape == (len(clients), len(articles))
    i, j = clients.get_loc("B1"), articles.get_loc("X")
    assert m[i, j] == 1
    assert m[clients.get_loc("P1"), j] == 0


def test_basket_matrix_is_binary_not_a_count():
    """Two months of the same pair must still read as one purchase."""
    fact = pd.DataFrame([
        {"vendor_code": "V1", "client_id": "C1", "article_id": "A",
         "month": pd.Timestamp("2024-03"), "ca": 10.0},
        {"vendor_code": "V1", "client_id": "C1", "article_id": "A",
         "month": pd.Timestamp("2024-04"), "ca": 10.0},
    ])
    m, _, _ = pf.basket_matrix(pf.window(fact, 2024))
    assert m[0, 0] == 1


def test_client_similarity_is_symmetric_with_zero_diagonal():
    win = pf.window(make_fact(), 2024)
    m, clients, _ = pf.basket_matrix(win)
    sim = pf.client_similarity(m)
    assert sim.shape == (len(clients), len(clients))
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 0.0)


def test_similar_baskets_score_above_unrelated_ones():
    win = pf.window(make_fact(), 2024)
    m, clients, _ = pf.basket_matrix(win)
    sim = pf.client_similarity(m)
    p1, b1, f1 = clients.get_loc("P1"), clients.get_loc("B1"), clients.get_loc("F1")
    assert sim[p1, b1] > sim[p1, f1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -k "basket or similarity or similar_baskets" -v`
Expected: FAIL — `AttributeError: module 'prospect_features' has no attribute 'basket_matrix'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_features.py` (and add the two imports at the top of the file, beside the existing ones):

```python
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
```

```python
def basket_matrix(win: pd.DataFrame):
    """Sparse binary client x article matrix for one window.

    Sparse because the dense form is ~1,600 x 40,000 floats (512 MB); as CSR it is a few MB.
    Binary rather than CA-weighted: the question is whether the client buys the part at all,
    and weighting by amount would let a handful of expensive lines dominate the similarity.

    Returns (matrix, clients, articles) — the two Index objects map ids to row/column numbers.
    """
    pairs = win[["client_id", "article_id"]].drop_duplicates()
    clients = pd.Index(sorted(pairs["client_id"].unique()))
    articles = pd.Index(sorted(pairs["article_id"].unique()))
    rows = clients.get_indexer(pairs["client_id"])
    cols = articles.get_indexer(pairs["article_id"])
    matrix = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (rows, cols)),
        shape=(len(clients), len(articles)))
    return matrix, clients, articles


def client_similarity(matrix) -> np.ndarray:
    """Dense pairwise cosine similarity between client baskets.

    Dense is affordable here and nowhere else: the result is clients x clients (~1,600^2,
    about 10 MB), not clients x articles. The diagonal is zeroed so a client is never its own
    nearest neighbour in knn_buyers.
    """
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    return sim
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): add sparse basket matrix and client similarity"
```

---

### Task 3: Client and article aggregate features

**Files:**
- Modify: `src/prospect_features.py` (append)
- Modify: `tests/test_prospect_features.py` (append)

**Interfaces:**
- Consumes: `window`, `family_map` from Task 1.
- Produces: `client_features(win, fam) -> pd.DataFrame` indexed by `client_id` with columns `CLIENT_COLUMNS`; `article_features(win, articles, fam) -> pd.DataFrame` indexed by `article_id` with columns `ARTICLE_COLUMNS`. Both constants are module-level lists.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_features.py`:

```python
def test_client_features_are_all_behavioural():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.client_features(win, fam)
    assert list(out.columns) == pf.CLIENT_COLUMNS
    b1 = out.loc["B1"]
    assert b1["client_n_articles"] == 5           # X S1 S2 S3 N
    assert b1["client_n_vendors"] == 1
    assert b1["client_ca"] == pytest.approx(500.0)
    assert b1["client_mean_ca_per_article"] == pytest.approx(100.0)


def test_client_features_count_distinct_families():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.client_features(win, fam)
    # B1 holds PLAQUETTE(X), DISQUE(S1), PLAQUETTE(S2), FILTRE(S3), INCONNU(N) -> 4 distinct
    assert out.loc["B1"]["client_n_families"] == 4


def test_article_features_carry_the_categoricals():
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.article_features(win, make_articles(), fam)
    assert list(out.columns) == pf.ARTICLE_COLUMNS
    assert out.loc["X"]["article_n_buyers"] == 5      # B1..B4 under V1, P4 under V2
    assert out.loc["X"]["enseigne"] == "MRQ01"
    assert out.loc["X"]["family"] == "PLAQUETTE"


def test_article_absent_from_the_reference_still_gets_features():
    """15,523 article ids in the fact table have no row in articles.csv at all."""
    win = pf.window(make_fact(), 2024)
    fam = pf.family_map(make_articles())
    out = pf.article_features(win, make_articles(), fam)
    assert out.loc["Z"]["family"] == pf.FAMILY_UNKNOWN
    assert out.loc["Z"]["enseigne"] == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -k "client_features or article_features or absent_from" -v`
Expected: FAIL — `AttributeError: module 'prospect_features' has no attribute 'client_features'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_features.py`:

```python
CLIENT_COLUMNS = ["client_ca", "client_n_articles", "client_n_months",
                  "client_n_vendors", "client_n_families", "client_mean_ca_per_article"]

ARTICLE_COLUMNS = ["article_n_buyers", "article_ca", "article_mean_ca_per_buyer",
                   "article_n_vendors", "enseigne", "family"]


def client_features(win: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """One row per client active in the window, indexed by client_id.

    Every feature is behavioural because clients.csv carries only a name — there is no sector,
    region or size attribute anywhere in the extract to use instead.
    """
    w = win.copy()
    w["family"] = w["article_id"].map(fam).fillna(FAMILY_UNKNOWN)
    g = w.groupby("client_id")
    out = pd.DataFrame({
        "client_ca": g["ca"].sum(),
        "client_n_articles": g["article_id"].nunique(),
        "client_n_months": g["month"].nunique(),
        "client_n_vendors": g["vendor_code"].nunique(),
        "client_n_families": g["family"].nunique(),
    })
    out["client_mean_ca_per_article"] = out["client_ca"] / out["client_n_articles"]
    return out[CLIENT_COLUMNS]


def article_features(win: pd.DataFrame, articles: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """One row per article sold in the window, indexed by article_id.

    `mean_ca_per_buyer` stands in for a price band: the extract carries no unit price, and the
    monthly CA of a part divided by its buyers is the closest available proxy.
    """
    g = win.groupby("article_id")
    out = pd.DataFrame({
        "article_n_buyers": g["client_id"].nunique(),
        "article_ca": g["ca"].sum(),
        "article_n_vendors": g["vendor_code"].nunique(),
    })
    out["article_mean_ca_per_buyer"] = out["article_ca"] / out["article_n_buyers"]
    ens = dict(zip(articles["article_id"], articles["enseigne"].fillna("")))
    ids = pd.Series(out.index, index=out.index)
    out["enseigne"] = ids.map(ens).fillna("")
    out["family"] = ids.map(fam).fillna(FAMILY_UNKNOWN)
    return out[ARTICLE_COLUMNS]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): add client and article aggregate features"
```

---

### Task 4: The window context

**Files:**
- Modify: `src/prospect_features.py` (append)
- Modify: `tests/test_prospect_features.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `WindowContext` (a dataclass with fields `win`, `fam`, `clients`, `articles`, `matrix`, `similarity`, `client_feats`, `article_feats`) and `build_context(fact, articles_ref, year) -> WindowContext`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_features.py`:

```python
def test_context_carries_every_precomputed_piece():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    assert len(ctx.win) > 0
    assert ctx.matrix.shape == (len(ctx.clients), len(ctx.articles))
    assert ctx.similarity.shape == (len(ctx.clients), len(ctx.clients))
    assert list(ctx.client_feats.columns) == pf.CLIENT_COLUMNS
    assert list(ctx.article_feats.columns) == pf.ARTICLE_COLUMNS


def test_context_holds_only_the_requested_window():
    """The 2025 rows must be invisible to a 2024 context — that is the leakage guarantee."""
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    assert ctx.win["month"].dt.year.unique().tolist() == [2024]
    assert "P1" in ctx.clients
    assert ctx.matrix[ctx.clients.get_loc("P1"), ctx.articles.get_loc("X")] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -k context -v`
Expected: FAIL — `AttributeError: module 'prospect_features' has no attribute 'build_context'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_features.py` (and add `from dataclasses import dataclass` to the imports):

```python
@dataclass
class WindowContext:
    """Everything derived from one snapshot window, computed once and reused.

    Training loops over thousands of articles and the page re-runs on every interaction; both
    would otherwise rebuild the basket matrix and the similarity grid per article. Holding them
    here makes the expensive work happen once per window instead.
    """
    win: pd.DataFrame
    fam: dict
    clients: pd.Index
    articles: pd.Index
    matrix: object            # scipy.sparse.csr_matrix
    similarity: np.ndarray
    client_feats: pd.DataFrame
    article_feats: pd.DataFrame


def build_context(fact: pd.DataFrame, articles_ref: pd.DataFrame, year: int) -> WindowContext:
    """Slice one window and precompute every window-level structure.

    Nothing outside `year` is read. A context built for the training window physically cannot
    see the label window, which is what makes the leakage test in Task 6 meaningful rather
    than aspirational.
    """
    win = window(fact, year)
    fam = family_map(articles_ref)
    matrix, clients, articles = basket_matrix(win)
    return WindowContext(
        win=win,
        fam=fam,
        clients=clients,
        articles=articles,
        matrix=matrix,
        similarity=client_similarity(matrix),
        client_feats=client_features(win, fam),
        article_feats=article_features(win, articles_ref, fam),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): add the reusable window context"
```

---

### Task 5: Signature with lift, and the affinity features

**Files:**
- Modify: `src/prospect_features.py` (append)
- Modify: `tests/test_prospect_features.py` (append)

**Interfaces:**
- Consumes: `WindowContext` from Task 4, `article_prospects.buyers_of`.
- Produces: `signature_with_lift(win, article_id, ...) -> pd.Series` (index `article_id`, value lift, sorted descending); `affinity_features(ctx, article_id, client_ids, sig) -> pd.DataFrame` indexed by `client_id` with columns `AFFINITY_COLUMNS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_features.py`:

```python
def test_signature_with_lift_returns_values_not_just_ids():
    win = pf.window(make_fact(), 2024)
    sig = pf.signature_with_lift(win, "X")
    assert set(sig.index) == {"S1", "S2", "S3"}
    assert sig.iloc[0] == sig.max()                    # sorted descending
    assert sig["S1"] > sig["S3"]
    assert "N" not in sig.index                        # popular but lift < 1
    assert "X" not in sig.index                        # never its own signature


def test_signature_is_refused_below_the_minimum_buyer_count():
    win = pf.window(make_fact(), 2024)
    assert pf.signature_with_lift(win, "Z").empty


def test_affinity_counts_signature_overlap_and_weights_it_by_lift():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "P2", "P3"], sig)
    assert list(out.columns) == pf.AFFINITY_COLUMNS
    assert out.loc["P1"]["n_signature_held"] == 3
    assert out.loc["P2"]["n_signature_held"] == 1
    assert out.loc["P3"]["n_signature_held"] == 0
    assert out.loc["P1"]["share_signature_held"] == pytest.approx(1.0)
    assert out.loc["P1"]["lift_weighted_overlap"] > out.loc["P2"]["lift_weighted_overlap"]


def test_affinity_measures_brand_and_family_holdings():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "P3"], sig)
    # X is MRQ01/PLAQUETTE. P1 holds S1 (MRQ01) -> brand_n_held 1; S2 is PLAQUETTE -> family 1.
    assert out.loc["P1"]["brand_n_held"] == 1
    assert out.loc["P1"]["family_n_held"] == 1
    assert out.loc["P3"]["brand_n_held"] == 0
    assert out.loc["P3"]["family_n_held"] == 0


def test_affinity_knn_counts_buyers_among_nearest_neighbours():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["P1", "F1"], sig, k=3)
    assert out.loc["P1"]["knn_buyers"] > out.loc["F1"]["knn_buyers"]


def test_affinity_returns_a_typed_frame_for_clients_absent_from_the_window():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    sig = pf.signature_with_lift(ctx.win, "X")
    out = pf.affinity_features(ctx, "X", ["GHOST"], sig)
    assert list(out.columns) == pf.AFFINITY_COLUMNS
    assert out.loc["GHOST"]["n_signature_held"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -k "signature_with_lift or affinity" -v`
Expected: FAIL — `AttributeError: module 'prospect_features' has no attribute 'signature_with_lift'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_features.py`:

```python
AFFINITY_COLUMNS = ["n_signature_held", "share_signature_held", "lift_weighted_overlap",
                    "brand_n_held", "brand_ca_share", "family_n_held", "family_ca_share",
                    "cosine_basket_sim", "knn_buyers"]


def signature_with_lift(win: pd.DataFrame, article_id: str,
                        min_share: float = ap.MIN_SHARE, min_lift: float = ap.MIN_LIFT,
                        top: int = ap.TOP_SIGNATURE) -> pd.Series:
    """Signature articles WITH their lift, rather than just their ids.

    article_prospects.signature_articles() discards the lift values; the model wants them, so
    that a client holding three strongly-associated articles outranks one holding three
    marginal ones. Same thresholds, same ordering — only the return type differs.
    """
    buyers = ap.buyers_of(win, article_id)
    if len(buyers) < ap.MIN_BUYERS:
        return pd.Series(dtype=float)

    pairs = win[["client_id", "article_id"]].drop_duplicates()
    among = pairs[pairs["client_id"].isin(buyers)]
    share_buyers = among["article_id"].value_counts() / len(buyers)
    share_all = pairs["article_id"].value_counts() / pairs["client_id"].nunique()

    lift = (share_buyers / share_all.reindex(share_buyers.index)).dropna()
    keep = lift[(share_buyers.reindex(lift.index) >= min_share) & (lift > min_lift)]
    keep = keep.drop(labels=[article_id], errors="ignore")
    return keep.sort_values(ascending=False).head(top)


def affinity_features(ctx: WindowContext, article_id: str, client_ids,
                      sig: pd.Series, k: int = KNN_K) -> pd.DataFrame:
    """Pair features that need NO history for the pair itself — the heart of the model.

    Every one of these is computable for a client who has never bought the article, which is
    exactly the population the CatBoost pair model cannot score. `n_signature_held` is today's
    entire block-2 ranking; here it is one input among nine, with its weight learned.
    """
    client_ids = list(client_ids)
    out = pd.DataFrame(index=pd.Index(client_ids, name="client_id"),
                       columns=AFFINITY_COLUMNS, dtype=float).fillna(0.0)

    held = (ctx.win[ctx.win["client_id"].isin(client_ids)]
            [["client_id", "article_id", "ca"]])
    sig_set = set(sig.index)

    # ---- signature overlap, plain and lift-weighted ----
    if sig_set:
        ov = held[held["article_id"].isin(sig_set)][["client_id", "article_id"]].drop_duplicates()
        counts = ov.groupby("client_id").size()
        weights = ov.assign(w=ov["article_id"].map(sig)).groupby("client_id")["w"].sum()
        out["n_signature_held"] = counts.reindex(client_ids).fillna(0.0)
        out["share_signature_held"] = out["n_signature_held"] / len(sig_set)
        out["lift_weighted_overlap"] = weights.reindex(client_ids).fillna(0.0)

    # ---- brand and family holdings ----
    target_brand = ctx.article_feats["enseigne"].get(article_id, "")
    target_family = ctx.fam.get(article_id, FAMILY_UNKNOWN)
    h = held[held["article_id"] != article_id].copy()
    h["enseigne"] = h["article_id"].map(ctx.article_feats["enseigne"]).fillna("")
    h["family"] = h["article_id"].map(ctx.fam).fillna(FAMILY_UNKNOWN)
    total_ca = h.groupby("client_id")["ca"].sum()

    for label, col, value in [("enseigne", "brand", target_brand),
                              ("family", "family", target_family)]:
        if label == "enseigne" and not value:
            continue
        sub = h[h[label] == value]
        n = sub.groupby("client_id")["article_id"].nunique()
        ca = sub.groupby("client_id")["ca"].sum()
        out[f"{col}_n_held"] = n.reindex(client_ids).fillna(0.0)
        out[f"{col}_ca_share"] = (ca / total_ca).reindex(client_ids).fillna(0.0)

    # ---- basket cosine against the article's buyer profile, and knn ----
    buyers = ap.buyers_of(ctx.win, article_id)
    known = [c for c in client_ids if c in ctx.clients]
    if buyers and known:
        buyer_rows = ctx.clients.get_indexer([b for b in buyers if b in ctx.clients])
        buyer_rows = buyer_rows[buyer_rows >= 0]
        if len(buyer_rows):
            profile = np.asarray(ctx.matrix[buyer_rows].mean(axis=0))
            rows = ctx.clients.get_indexer(known)
            cos = cosine_similarity(ctx.matrix[rows], profile).ravel()
            out.loc[known, "cosine_basket_sim"] = cos

            is_buyer = np.zeros(len(ctx.clients), dtype=bool)
            is_buyer[buyer_rows] = True
            for cid, row in zip(known, rows):
                nearest = np.argpartition(-ctx.similarity[row], min(k, len(ctx.clients) - 1))[:k]
                out.loc[cid, "knn_buyers"] = float(is_buyer[nearest].sum())

    return out[AFFINITY_COLUMNS]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): add lift-weighted signature and affinity features"
```

---

### Task 6: Assemble the feature frame, and prove it cannot leak

**Files:**
- Modify: `src/prospect_features.py` (append)
- Modify: `tests/test_prospect_features.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `candidates_for(ctx, article_id) -> list[str]` (clients active in the window who do **not** hold the article, sorted); `build_features(ctx, article_id, client_ids=None) -> pd.DataFrame` indexed by `client_id` with columns `FEATURE_COLUMNS`; the constants `FEATURE_COLUMNS` and `CATEGORICAL_COLUMNS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_features.py`:

```python
def test_candidates_exclude_everyone_who_already_holds_the_article():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    cands = pf.candidates_for(ctx, "X")
    assert "P1" in cands and "P2" in cands and "P3" in cands
    for holder in ["B1", "B2", "B3", "B4", "P4"]:
        assert holder not in cands


def test_build_features_returns_every_column_in_order():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X")
    assert list(out.columns) == pf.FEATURE_COLUMNS
    assert set(out.index) == set(pf.candidates_for(ctx, "X"))
    assert out.notna().all().all()


def test_build_features_categoricals_are_strings():
    """CatBoost rejects a categorical column that arrives as float or NaN."""
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X")
    for col in pf.CATEGORICAL_COLUMNS:
        assert out[col].map(type).eq(str).all()


def test_build_features_accepts_an_explicit_client_list():
    ctx = pf.build_context(make_fact(), make_articles(), 2024)
    out = pf.build_features(ctx, "X", client_ids=["P1"])
    assert list(out.index) == ["P1"]


def test_no_feature_can_see_the_label_window():
    """The leakage guard: a 2024 frame must be identical whether or not 2025 exists.

    If any feature reached into the label window, dropping those rows would change it — which
    is precisely the bug that produces a beautiful offline score and a useless model.
    """
    fact = make_fact()
    ctx_with = pf.build_context(fact, make_articles(), 2024)
    ctx_without = pf.build_context(fact[fact["month"].dt.year == 2024],
                                   make_articles(), 2024)
    a = pf.build_features(ctx_with, "X")
    b = pf.build_features(ctx_without, "X")
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_features.py -k "candidates or build_features or label_window" -v`
Expected: FAIL — `AttributeError: module 'prospect_features' has no attribute 'candidates_for'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_features.py`:

```python
FEATURE_COLUMNS = CLIENT_COLUMNS + ARTICLE_COLUMNS + AFFINITY_COLUMNS
CATEGORICAL_COLUMNS = ["enseigne", "family"]


def candidates_for(ctx: WindowContext, article_id: str) -> list:
    """Clients active in the window who do NOT already hold the article.

    Vendor scoping is deliberately absent: the model scores a client-article couple, and the
    vendor filter is applied afterwards at display time. Training on vendor-filtered candidates
    would make the model learn each vendor's book rather than the article's appeal.
    """
    holders = ap.buyers_of(ctx.win, article_id)
    return sorted(set(ctx.clients) - holders)


def build_features(ctx: WindowContext, article_id: str, client_ids=None) -> pd.DataFrame:
    """The single feature builder, called identically by training and by the page.

    Article-level values are broadcast across the rows: CatBoost sees them as constants within
    one article's group, which is exactly the comparison the page asks for — rank these clients
    against each other for THIS part.
    """
    if client_ids is None:
        client_ids = candidates_for(ctx, article_id)
    client_ids = list(client_ids)
    if not client_ids:
        return pd.DataFrame(columns=FEATURE_COLUMNS,
                            index=pd.Index([], name="client_id"))

    sig = signature_with_lift(ctx.win, article_id)
    out = ctx.client_feats.reindex(client_ids).fillna(0.0)
    out.index.name = "client_id"
    out = out.join(affinity_features(ctx, article_id, client_ids, sig))

    art = ctx.article_feats.reindex([article_id]).iloc[0]
    for col in ARTICLE_COLUMNS:
        value = art[col]
        if col in CATEGORICAL_COLUMNS:
            out[col] = FAMILY_UNKNOWN if pd.isna(value) else str(value)
        else:
            out[col] = 0.0 if pd.isna(value) else float(value)

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    return out[FEATURE_COLUMNS]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_features.py -v`
Expected: PASS, 25 tests.

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `py -m pytest tests/ -q`
Expected: PASS, 128 tests (103 baseline + 25 new).

- [ ] **Step 6: Commit**

```bash
git add src/prospect_features.py tests/test_prospect_features.py
git commit -m "feat(prospect-model): assemble the feature frame with a leakage guard"
```

---

### Task 7: Within-article negative sampling

**Files:**
- Create: `src/prospect_model.py`
- Create: `tests/test_prospect_model.py`

**Interfaces:**
- Consumes: `prospect_features` (Tasks 1–6).
- Produces: `adopters(win_next, article_id, candidates) -> set[str]`; `sample_article(ctx, win_next, article_id, neg_ratio, rng) -> pd.DataFrame` (feature frame plus `label` and `article_id` columns); `build_training_set(ctx, win_next, article_ids, neg_ratio=NEG_RATIO, seed=SEED) -> pd.DataFrame`. Constants `NEG_RATIO = 20`, `SEED = 7`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prospect_model.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for src/prospect_model.py. Synthetic frames only — no dependency on data/."""
import numpy as np
import pandas as pd
import pytest

import prospect_features as pf
import prospect_model as pm
# Package-qualified: tests/__init__.py exists, so `tests` is a package and pytest puts the
# repo root on sys.path rather than tests/ itself.
from tests.test_prospect_features import make_articles, make_fact


@pytest.fixture
def ctx():
    return pf.build_context(make_fact(), make_articles(), 2024)


@pytest.fixture
def win_next():
    return pf.window(make_fact(), 2025)


def test_adopters_are_candidates_who_bought_it_next_window(ctx, win_next):
    cands = pf.candidates_for(ctx, "X")
    got = pm.adopters(win_next, "X", cands)
    assert got == {"P1"}                      # B1 also bought X in 2025, but is not a candidate


def test_sampling_labels_adopters_positive(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=10, rng=np.random.default_rng(0))
    assert out.loc[out["client_id"] == "P1", "label"].iloc[0] == 1
    assert out["label"].sum() == 1


def test_negatives_come_only_from_this_articles_candidate_pool(ctx, win_next):
    """The whole point of within-article sampling — a negative must be a client who COULD
    have bought this part, not a random client-article pair from elsewhere."""
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=10, rng=np.random.default_rng(0))
    pool = set(pf.candidates_for(ctx, "X"))
    assert set(out["client_id"]) <= pool
    for holder in ["B1", "B2", "B3", "B4", "P4"]:
        assert holder not in set(out["client_id"])


def test_sampling_honours_the_negative_ratio(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=2, rng=np.random.default_rng(0))
    assert (out["label"] == 0).sum() <= 2 * (out["label"] == 1).sum()


def test_sampling_is_deterministic_under_a_seed(ctx, win_next):
    a = pm.sample_article(ctx, win_next, "X", neg_ratio=3, rng=np.random.default_rng(42))
    b = pm.sample_article(ctx, win_next, "X", neg_ratio=3, rng=np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)


def test_article_with_no_adopters_contributes_nothing(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "S1", neg_ratio=5, rng=np.random.default_rng(0))
    assert out.empty


def test_training_set_carries_the_article_id_for_grouping(ctx, win_next):
    out = pm.build_training_set(ctx, win_next, ["X"], neg_ratio=5, seed=1)
    assert "article_id" in out.columns
    assert set(out["article_id"]) == {"X"}
    assert list(out.columns) == pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prospect_model'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/prospect_model.py`:

```python
# -*- coding: utf-8 -*-
"""Training-set construction, evaluation and scoring for the block-2 prospect model.

Deliberately free of any Streamlit import: the page is a display layer, and this module has to
be unit-testable and callable from a training script or an API route without a browser.

The sampling rule here is the one design decision most able to invalidate the whole model, and
it is the reason this file exists separately from prospect_features.py — see sample_article().
"""
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prospect_features as pf

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"
REPORTS = DATA_ROOT / "reports"

MODEL_PATH = DATA / "prospect_model.cbm"

NEG_RATIO = 20          # negatives drawn per positive, within the same article
SEED = 7
TOP_N = pf.ap.TOP_N


def adopters(win_next: pd.DataFrame, article_id: str, candidates) -> set:
    """Candidates who went on to buy the article in the label window.

    Intersected with `candidates` on purpose: a client who already held the article is a repeat
    buyer, not an adoption, and counting them would train the model on block 1's population.
    """
    bought = pf.ap.buyers_of(win_next, article_id)
    return bought & set(candidates)


def sample_article(ctx: pf.WindowContext, win_next: pd.DataFrame, article_id: str,
                   neg_ratio: int, rng) -> pd.DataFrame:
    """Positives and sampled negatives for ONE article.

    Negatives are drawn from this article's own candidate pool, never from the global grid.
    The page ranks clients within a single article, so training must pose the same question:
    "among clients who could buy X, which ones will". Sampling across articles instead teaches
    the model to recognise plausible articles, which inflates AUC and ranks nothing usefully.

    Articles with no adoption in the label window are skipped: they contribute negatives only,
    and a group with no positive cannot inform a within-article ranking.
    """
    cands = pf.candidates_for(ctx, article_id)
    if not cands:
        return pd.DataFrame()

    pos = adopters(win_next, article_id, cands)
    if not pos:
        return pd.DataFrame()

    neg_pool = [c for c in cands if c not in pos]
    n_neg = min(len(neg_pool), neg_ratio * len(pos))
    neg = list(rng.choice(neg_pool, size=n_neg, replace=False)) if n_neg else []

    chosen = sorted(pos) + sorted(neg)
    feats = pf.build_features(ctx, article_id, chosen)
    feats = feats.reset_index()
    feats["article_id"] = article_id
    feats["label"] = feats["client_id"].isin(pos).astype(int)
    return feats[pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"]]


def build_training_set(ctx: pf.WindowContext, win_next: pd.DataFrame, article_ids,
                       neg_ratio: int = NEG_RATIO, seed: int = SEED) -> pd.DataFrame:
    """Stack per-article samples into one frame, keeping article_id so groups stay visible."""
    rng = np.random.default_rng(seed)
    parts = [sample_article(ctx, win_next, a, neg_ratio, rng) for a in article_ids]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"])
    return pd.concat(parts, ignore_index=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_model.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_model.py tests/test_prospect_model.py
git commit -m "feat(prospect-model): add within-article negative sampling"
```

---

### Task 8: Evaluation metrics and the baselines to beat

**Files:**
- Modify: `src/prospect_model.py` (append)
- Modify: `tests/test_prospect_model.py` (append)

**Interfaces:**
- Consumes: Task 7.
- Produces: `precision_at_k(ranked, actual, k=TOP_N) -> float`; `recall_at_k(ranked, actual, k=TOP_N) -> float`; `evaluate_ranker(ctx, win_next, article_ids, rank_fn, k=TOP_N) -> dict` returning `{"precision_at_k", "recall_at_k", "n_articles"}`; `heuristic_ranking(ctx, article_id, client_ids) -> list[str]` and `size_ranking(ctx, article_id, client_ids) -> list[str]`, both usable as `rank_fn`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_model.py`:

```python
def test_precision_at_k_counts_hits_in_the_top_k():
    assert pm.precision_at_k(["a", "b", "c"], {"a", "c"}, k=2) == pytest.approx(0.5)
    assert pm.precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
    assert pm.precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_divides_by_k_not_by_the_list_length():
    """Three candidates and one hit is 1/5, not 1/3 — the page shows five slots either way."""
    assert pm.precision_at_k(["a", "b", "c"], {"a"}, k=5) == pytest.approx(0.2)


def test_recall_at_k_is_relative_to_what_was_available():
    assert pm.recall_at_k(["a", "b"], {"a", "c"}, k=2) == pytest.approx(0.5)
    assert pm.recall_at_k(["a"], set(), k=5) == 0.0


def test_evaluate_skips_articles_with_no_adoption(ctx, win_next):
    """S1 gains no new buyer in 2025; including it would measure the catalogue, not the ranker."""
    out = pm.evaluate_ranker(ctx, win_next, ["X", "S1"], pm.heuristic_ranking)
    assert out["n_articles"] == 1


def test_heuristic_ranking_puts_the_strongest_basket_overlap_first(ctx):
    ranked = pm.heuristic_ranking(ctx, "X", ["P3", "P2", "P1"])
    assert ranked[0] == "P1"
    assert ranked.index("P2") < ranked.index("P3")


def test_size_ranking_ignores_the_article_entirely(ctx):
    """The trap baseline: it returns the biggest clients whatever the part."""
    a = pm.size_ranking(ctx, "X", ["P1", "P2", "P3"])
    b = pm.size_ranking(ctx, "S1", ["P1", "P2", "P3"])
    assert a == b


def test_popularity_and_random_baselines_return_every_candidate(ctx):
    """The two remaining floors. Random is seeded so a run is reproducible."""
    cands = ["P1", "P2", "P3"]
    assert sorted(pm.popularity_ranking(ctx, "X", cands)) == sorted(cands)
    assert sorted(pm.random_ranking(ctx, "X", cands)) == sorted(cands)
    assert pm.random_ranking(ctx, "X", cands) == pm.random_ranking(ctx, "X", cands)


def test_evaluate_reports_the_heuristic_finding_the_adopter(ctx, win_next):
    out = pm.evaluate_ranker(ctx, win_next, ["X"], pm.heuristic_ranking)
    assert out["precision_at_k"] == pytest.approx(1 / pm.TOP_N)
    assert out["recall_at_k"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_model.py -k "precision or recall or evaluate or ranking" -v`
Expected: FAIL — `AttributeError: module 'prospect_model' has no attribute 'precision_at_k'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_model.py`:

```python
def precision_at_k(ranked, actual, k: int = TOP_N) -> float:
    """Hits in the top k, divided by k.

    Divided by k rather than by len(ranked): the page always offers five slots, so a short
    list that fills three of them has still spent five slots' worth of the vendor's attention.
    """
    if not k:
        return 0.0
    return len([c for c in list(ranked)[:k] if c in actual]) / k


def recall_at_k(ranked, actual, k: int = TOP_N) -> float:
    """Share of the window's actual adopters that the top k captured."""
    if not actual:
        return 0.0
    return len([c for c in list(ranked)[:k] if c in actual]) / len(actual)


def heuristic_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """The incumbent: rank by signature overlap, ties broken by client CA.

    Reproduces article_prospects.prospects() over an explicit client list so the two can be
    scored on identical candidates. This is the bar the model has to clear.
    """
    sig = pf.signature_with_lift(ctx.win, article_id)
    aff = pf.affinity_features(ctx, article_id, client_ids, sig)
    ca = ctx.client_feats["client_ca"].reindex(list(client_ids)).fillna(0.0)
    order = pd.DataFrame({"n": aff["n_signature_held"], "ca": ca})
    order = order[order["n"] > 0].sort_values(["n", "ca"], ascending=False)
    return list(order.index)


def size_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: biggest clients first, article ignored.

    Named explicitly because it is the specific failure the signature was built to avoid —
    a ranking that returns the same large clients for every part in the catalogue.
    """
    ca = ctx.client_feats["client_ca"].reindex(list(client_ids)).fillna(0.0)
    return list(ca.sort_values(ascending=False).index)


def popularity_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: clients who buy the widest range of articles first.

    Distinct from size_ranking: a client can spend heavily on few references, or little across
    many. Breadth is the more plausible naive predictor of trying something new, so it is the
    tougher of the two floors.
    """
    n = ctx.client_feats["client_n_articles"].reindex(list(client_ids)).fillna(0.0)
    return list(n.sort_values(ascending=False).index)


def random_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: shuffle. Seeded per article so a run is reproducible.

    Seeding on the article id rather than a global counter keeps each article's shuffle
    independent of the order articles happen to be evaluated in.

    crc32, not hash(): Python randomises string hashing per process unless PYTHONHASHSEED is
    set, so hash() would give a different "reproducible" baseline on every run.
    """
    ids = list(client_ids)
    rng = np.random.default_rng(zlib.crc32(article_id.encode("utf-8")))
    return [ids[i] for i in rng.permutation(len(ids))]


def evaluate_ranker(ctx: pf.WindowContext, win_next: pd.DataFrame, article_ids,
                    rank_fn, k: int = TOP_N) -> dict:
    """Average precision@k and recall@k over articles that actually gained a new buyer.

    Articles with no adoption are excluded: precision@k is 0 for them however good the
    ranking, so averaging them in would measure the catalogue's adoption rate instead of the
    ranker. Their count is reported so the coverage stays visible.
    """
    precisions, recalls, used = [], [], 0
    for article_id in article_ids:
        cands = pf.candidates_for(ctx, article_id)
        if not cands:
            continue
        actual = adopters(win_next, article_id, cands)
        if not actual:
            continue
        ranked = rank_fn(ctx, article_id, cands)
        precisions.append(precision_at_k(ranked, actual, k))
        recalls.append(recall_at_k(ranked, actual, k))
        used += 1
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "n_articles": used,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_model.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add src/prospect_model.py tests/test_prospect_model.py
git commit -m "feat(prospect-model): add precision@5 evaluation and baseline rankers"
```

---

### Task 9: Model scoring, ranking entry point and fallback

**Files:**
- Modify: `src/prospect_model.py` (append)
- Modify: `tests/test_prospect_model.py` (append)

**Interfaces:**
- Consumes: Tasks 7–8.
- Produces: `load_model(path=MODEL_PATH)` returning a `CatBoostClassifier` or `None`; `model_ranking(ctx, article_id, client_ids, model) -> list[str]`; `rank_prospects(ctx, article_id, vendor_code, model=None, top=TOP_N) -> tuple[pd.DataFrame, list[str], str]` where the frame has columns `RANKED_COLUMNS` and the third element is `"model"` or `"heuristic"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prospect_model.py`:

```python
def test_load_model_returns_none_when_no_artifact_exists(tmp_path):
    assert pm.load_model(tmp_path / "absent.cbm") is None


def test_rank_prospects_falls_back_to_the_heuristic_without_a_model(ctx):
    out, sig, source = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert source == "heuristic"
    assert list(out["client_id"]) == ["P1", "P2"]
    assert sig


def test_rank_prospects_keeps_the_vendor_scope(ctx):
    """O1 belongs to V2 and must never appear in V1's list."""
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert "O1" not in set(out["client_id"])


def test_rank_prospects_excludes_clients_buying_it_through_another_vendor(ctx):
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert "P4" not in set(out["client_id"])


def test_rank_prospects_returns_a_typed_empty_frame_when_the_signature_is_refused(ctx):
    out, sig, _ = pm.rank_prospects(ctx, "Z", "V1", model=None)
    assert sig == []
    assert out.empty
    assert list(out.columns) == pm.RANKED_COLUMNS


def test_rank_prospects_honours_the_top_limit(ctx):
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None, top=1)
    assert list(out["client_id"]) == ["P1"]


def test_model_ranking_orders_by_predicted_probability(ctx):
    class FakeModel:
        """Scores P2 above P1, the reverse of the heuristic — proves the model is in charge."""
        def predict_proba(self, pool):
            order = {"P1": 0.1, "P2": 0.9, "P3": 0.5}
            ids = pool if isinstance(pool, list) else list(pool.index)
            return np.array([[1 - order.get(c, 0.0), order.get(c, 0.0)] for c in ids])

    ranked = pm.model_ranking(ctx, "X", ["P1", "P2", "P3"], FakeModel())
    assert ranked[0] == "P2"


def test_rank_prospects_reports_the_model_as_its_source(ctx):
    class FakeModel:
        def predict_proba(self, pool):
            ids = pool if isinstance(pool, list) else list(pool.index)
            return np.array([[0.5, 0.5] for _ in ids])

    _, _, source = pm.rank_prospects(ctx, "X", "V1", model=FakeModel())
    assert source == "model"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m pytest tests/test_prospect_model.py -k "load_model or rank_prospects or model_ranking" -v`
Expected: FAIL — `AttributeError: module 'prospect_model' has no attribute 'load_model'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/prospect_model.py`:

```python
RANKED_COLUMNS = ["client_id", "n_signature", "client_ca", "score"]


def load_model(path=MODEL_PATH):
    """The fitted CatBoost model, or None when it has not been trained yet.

    Returning None rather than raising keeps a fresh clone working: the page falls back to the
    heuristic and renders normally instead of failing on a missing artifact in data/, which is
    gitignored and therefore absent for everyone who has not run training.
    """
    from catboost import CatBoostClassifier

    path = Path(path)
    if not path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def model_ranking(ctx: pf.WindowContext, article_id: str, client_ids, model) -> list:
    """Rank candidates by the model's probability of adoption, highest first."""
    client_ids = list(client_ids)
    if not client_ids:
        return []
    feats = pf.build_features(ctx, article_id, client_ids)
    proba = model.predict_proba(feats)[:, 1]
    return list(pd.Series(proba, index=feats.index).sort_values(ascending=False).index)


def rank_prospects(ctx: pf.WindowContext, article_id: str, vendor_code: str,
                   model=None, top: int = TOP_N):
    """Block 2, model-ranked when a model exists and heuristic-ranked when it does not.

    Returns (frame, signature, source). `source` is surfaced so the page — and any future API
    route — can tell the two apart, and so a screenshot of the page is self-describing.

    The vendor filter is applied AFTER scoring: the model answers "who wants this part", and
    ownership is a separate question about who is allowed to be shown it.
    """
    signature = list(pf.signature_with_lift(ctx.win, article_id).index)
    empty = pd.DataFrame(columns=RANKED_COLUMNS)
    if not signature:
        return empty, [], "heuristic"

    owned = pf.ap.vendor_clients(ctx.win, vendor_code)
    candidates = [c for c in pf.candidates_for(ctx, article_id) if c in owned]
    if not candidates:
        return empty, signature, "heuristic"

    source = "model" if model is not None else "heuristic"
    ranked = (model_ranking(ctx, article_id, candidates, model) if model is not None
              else heuristic_ranking(ctx, article_id, candidates))
    if not ranked:
        return empty, signature, source

    sig = pf.signature_with_lift(ctx.win, article_id)
    aff = pf.affinity_features(ctx, article_id, ranked, sig)
    ca = ctx.client_feats["client_ca"].reindex(ranked).fillna(0.0)
    out = pd.DataFrame({
        "client_id": ranked,
        "n_signature": aff["n_signature_held"].reindex(ranked).values,
        "client_ca": ca.values,
        "score": range(1, len(ranked) + 1),
    })
    return out.head(top).reset_index(drop=True), signature, source
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m pytest tests/test_prospect_model.py -v`
Expected: PASS, 23 tests.

- [ ] **Step 5: Run the whole suite**

Run: `py -m pytest tests/ -q`
Expected: PASS, 151 tests (103 baseline + 25 features + 23 model).

- [ ] **Step 6: Commit**

```bash
git add src/prospect_model.py tests/test_prospect_model.py
git commit -m "feat(prospect-model): add model scoring with heuristic fallback"
```

---

### Task 10: The training script, and a real training run

**Files:**
- Create: `src/train_prospect_model.py`

**Interfaces:**
- Consumes: `prospect_model`, `prospect_features`.
- Produces: `data/prospect_model.cbm`, `reports/prospect_model.md`, `reports/prospect_metrics.json`. Nothing later in this plan imports from it.

- [ ] **Step 1: Write the script**

Create `src/train_prospect_model.py`:

```python
# -*- coding: utf-8 -*-
"""Train and evaluate the block-2 prospect model.

Run:  py src/train_prospect_model.py

  TRAIN : features 2024 -> label = adopted in 2025
  TEST  : features 2025 -> label = adopted in 2026   (untouched until the final evaluation)

Mirrors src/train_model.py: same split, same snapshot-year symmetry, a human-readable report
in reports/ and a machine-readable JSON beside it so the validation page and the API can read
the numbers exactly as they read model_metrics.json today.

The model must beat the incumbent heuristic on precision@5 to be worth shipping. If it does
not, that is a result: record it and keep the heuristic.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prospect_features as pf
import prospect_model as pm

TRAIN_ARTICLES = 4000     # articles sampled for the training set; the full catalogue is not
EVAL_ARTICLES = 2000      # articles scored at evaluation time
MIN_BUYERS = pf.ap.MIN_BUYERS

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def eligible_articles(ctx, rng, n):
    """Articles with enough buyers to have a signature, sampled without replacement."""
    counts = ctx.article_feats["article_n_buyers"]
    pool = list(counts[counts >= MIN_BUYERS].index)
    if len(pool) <= n:
        return sorted(pool)
    return sorted(rng.choice(pool, size=n, replace=False))


def main() -> None:
    fact = pd.read_parquet(pf.DATA / "ca_monthly.parquet")
    articles = pd.read_csv(pf.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")

    emit("# Prospect model — block 2")
    emit()
    emit("Label: the client bought an article they had NEVER bought in the snapshot window.")
    emit("Features come from the snapshot window only, so train and test see the same shapes.")
    emit()

    emit("## Building windows")
    ctx_train = pf.build_context(fact, articles, 2024)
    ctx_test = pf.build_context(fact, articles, 2025)
    win_2025 = pf.window(fact, 2025)
    win_2026 = pf.window(fact, 2026)
    emit(f"- train window 2024: {len(ctx_train.clients):,} clients, "
         f"{len(ctx_train.articles):,} articles")
    emit(f"- test  window 2025: {len(ctx_test.clients):,} clients, "
         f"{len(ctx_test.articles):,} articles")
    emit()

    rng = np.random.default_rng(pm.SEED)
    train_ids = eligible_articles(ctx_train, rng, TRAIN_ARTICLES)
    emit(f"## Sampling {len(train_ids):,} articles, {pm.NEG_RATIO} negatives per positive")
    train = pm.build_training_set(ctx_train, win_2025, train_ids)
    emit(f"- rows: {len(train):,}   positives: {int(train['label'].sum()):,} "
         f"({100 * train['label'].mean():.2f}%)")
    emit()

    feats = pf.FEATURE_COLUMNS
    cat_idx = [feats.index(c) for c in pf.CATEGORICAL_COLUMNS]
    cut = int(0.85 * len(train))
    tr, val = train.iloc[:cut], train.iloc[cut:]

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        eval_metric="AUC", random_seed=pm.SEED,
        early_stopping_rounds=50, verbose=100)
    model.fit(Pool(tr[feats], tr["label"], cat_features=cat_idx),
              eval_set=Pool(val[feats], val["label"], cat_features=cat_idx))

    pm.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(pm.MODEL_PATH))
    emit(f"- model saved to {pm.MODEL_PATH}")
    emit()

    emit("## Held-out evaluation — features 2025, labels 2026")
    eval_ids = eligible_articles(ctx_test, np.random.default_rng(pm.SEED + 1), EVAL_ARTICLES)

    def model_rank(ctx, article_id, client_ids):
        return pm.model_ranking(ctx, article_id, client_ids, model)

    results = {}
    for name, fn in [("model", model_rank),
                     ("heuristic", pm.heuristic_ranking),
                     ("client size", pm.size_ranking),
                     ("client breadth", pm.popularity_ranking),
                     ("random", pm.random_ranking)]:
        results[name] = pm.evaluate_ranker(ctx_test, win_2026, eval_ids, fn)

    emit()
    emit("| ranker | precision@5 | recall@5 | articles |")
    emit("| --- | --- | --- | --- |")
    for name, r in results.items():
        emit(f"| {name} | {r['precision_at_k']:.4f} | {r['recall_at_k']:.4f} "
             f"| {r['n_articles']:,} |")
    emit()

    test = pm.build_training_set(ctx_test, win_2026, eval_ids[:500])
    auc = float(roc_auc_score(test["label"], model.predict_proba(test[feats])[:, 1])) \
        if test["label"].nunique() > 1 else float("nan")
    emit(f"AUC on a sampled held-out set: {auc:.4f}")
    emit()

    imp = pd.Series(model.get_feature_importance(), index=feats).sort_values(ascending=False)
    emit("## Feature importance (top 12)")
    emit()
    emit("| feature | importance |")
    emit("| --- | --- |")
    for k, v in imp.head(12).items():
        emit(f"| {k} | {v:.2f} |")
    emit()

    won = results["model"]["precision_at_k"] > results["heuristic"]["precision_at_k"]
    emit(f"## Verdict: {'model beats the heuristic' if won else 'HEURISTIC NOT BEATEN'}")
    if not won:
        emit()
        emit("Ship criterion not met. Keep the heuristic and record the finding.")

    REPORTS = pf.DATA_ROOT / "reports"
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "prospect_model.md").write_text("\n".join(LINES), encoding="utf-8")
    (REPORTS / "prospect_metrics.json").write_text(json.dumps({
        "rankers": results,
        "auc": auc,
        "beats_heuristic": bool(won),
        "train_rows": int(len(train)),
        "train_positive_rate": float(train["label"].mean()),
        "feature_importance": {k: float(v) for k, v in imp.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORTS / 'prospect_model.md'} and prospect_metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the training**

```bash
py src/train_prospect_model.py
```

Expected: completes without traceback, prints the ranker comparison table, writes
`reports/prospect_model.md` and `reports/prospect_metrics.json`, and saves
`data/prospect_model.cbm`. Expect this to take several minutes.

- [ ] **Step 3: Read the verdict**

Open `reports/prospect_model.md` and check the comparison table.

- If **model precision@5 > heuristic precision@5**: the ship criterion is met; continue to Task 11.
- If **not**: stop and report. The heuristic stays, the finding is recorded, and Task 11 is not
  performed. This is a valid outcome, not a failure to fix by tuning until the number moves.

- [ ] **Step 4: Confirm the suite still passes**

Run: `py -m pytest tests/ -q`
Expected: PASS, 151 tests. Training writes only to `data/` and `reports/`, so nothing should change.

- [ ] **Step 5: Commit**

```bash
git add src/train_prospect_model.py reports/prospect_model.md reports/prospect_metrics.json
git commit -m "feat(prospect-model): add the training script and record the first run"
```

---

### Task 11: Wire the page, and final verification

**Files:**
- Modify: `src/pages/2_Prospection.py:1-40` (imports and cached loader) and the block-2 section
- Modify: `tests/test_prospection_page.py` (append)

**Interfaces:**
- Consumes: `prospect_model.rank_prospects`, `prospect_features.build_context`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the context and model to the cached loader**

In `src/pages/2_Prospection.py`, add these imports beside the existing ones:

```python
import prospect_features as pfeat
import prospect_model as pmod
```

Then add this cached loader immediately after the existing `load_all()` function:

```python
@st.cache_resource
def load_prospect_model():
    """The fitted model, or None when training has not been run.

    cache_resource rather than cache_data: a CatBoost model is an unhashable object that must
    not be copied per session.
    """
    return pmod.load_model()


@st.cache_data
def load_context(year: int):
    """The window context, built once per year and reused across interactions.

    Without this the basket matrix and the client-similarity grid would be rebuilt on every
    widget change, which is seconds of work per keystroke in the search box.
    """
    return pfeat.build_context(fact, articles, year)
```

- [ ] **Step 2: Replace the block-2 computation**

Find this line in the block 2 section:

```python
block2, signature = ap.prospects(fact, article_id, vendor_code)
```

Replace it with:

```python
ctx = load_context(2025)
block2, signature, source = pmod.rank_prospects(
    ctx, article_id, vendor_code, model=load_prospect_model())
```

Leave every `st.info` / `st.caption` / `st.warning` string in that section exactly as it is.
The ranking changes; the wording does not. Block 2 still shows no percentage.

- [ ] **Step 3: Write the test**

Append to `tests/test_prospection_page.py`:

```python
def test_block2_still_shows_no_probability_after_the_model_change(at):
    """The model changes the ORDER of the five, never what is displayed about them."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    assert "propension" not in tail
    assert "%" not in tail


def test_page_renders_when_no_model_artifact_exists(at):
    """A fresh clone has no data/prospect_model.cbm; the page must fall back, not crash."""
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "À prospecter" in text
```

- [ ] **Step 4: Run the page tests**

Run: `py -m pytest tests/test_prospection_page.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Boot the app and confirm the page renders**

```bash
py -m streamlit run src/app.py --server.headless true --server.port 8512
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8512
```

Expected: `200`, no traceback in the console. Stop the server afterwards.

- [ ] **Step 6: Run the full suite**

Run: `py -m pytest tests/ -q`
Expected: PASS, **153 tests** (103 baseline + 25 features + 23 model + 2 page).

- [ ] **Step 7: Confirm the untouched surfaces still work**

```bash
py -m pytest tests/test_app.py tests/test_validation_page.py tests/test_api.py tests/test_article_prospects.py -q
```

Expected: PASS. These cover the vendor queue, the validation page, the API and the block-2
heuristic — none of which this feature modifies. A failure here means the additive-only
constraint was broken.

- [ ] **Step 8: Commit**

```bash
git add src/pages/2_Prospection.py tests/test_prospection_page.py
git commit -m "feat(prospect-model): rank block 2 with the model, falling back to the heuristic"
```

---

## Verification checklist

- [ ] `py -m pytest tests/ -q` → 153 passed
- [ ] `src/article_prospects.py` is byte-identical to its state at the start: `git diff main -- src/article_prospects.py` is empty
- [ ] Neither `prospect_features.py` nor `prospect_model.py` contains `import streamlit`
- [ ] `reports/prospect_model.md` shows the model beating the heuristic on precision@5
- [ ] Block 2 renders no percentage anywhere
- [ ] The page boots inside a real Streamlit server (HTTP 200, clean logs)
- [ ] Deleting `data/prospect_model.cbm` still leaves the page working (fallback path)
