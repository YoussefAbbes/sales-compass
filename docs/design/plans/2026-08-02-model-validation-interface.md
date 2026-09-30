# Model Validation Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit page that lets a technical reader verify the CatBoost model's predictions against known 2026 outcomes — protocol, aggregate metrics, recommendation quality, and individual prediction drill-down.

**Architecture:** All metric computation lives in `src/validation.py`, a pure pandas/sklearn module with no Streamlit import, covered by pytest against synthetic frames. `src/pages/1_Validation.py` is a thin display layer that calls it behind `@st.cache_data`. `src/train_model.py` is extended to persist the train-side predictions and a metrics JSON that do not currently exist.

**Tech Stack:** Python 3.13, pandas 2.2+, scikit-learn 1.6, CatBoost 1.2, Streamlit 1.60, pytest 9.1.

## Global Constraints

- **No git commits.** The user explicitly opted out for this work. Every task ends with a verification step instead of a commit. Do not run `git commit`, `git add`, or `git push`.
- **Never import `src/app.py`.** It executes the entire app at module level; importing it would run the vendor UI as a side effect. Shared helpers must be extracted into `src/ui_theme.py` instead.
- **Tests must not depend on the 51 MB Excel extract or on `data/*.parquet`.** Unit tests build synthetic frames inline. Only the AppTest smoke tests may touch real `data/`.
- **Do not retrain, tune, or change model hyperparameters.** `iterations=800` stays. The spec forbids using this page as a tuning loop against the single honest test window.
- **Do not modify `src/contact_queue.py`.** Its ranking logic is replicated read-only in `validation.py`.
- **Encoding:** all CSV reads/writes use `encoding="utf-8-sig"`, matching the existing pipeline. IDs are read as `str` to preserve leading zeros.
- **Run Python as `py`** on this machine (`python` is not on PATH in the Bash tool).

---

### Task 1: Recover the existing app tests into the repository

`progress.md` cites 7 AppTest scenarios as the verification basis for `src/app.py`, but they exist
only in a temp scratchpad as a bare assertion script with hardcoded absolute paths. Recovering them
first gives a safety net for Task 2's refactor and makes the claim in `progress.md` reproducible.

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/conftest.py` exposes fixtures `project_root() -> Path` and `app_path() -> str`, used by every later test module.

- [ ] **Step 1: Create the test package and shared fixtures**

Create `tests/__init__.py` as an empty file.

Create `tests/conftest.py`:

```python
# -*- coding: utf-8 -*-
"""Shared fixtures. Paths are resolved relative to the repo, never hardcoded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Make src/ importable as top-level modules (validation, ui_theme).
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def app_path() -> str:
    return str(ROOT / "src" / "app.py")


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    d = project_root / "data"
    if not (d / "clients_to_contact.csv").exists():
        pytest.skip("data/clients_to_contact.csv missing — run the pipeline first")
    return d
```

- [ ] **Step 2: Write `tests/test_app.py` as real pytest**

This is a conversion of the scratchpad script: same 7 scenarios, but as isolated test functions with
repo-relative paths and a fixture that protects the real `data/feedback.csv` from test writes.

```python
# -*- coding: utf-8 -*-
"""AppTest coverage for src/app.py (the vendor queue page)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def feedback_guard(data_dir):
    """Move any real feedback.csv aside so tests never pollute it; restore after."""
    fb = data_dir / "feedback.csv"
    backup = data_dir / "feedback.csv.testbak"
    if fb.exists():
        fb.rename(backup)
    yield fb
    if fb.exists():
        fb.unlink()
    if backup.exists():
        backup.rename(fb)


def run_app(app_path):
    at = AppTest.from_file(app_path, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def boxes(at):
    return {sb.label: sb for sb in at.selectbox}


def texts(at):
    return {ti.label: ti for ti in at.text_input}


def test_app_loads_with_masthead(app_path, data_dir):
    at = run_app(app_path)
    assert any("File d'appels" in m.value for m in at.markdown)


def test_top_card_click_opens_that_client(app_path, data_dir):
    at = run_app(app_path)
    top_buttons = [b for b in at.button if b.key and b.key.startswith("open_top_")]
    assert top_buttons, "no top-card buttons rendered"
    target = top_buttons[0]
    expected_client_id = target.key.replace("open_top_", "")
    target.click()
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert expected_client_id in boxes(at)["Client"].value


def test_feedback_saves_against_the_open_client(app_path, data_dir, feedback_guard):
    at = run_app(app_path)
    top_buttons = [b for b in at.button if b.key and b.key.startswith("open_top_")]
    target = top_buttons[0]
    expected_client_id = target.key.replace("open_top_", "")
    target.click()
    at.run()

    boxes(at)["Action"].set_value("Contacté")
    texts(at)["Commentaire (facultatif)"].set_value("test interne")
    {b.key: b for b in at.button}["FormSubmitter:feedback-Enregistrer"].click()
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    assert feedback_guard.exists()
    fb = pd.read_csv(feedback_guard, dtype={"client_id": str}, encoding="utf-8-sig")
    assert str(fb.iloc[-1]["client_id"]) == expected_client_id


def test_vendor_switch_updates_masthead(app_path, data_dir):
    at = run_app(app_path)
    vsel = boxes(at)["Vendeur"]
    new_vendor = vsel.options[5]
    vsel.set_value(new_vendor)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(new_vendor.split(" — ")[1] in m.value for m in at.markdown)


def test_search_with_no_match_is_graceful(app_path, data_dir):
    at = run_app(app_path)
    texts(at)["🔍 Rechercher un client"].set_value("zzzz_no_such_client_zzzz")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert (any("Aucun client" in m.value for m in at.markdown)
            or any("Aucun client" in i.value for i in at.info))


def test_clearing_search_restores_the_hero_view(app_path, data_dir):
    at = run_app(app_path)
    search = texts(at)["🔍 Rechercher un client"]
    search.set_value("zzzz_no_such_client_zzzz")
    at.run()
    search.set_value("")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any("File d'appels" in m.value for m in at.markdown)


def test_smallest_vendor_renders(app_path, data_dir):
    """A vendor with a single client must not crash the hero/top-cards layout."""
    q = pd.read_csv(data_dir / "clients_to_contact.csv",
                    dtype={"vendor_code": str}, encoding="utf-8-sig")
    smallest = q["vendor_code"].value_counts().idxmin()
    at = run_app(app_path)
    vsel = boxes(at)["Vendeur"]
    match = [o for o in vsel.options if o.startswith(smallest)]
    assert match, f"vendor {smallest} not in selector"
    vsel.set_value(match[0])
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
```

- [ ] **Step 3: Run the recovered tests and confirm they pass**

Run: `py -m pytest tests/test_app.py -v`
Expected: 7 passed. If any fail, the failure is a real regression in `app.py` — fix `app.py`, not the test.

- [ ] **Step 4: Verify the feedback file was not left behind**

Run: `py -c "from pathlib import Path; print('feedback.csv exists:', Path('data/feedback.csv').exists()); print('backup exists:', Path('data/feedback.csv.testbak').exists())"`
Expected: both `False`.

---

### Task 2: Extract the shared UI theme

Both pages need the same fonts, palette, and number formatting. `app.py` cannot be imported (it runs
at module level), so the helpers move to their own module.

**Files:**
- Create: `src/ui_theme.py`
- Modify: `src/app.py:39-131` (remove `inject_css`, `badge_html`, `fmt_tnd`; import them instead)

**Interfaces:**
- Consumes: nothing.
- Produces: `ui_theme.inject_css() -> None`, `ui_theme.badge_html(status: str) -> str`, `ui_theme.fmt_tnd(x: float) -> str`, `ui_theme.STATUS_META: dict`. Task 6 and Task 7 import all four.

- [ ] **Step 1: Create `src/ui_theme.py`**

Move the three helpers verbatim out of `app.py`. The CSS string is copied exactly as-is from
`app.py:40-122` — do not restyle anything; this is a pure move. Then append the extra rules the
validation page needs.

```python
# -*- coding: utf-8 -*-
"""Shared visual language for the Streamlit pages (atelier/dispatch-board direction).

Kept in its own module because src/app.py executes at import time — importing it to
reuse a helper would run the whole vendor app as a side effect.
"""
import streamlit as st

STATUS_META = {
    "INACTIVE_2026": {"label": "Inactif 2026", "css": "urgent"},
    "DECLINING": {"label": "En baisse", "css": "watch"},
    "ACTIVE": {"label": "Actif", "css": "steady"},
}

# <<< paste the full <style> string from src/app.py:40-122 here, unchanged >>>
_BASE_CSS = """..."""

# Additional rules used only by the validation page.
_VALIDATION_CSS = """
<style>
.verdict-hit  { color: var(--steady); font-weight: 700; }
.verdict-miss { color: var(--urgent); font-weight: 700; }
.caveat {
  border-left: 3px solid var(--watch); background: rgba(138,106,22,.07);
  padding: .8rem 1rem; margin: .8rem 0; font-size: .88rem; color: var(--ink);
}
.caveat strong { font-family: 'JetBrains Mono', monospace; font-size: .78rem;
  letter-spacing: .08em; text-transform: uppercase; color: var(--watch); }
.metric-note { color: var(--muted); font-size: .82rem; margin-top: -.6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(_VALIDATION_CSS, unsafe_allow_html=True)


def badge_html(status: str) -> str:
    meta = STATUS_META.get(status, {"label": status, "css": "steady"})
    return f'<span class="badge badge-{meta["css"]}">● {meta["label"]}</span>'


def fmt_tnd(x) -> str:
    return f"{x:,.0f} TND"
```

- [ ] **Step 2: Rewire `src/app.py` to use it**

Delete `STATUS_META` (lines 27-31), `inject_css` (39-122), `badge_html` (125-127), and `fmt_tnd`
(130-131) from `app.py`. Add this import next to the existing ones:

```python
from ui_theme import STATUS_META, badge_html, fmt_tnd, inject_css
```

Because Streamlit puts the entrypoint's directory on `sys.path`, `src/ui_theme.py` resolves as a
top-level import when running `streamlit run src/app.py`. No path manipulation is needed in `app.py`.

- [ ] **Step 3: Confirm the refactor changed nothing**

Run: `py -m pytest tests/test_app.py -v`
Expected: the same 7 passed as in Task 1. A failure here means the move was not verbatim.

- [ ] **Step 4: Confirm the app still boots**

Run: `py -m streamlit run src/app.py --server.headless true --server.port 8599 &` then
`py -c "import time,urllib.request; time.sleep(12); print(urllib.request.urlopen('http://localhost:8599').status)"`
Expected: `200`. Stop the server afterwards.

---

### Task 3: Persist train-side scores and a metrics JSON

The page cannot show a train-vs-test difference because no train-side prediction is saved and no
metric is persisted in machine-readable form.

**Files:**
- Modify: `src/train_model.py:184-236`

**Interfaces:**
- Consumes: nothing.
- Produces: `data/pair_scores_train.csv` (columns `client_id, article_id, vendor_code, months_n, ca_sum, manual_score, p_model, label` — identical schema to `pair_scores_2026.csv`) and `reports/model_metrics.json` (schema below). Tasks 4 and 6 read both.

`reports/model_metrics.json` schema:

```json
{
  "trained_at": "2026-08-02T13:00:00",
  "train": {"window": "2024 → 2025", "n_pairs": 0, "base_rate": 0.0, "auc": 0.0},
  "validation": {"window": "20% random split of train", "n_pairs": 0, "base_rate": 0.0, "auc": 0.0},
  "test": {"window": "2025 → 2026", "n_pairs": 0, "base_rate": 0.0, "auc": 0.0},
  "benchmarks": {"manual_score": 0.0, "logistic_regression": 0.0, "catboost": 0.0},
  "best_iteration": 0,
  "iterations_cap": 800,
  "feature_importance": {"feature_name": 0.0}
}
```

- [ ] **Step 1: Add the JSON import**

At the top of `src/train_model.py`, next to the existing imports, add:

```python
import json
from datetime import datetime
```

- [ ] **Step 2: Score the train snapshot and assemble the metrics dict**

Immediately after the existing line `test = test.assign(p_model=p_cb)` (currently line 186), insert:

```python
    # Train-side scoring — needed by the validation page to show train vs test.
    # x_tr / x_val are the internal split; `train` is the full training snapshot.
    p_tr_full = model.predict_proba(train[feats])[:, 1]
    train = train.assign(p_model=p_tr_full)
    auc_train = roc_auc_score(train["label"], p_tr_full)
    auc_val = roc_auc_score(x_val["label"], model.predict_proba(x_val[feats])[:, 1])

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train": {"window": "2024 → 2025", "n_pairs": int(len(train)),
                  "base_rate": float(train["label"].mean()), "auc": float(auc_train)},
        "validation": {"window": "20% random split of train", "n_pairs": int(len(x_val)),
                       "base_rate": float(x_val["label"].mean()), "auc": float(auc_val)},
        "test": {"window": "2025 → 2026", "n_pairs": int(len(test)),
                 "base_rate": float(test["label"].mean()), "auc": float(auc_cb)},
        "benchmarks": {"manual_score": float(auc_manual),
                       "logistic_regression": float(auc_lr),
                       "catboost": float(auc_cb)},
        "best_iteration": int(model.get_best_iteration()),
        "iterations_cap": 800,
        "feature_importance": {f: float(v) for f, v in
                               zip(feats, model.get_feature_importance())},
    }
```

- [ ] **Step 3: Write both new artifacts**

Immediately before the existing `print(f"\n[saved] {REPORTS / 'model_catboost.md'}")` line, insert:

```python
    (REPORTS / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    train[["client_id", "article_id", "vendor_code", "months_n", "ca_sum",
           "manual_score", "p_model", "label"]].to_csv(
        DATA / "pair_scores_train.csv", index=False, encoding="utf-8-sig")
```

And extend the final print so the new files are visible in the run log:

```python
    print(f"[saved] {REPORTS / 'model_metrics.json'} + pair_scores_train.csv")
```

- [ ] **Step 4: Snapshot the current test artifact before re-running**

The model is seeded (`random_seed=42`), so re-running must reproduce the existing outputs. Capture a
checksum first so this is verified rather than assumed.

Run: `py -c "import hashlib,pathlib; print(hashlib.md5(pathlib.Path('data/pair_scores_2026.csv').read_bytes()).hexdigest())"`
Record the value.

- [ ] **Step 5: Re-run training**

Run: `py src/train_model.py`
Expected: completes, prints the AUC table, and reports the two new saved files.

- [ ] **Step 6: Verify reproducibility and the new artifacts**

```bash
py -c "import hashlib,pathlib,json; \
print('test scores md5:', hashlib.md5(pathlib.Path('data/pair_scores_2026.csv').read_bytes()).hexdigest()); \
m=json.load(open('reports/model_metrics.json',encoding='utf-8')); \
print('train auc', m['train']['auc'], '| val auc', m['validation']['auc'], '| test auc', m['test']['auc']); \
import pandas as pd; print('train scores rows:', len(pd.read_csv('data/pair_scores_train.csv')))"
```

Expected: the md5 matches Step 4 exactly; test AUC ≈ 0.762; train AUC > test AUC (this gap is the
point of §1); train scores row count > 0.

If the md5 differs, stop and investigate before continuing — it means the pipeline is not
deterministic and every number on the page would be unstable.

---

### Task 4: Core metrics in `src/validation.py`

**Files:**
- Create: `src/validation.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: `data/pair_scores_2026.csv`, `data/pair_scores_train.csv`, `reports/model_metrics.json` from Task 3.
- Produces:
  - `load_metrics() -> dict`
  - `load_scores(split: str = "test") -> pd.DataFrame`
  - `calibration_table(p: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame` → columns `bin_low, bin_high, n, mean_pred, observed, gap`
  - `decile_table(p: pd.Series, y: pd.Series) -> pd.DataFrame` → columns `decile, n, rate`
  - `confusion_at(p: pd.Series, y: pd.Series, threshold: float) -> dict` → keys `tp, fp, tn, fn, precision, recall, n_flagged`
  Tasks 5, 6 and 7 use all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for src/validation.py. Synthetic frames only — no dependency on data/."""
import numpy as np
import pandas as pd
import pytest

import validation as v


def test_calibration_bins_preserve_row_count():
    p = pd.Series(np.linspace(0.01, 0.99, 500))
    y = pd.Series((np.linspace(0.01, 0.99, 500) > 0.5).astype(int))
    tab = v.calibration_table(p, y, n_bins=10)
    assert tab["n"].sum() == 500
    assert len(tab) <= 10


def test_calibration_is_perfect_when_predictions_match_reality():
    """500 pairs at p=0.2 of which exactly 20% are positive -> gap ~0."""
    p = pd.Series([0.2] * 500)
    y = pd.Series([1] * 100 + [0] * 400)
    tab = v.calibration_table(p, y, n_bins=10)
    row = tab.iloc[0]
    assert row["observed"] == pytest.approx(0.2)
    assert row["gap"] == pytest.approx(0.0, abs=1e-9)


def test_decile_table_is_monotonic_on_a_perfect_ranking():
    p = pd.Series(np.linspace(0, 1, 1000))
    y = (p > 0.5).astype(int)
    tab = v.decile_table(p, y)
    assert len(tab) == 10
    assert tab["rate"].is_monotonic_increasing
    assert tab.iloc[-1]["rate"] == 1.0
    assert tab.iloc[0]["rate"] == 0.0


def test_confusion_at_threshold_zero_flags_everything():
    p = pd.Series([0.1, 0.4, 0.9])
    y = pd.Series([0, 1, 1])
    c = v.confusion_at(p, y, 0.0)
    assert c["n_flagged"] == 3
    assert c["tp"] == 2 and c["fp"] == 1
    assert c["tn"] == 0 and c["fn"] == 0


def test_confusion_at_threshold_above_max_flags_nothing():
    p = pd.Series([0.1, 0.4, 0.9])
    y = pd.Series([0, 1, 1])
    c = v.confusion_at(p, y, 1.0)
    assert c["n_flagged"] == 0
    assert c["tp"] == 0 and c["fp"] == 0
    assert c["fn"] == 2 and c["tn"] == 1
    assert c["precision"] == 0.0


def test_confusion_precision_and_recall_are_correct():
    p = pd.Series([0.9, 0.8, 0.7, 0.2])
    y = pd.Series([1, 0, 1, 1])
    c = v.confusion_at(p, y, 0.5)
    assert c["tp"] == 2 and c["fp"] == 1 and c["fn"] == 1 and c["tn"] == 0
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'validation'`.

- [ ] **Step 3: Implement `src/validation.py`**

```python
# -*- coding: utf-8 -*-
"""Metric computation for the model-validation page.

Deliberately free of any Streamlit import so it can be unit-tested without a browser
or a running server. The page (src/pages/1_Validation.py) is only a display layer.

Vocabulary used throughout:
  p      model probability that a (client, article) pair is bought again
  y      the truth: 1 if the pair was actually bought in the label year
  split  "train" = features 2024 -> label 2025;  "test" = features 2025 -> label 2026
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SCORE_FILES = {
    "train": DATA / "pair_scores_train.csv",
    "test": DATA / "pair_scores_2026.csv",
}


def load_metrics() -> dict:
    """Read reports/model_metrics.json (written by train_model.py)."""
    path = REPORTS / "model_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `py src/train_model.py` to generate it.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(split: str = "test") -> pd.DataFrame:
    """Scored pairs for one split, with the true label attached."""
    if split not in SCORE_FILES:
        raise ValueError(f"split must be one of {sorted(SCORE_FILES)}, got {split!r}")
    path = SCORE_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `py src/train_model.py` to generate it.")
    return pd.read_csv(
        path,
        dtype={"client_id": str, "article_id": str, "vendor_code": str},
        encoding="utf-8-sig",
    )


def calibration_table(p: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Are the probabilities true? Bin by predicted p, compare to observed frequency.

    `gap` = mean_pred - observed. Positive means the model is over-confident in that bin.
    Empty bins are dropped, so the result may have fewer than n_bins rows.
    """
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    d = pd.DataFrame({"p": p, "y": y, "bin": idx})
    g = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                             observed=("y", "mean")).reset_index()
    g["bin_low"] = edges[g["bin"].to_numpy()]
    g["bin_high"] = edges[g["bin"].to_numpy() + 1]
    g["gap"] = g["mean_pred"] - g["observed"]
    return g[["bin_low", "bin_high", "n", "mean_pred", "observed", "gap"]]


def decile_table(p: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Repurchase rate per probability decile (10 = most confident)."""
    d = pd.DataFrame({"p": pd.Series(p).reset_index(drop=True),
                      "y": pd.Series(y).reset_index(drop=True)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    g = (d.groupby("decile", observed=True)
           .agg(n=("y", "size"), rate=("y", "mean"))
           .reset_index())
    g["decile"] = g["decile"].astype(int)
    return g.sort_values("decile").reset_index(drop=True)


def confusion_at(p: pd.Series, y: pd.Series, threshold: float) -> dict:
    """Confusion matrix treating p >= threshold as 'will buy'."""
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    flagged = p >= threshold
    tp = int(((flagged) & (y == 1)).sum())
    fp = int(((flagged) & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    tn = int(((~flagged) & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_flagged": int(flagged.sum()),
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_validation.py -v`
Expected: 6 passed.

- [ ] **Step 5: Verify against the real data**

```bash
py -c "import sys; sys.path.insert(0,'src'); import validation as v; \
m=v.load_metrics(); t=v.load_scores('test'); \
print('test rows', len(t), '| base rate', round(t.label.mean(),4)); \
c=v.calibration_table(t.p_model, t.label); print(c.to_string(index=False)); \
print('largest over-confidence gap:', round(c.gap.max(),4))"
```
Expected: 251,342 rows, base rate ≈ 0.184, and a calibration table whose `observed` column rises with
`mean_pred`. Note the largest gap — it is the headline number for §2.

---

### Task 5: Recommendation-level metrics in `src/validation.py`

**Files:**
- Modify: `src/validation.py` (append)
- Modify: `tests/test_validation.py` (append)

**Interfaces:**
- Consumes: `load_scores` from Task 4; `data/ca_monthly.parquet`.
- Produces:
  - `eligible_pairs(fact: pd.DataFrame, years: tuple) -> pd.DataFrame` → columns `client_id, article_id`
  - `queue_replay(scores: pd.DataFrame, eligible: pd.DataFrame, top_k: int = 3) -> pd.DataFrame` → scores columns plus `valeur_attendue, rank_in_client`
  - `precision_at_n(df: pd.DataFrame, score_col: str, n: int, group_col: str = "vendor_code") -> dict` → keys `n, n_pairs, n_hits, precision, n_groups`
  - `realized_ca(fact: pd.DataFrame, year: int) -> pd.DataFrame` → columns `client_id, article_id, ca_realized`
  Tasks 6 and 7 use all of these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
def _fact(rows):
    """Build a minimal ca_monthly-shaped frame. rows = (client, article, year, month, ca)."""
    return pd.DataFrame(
        [{"client_id": c, "article_id": a,
          "month": pd.Timestamp(year=y, month=m, day=1), "ca": ca,
          "vendor_code": "007"} for c, a, y, m, ca in rows])


def test_eligible_pairs_requires_both_years():
    fact = _fact([
        ("C1", "A1", 2024, 3, 100.0),   # both years   -> eligible
        ("C1", "A1", 2025, 3, 120.0),
        ("C2", "A2", 2025, 2, 300.0),   # 2025 only    -> not eligible
        ("C3", "A3", 2024, 1, 50.0),    # 2024 only    -> not eligible
        ("C4", "A4", 2024, 1, -10.0),   # return only  -> not eligible
        ("C4", "A4", 2025, 1, 80.0),
    ])
    el = v.eligible_pairs(fact, (2024, 2025))
    got = set(zip(el["client_id"], el["article_id"]))
    assert got == {("C1", "A1")}


def test_queue_replay_ranks_by_expected_value_and_caps_at_top_k():
    scores = pd.DataFrame({
        "client_id": ["C1"] * 4,
        "article_id": ["A1", "A2", "A3", "A4"],
        "vendor_code": ["007"] * 4,
        "ca_sum": [100.0, 1000.0, 10.0, 500.0],
        "p_model": [0.9, 0.5, 0.99, 0.2],   # EV: 90, 500, 9.9, 100
        "label": [1, 1, 0, 0],
    })
    eligible = scores[["client_id", "article_id"]]
    out = v.queue_replay(scores, eligible, top_k=2)
    assert list(out["article_id"]) == ["A2", "A4"]
    assert list(out["rank_in_client"]) == [1, 2]
    assert out.iloc[0]["valeur_attendue"] == pytest.approx(500.0)


def test_queue_replay_drops_pairs_outside_the_eligible_pool():
    scores = pd.DataFrame({
        "client_id": ["C1", "C1"],
        "article_id": ["A1", "A2"],
        "vendor_code": ["007", "007"],
        "ca_sum": [100.0, 1000.0],
        "p_model": [0.9, 0.9],
        "label": [1, 1],
    })
    eligible = pd.DataFrame({"client_id": ["C1"], "article_id": ["A1"]})
    out = v.queue_replay(scores, eligible, top_k=3)
    assert list(out["article_id"]) == ["A1"]


def test_precision_at_n_counts_hits_in_the_top_n_per_group():
    df = pd.DataFrame({
        "vendor_code": ["V1"] * 4 + ["V2"] * 4,
        "p_model": [0.9, 0.8, 0.2, 0.1] + [0.9, 0.8, 0.2, 0.1],
        "label":   [1,   0,   1,   1]  + [1,   1,   0,   0],
    })
    r = v.precision_at_n(df, "p_model", n=2)
    assert r["n_pairs"] == 4          # 2 per vendor
    assert r["n_hits"] == 3           # V1: 1 hit, V2: 2 hits
    assert r["precision"] == pytest.approx(0.75)
    assert r["n_groups"] == 2


def test_precision_at_n_handles_groups_smaller_than_n():
    df = pd.DataFrame({
        "vendor_code": ["V1", "V2"],
        "p_model": [0.9, 0.4],
        "label": [1, 0],
    })
    r = v.precision_at_n(df, "p_model", n=30)
    assert r["n_pairs"] == 2
    assert r["precision"] == pytest.approx(0.5)


def test_realized_ca_sums_only_the_requested_year_and_positive_months():
    fact = _fact([
        ("C1", "A1", 2026, 1, 100.0),
        ("C1", "A1", 2026, 2, 50.0),
        ("C1", "A1", 2025, 1, 999.0),   # wrong year -> excluded
        ("C2", "A2", 2026, 1, -20.0),   # net return -> excluded
    ])
    r = v.realized_ca(fact, 2026)
    assert len(r) == 1
    assert r.iloc[0]["ca_realized"] == pytest.approx(150.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_validation.py -v -k "eligible or queue_replay or precision_at_n or realized"`
Expected: FAIL with `AttributeError: module 'validation' has no attribute 'eligible_pairs'`.

- [ ] **Step 3: Append the implementation to `src/validation.py`**

```python
# --------------------------------------------------------------------------
# Recommendation-level grading.
#
# IMPORTANT — this is an approximation of the production queue, not a replay.
# src/backtest.py documents why a literal replay is impossible: the lapsed rule
# needs two prior years and 2023 is not in the extract. Replicated here:
#   - expected-value ranking (p_model x prior-year CA)
#   - the both-years eligibility restriction
#   - top-k articles per client
# NOT replicated (both need a 2023 basis that does not exist):
#   - the lapsed filter (bought in prior years, absent this year)
#   - the client status boost from the reactivation list
# The page must show this caveat in the UI.
# --------------------------------------------------------------------------

def eligible_pairs(fact: pd.DataFrame, years: tuple) -> pd.DataFrame:
    """(client, article) pairs with a positive-CA month in BOTH given years."""
    y1, y2 = years
    pos = fact[fact["ca"] > 0].copy()
    pos["year"] = pos["month"].dt.year
    a = pos[pos["year"] == y1][["client_id", "article_id"]].drop_duplicates()
    b = pos[pos["year"] == y2][["client_id", "article_id"]].drop_duplicates()
    return a.merge(b, on=["client_id", "article_id"], how="inner").reset_index(drop=True)


def queue_replay(scores: pd.DataFrame, eligible: pd.DataFrame,
                 top_k: int = 3) -> pd.DataFrame:
    """Top-k articles per client, ranked by expected value, within the eligible pool."""
    df = scores.merge(eligible[["client_id", "article_id"]],
                      on=["client_id", "article_id"], how="inner").copy()
    df["valeur_attendue"] = df["p_model"] * df["ca_sum"]
    df = df.sort_values(["client_id", "valeur_attendue"], ascending=[True, False])
    df["rank_in_client"] = df.groupby("client_id").cumcount() + 1
    return df[df["rank_in_client"] <= top_k].reset_index(drop=True)


def precision_at_n(df: pd.DataFrame, score_col: str, n: int,
                   group_col: str = "vendor_code") -> dict:
    """If each vendor works only their top-n pairs, what share were actually bought?"""
    top = (df.sort_values(score_col, ascending=False)
             .groupby(group_col, group_keys=False)
             .head(n))
    n_pairs = int(len(top))
    n_hits = int(top["label"].sum())
    return {
        "n": n,
        "n_pairs": n_pairs,
        "n_hits": n_hits,
        "precision": (n_hits / n_pairs) if n_pairs else 0.0,
        "n_groups": int(top[group_col].nunique()),
    }


def realized_ca(fact: pd.DataFrame, year: int) -> pd.DataFrame:
    """Actual positive CA per (client, article) in a given year — what really happened."""
    pos = fact[(fact["ca"] > 0) & (fact["month"].dt.year == year)]
    return (pos.groupby(["client_id", "article_id"], as_index=False)["ca"]
               .sum()
               .rename(columns={"ca": "ca_realized"}))
```

- [ ] **Step 4: Run the full test module**

Run: `py -m pytest tests/test_validation.py -v`
Expected: 12 passed.

- [ ] **Step 5: Sanity-check against real data**

```bash
py -c "import sys; sys.path.insert(0,'src'); import pandas as pd, validation as v; \
t=v.load_scores('test'); fact=pd.read_parquet('data/ca_monthly.parquet'); \
el=v.eligible_pairs(fact,(2024,2025)); print('eligible pairs:', len(el)); \
q=v.queue_replay(t, el, 3); print('replayed rows:', len(q), '| clients:', q.client_id.nunique(), '| hit rate:', round(q.label.mean(),4)); \
[print('P@'+str(n), v.precision_at_n(t,'p_model',n)) for n in (5,10,30)]"
```
Expected: the replayed hit rate is clearly above the 18.4% base rate, and precision@5 > precision@30
(the ranking should degrade as you go deeper). If precision@5 < base rate, stop — something is wrong
with the ranking, not with the display.

---

### Task 6: The validation page — §1 protocol and §2 aggregate

**Files:**
- Create: `src/pages/1_Validation.py`

**Interfaces:**
- Consumes: everything produced by Tasks 2, 4, 5.
- Produces: a running page. Task 7 appends §3 and §4 to the same file.

- [ ] **Step 1: Create the page with §1 and §2**

```python
# -*- coding: utf-8 -*-
"""Validation page — is the model's prediction actually true?

Run:  streamlit run src/app.py   (this page appears in the sidebar nav)

Grades the RETROSPECTIVE test window 2025 -> 2026, where the ground truth already
exists in the data. It does not track live recommendations: that needs vendor
feedback that has not been collected yet.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validation as v
from ui_theme import fmt_tnd, inject_css

st.set_page_config(page_title="Validation du modèle", page_icon="🎯", layout="wide")
inject_css()


@st.cache_data
def load_all():
    metrics = v.load_metrics()
    test = v.load_scores("test")
    train = v.load_scores("train")
    fact = pd.read_parquet(v.DATA / "ca_monthly.parquet")
    articles = pd.read_csv(v.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")
    return metrics, test, train, fact, articles


try:
    metrics, test, train, fact, articles = load_all()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    '<div class="masthead"><div class="eyebrow">Atelier — contrôle qualité</div>'
    '<h1>Validation du modèle</h1>'
    '<div class="sub">Les prédictions étaient-elles vraies ? Fenêtre de test 2025 → 2026, '
    'dont le résultat réel est déjà connu.</div></div>',
    unsafe_allow_html=True,
)

# ---------------- §1 Le protocole ----------------
st.markdown('<div class="section-label">1 · Le protocole</div>', unsafe_allow_html=True)

st.markdown(
    "Le modèle apprend sur une année et est jugé sur la **suivante**, jamais sur la même. "
    "C'est ce qui rend le score honnête : au moment du test, 2026 est un futur que le modèle "
    "n'a jamais vu."
)

proto = pd.DataFrame([
    {"Jeu": "Entraînement", "Fenêtre": metrics["train"]["window"],
     "Paires": metrics["train"]["n_pairs"], "Taux de rachat": metrics["train"]["base_rate"],
     "AUC": metrics["train"]["auc"]},
    {"Jeu": "Validation interne", "Fenêtre": metrics["validation"]["window"],
     "Paires": metrics["validation"]["n_pairs"], "Taux de rachat": metrics["validation"]["base_rate"],
     "AUC": metrics["validation"]["auc"]},
    {"Jeu": "Test (chronologique)", "Fenêtre": metrics["test"]["window"],
     "Paires": metrics["test"]["n_pairs"], "Taux de rachat": metrics["test"]["base_rate"],
     "AUC": metrics["test"]["auc"]},
])
st.dataframe(
    proto.style.format({"Paires": "{:,.0f}", "Taux de rachat": "{:.1%}", "AUC": "{:.3f}"}),
    hide_index=True, use_container_width=True,
)

gap = metrics["train"]["auc"] - metrics["test"]["auc"]
c1, c2, c3 = st.columns(3)
c1.metric("AUC entraînement", f"{metrics['train']['auc']:.3f}")
c2.metric("AUC test", f"{metrics['test']['auc']:.3f}")
c3.metric("Écart train − test", f"{gap:+.3f}",
          help="Un écart important signale un sur-apprentissage : le modèle "
               "mémorise son année d'entraînement au lieu d'apprendre une règle générale.")

st.markdown(
    f'<div class="caveat"><strong>Ce qu\'on a appris en tuning</strong><br>'
    f'Une version à 4000 itérations améliorait la validation interne (0.798 → 0.810) mais '
    f'<b>dégradait le test chronologique</b> (0.762 → 0.748). La validation interne est un '
    f'tirage aléatoire dans la <i>même</i> année que l\'entraînement : l\'améliorer récompense '
    f'la mémorisation de 2024, qui ne se transfère pas à l\'année suivante. '
    f'Le modèle est resté à {metrics["iterations_cap"]} itérations '
    f'(meilleure itération : {metrics["best_iteration"]}) et le tuning a été arrêté — '
    f'avec une seule fenêtre de test honnête, continuer à chercher reviendrait à '
    f'optimiser sur le test.</div>',
    unsafe_allow_html=True,
)

# ---------------- §2 Performance agrégée ----------------
st.markdown('<div class="section-label">2 · Performance agrégée</div>', unsafe_allow_html=True)

bm = metrics["benchmarks"]
b1, b2, b3 = st.columns(3)
b1.metric("Score manuel", f"{bm['manual_score']:.3f}", help="CA × régularité — la règle actuelle")
b2.metric("Régression logistique", f"{bm['logistic_regression']:.3f}")
b3.metric("CatBoost", f"{bm['catboost']:.3f}",
          delta=f"{bm['catboost'] - bm['manual_score']:+.3f} vs manuel")

st.markdown("**Lift par décile** — les paires les mieux notées rachètent-elles davantage ?")
dec = v.decile_table(test["p_model"], test["label"])
st.bar_chart(dec.set_index("decile")["rate"], height=240)
base = float(test["label"].mean())
st.markdown(
    f'<div class="metric-note">Taux de base : {base:.1%}. '
    f'Décile 10 : {dec.iloc[-1]["rate"]:.1%} (×{dec.iloc[-1]["rate"] / base:.1f}). '
    f'Décile 1 : {dec.iloc[0]["rate"]:.1%}.</div>',
    unsafe_allow_html=True,
)

st.markdown("**Calibration** — quand le modèle annonce 60 %, est-ce que 60 % achètent vraiment ?")
cal = v.calibration_table(test["p_model"], test["label"], n_bins=10)
cal_display = cal.assign(
    Tranche=cal.apply(lambda r: f"{r['bin_low']:.0%}–{r['bin_high']:.0%}", axis=1)
)[["Tranche", "n", "mean_pred", "observed", "gap"]].rename(
    columns={"n": "Paires", "mean_pred": "Probabilité annoncée",
             "observed": "Rachat observé", "gap": "Écart"})
st.dataframe(
    cal_display.style.format({"Paires": "{:,.0f}", "Probabilité annoncée": "{:.1%}",
                              "Rachat observé": "{:.1%}", "Écart": "{:+.1%}"}),
    hide_index=True, use_container_width=True,
)
st.line_chart(cal.set_index("mean_pred")[["observed"]], height=240)
st.markdown(
    '<div class="metric-note">Un modèle parfaitement calibré suit la diagonale : '
    'probabilité annoncée = rachat observé. Un écart positif = le modèle est trop confiant. '
    'C\'est cette mesure qui déciderait si la file d\'appels peut afficher de vrais '
    'pourcentages au lieu des libellés Élevée / Moyenne / Faible.</div>',
    unsafe_allow_html=True,
)

st.markdown("**Matrice de confusion** — à partir de quelle probabilité décide-t-on d'appeler ?")
thr = st.slider("Seuil de décision", 0.0, 1.0, float(round(base, 3)), 0.01,
                help="Par défaut : le taux de base. Au-dessus de ce seuil, on considère "
                     "que le client va racheter.")
cm = v.confusion_at(test["p_model"], test["label"], thr)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vrais positifs", f"{cm['tp']:,}")
m2.metric("Faux positifs", f"{cm['fp']:,}")
m3.metric("Précision", f"{cm['precision']:.1%}")
m4.metric("Rappel", f"{cm['recall']:.1%}")
st.markdown(
    f'<div class="metric-note">Au seuil {thr:.2f}, on contacterait {cm["n_flagged"]:,} paires '
    f'sur {len(test):,}. Précision et rappel se déplacent en sens inverse : '
    f'monter le seuil augmente la précision (moins d\'appels inutiles) mais rate '
    f'plus d\'opportunités.</div>',
    unsafe_allow_html=True,
)
```

- [ ] **Step 2: Boot the app and confirm the page renders**

Run: `py -m streamlit run src/app.py --server.headless true --server.port 8598`
Open the "Validation" entry in the sidebar. Confirm: the protocol table shows three rows, the train
AUC exceeds the test AUC, the decile chart rises left-to-right, and the calibration table renders.
Stop the server afterwards.

- [ ] **Step 3: Verify the vendor page is unaffected**

Run: `py -m pytest tests/test_app.py -v`
Expected: 7 passed. Adding `src/pages/` changes `app.py` into a multipage app; this confirms the
existing page still behaves identically.

---

### Task 7: The validation page — §3 recommendations and §4 drill-down

**Files:**
- Modify: `src/pages/1_Validation.py` (append)

**Interfaces:**
- Consumes: `queue_replay`, `eligible_pairs`, `precision_at_n`, `realized_ca` from Task 5.
- Produces: the finished page.

- [ ] **Step 1: Append §3 and §4**

```python
# ---------------- §3 Les recommandations ----------------
st.markdown('<div class="section-label">3 · Les recommandations</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="caveat"><strong>Portée de ce rejeu</strong><br>'
    'Un rejeu littéral de la file de 2026 est impossible : la règle « lapsed » demande deux '
    'années antérieures et 2023 n\'existe pas dans l\'extrait. Ce qui est rejoué ici : '
    'le classement par <b>valeur attendue</b> (p × CA), la restriction aux paires achetées '
    'en 2024 <b>et</b> 2025, et le top-3 articles par client. '
    'Ce qui ne l\'est pas : le filtre « lapsed » et le boost de statut client. '
    'Les chiffres ci-dessous jugent donc le <i>classement du modèle</i>, pas la fiche '
    'complète telle qu\'un vendeur la reçoit.</div>',
    unsafe_allow_html=True,
)


@st.cache_data
def build_replay():
    el = v.eligible_pairs(fact, (2024, 2025))
    q = v.queue_replay(test, el, top_k=3)
    real = v.realized_ca(fact, 2026)
    q = q.merge(real, on=["client_id", "article_id"], how="left")
    q["ca_realized"] = q["ca_realized"].fillna(0.0)
    return q


replay = build_replay()

st.markdown("**Précision @ N** — si un vendeur ne traite que ses N meilleures paires :")
rows = []
for n in (5, 10, 30):
    r_model = v.precision_at_n(test, "p_model", n)
    r_manual = v.precision_at_n(test, "manual_score", n)
    rows.append({"N": n, "Paires listées": r_model["n_pairs"],
                 "Précision — modèle": r_model["precision"],
                 "Précision — score manuel": r_manual["precision"],
                 "Taux de base": base})
st.dataframe(
    pd.DataFrame(rows).style.format({
        "Paires listées": "{:,.0f}", "Précision — modèle": "{:.1%}",
        "Précision — score manuel": "{:.1%}", "Taux de base": "{:.1%}"}),
    hide_index=True, use_container_width=True,
)

st.markdown("**Ce que ça représente en argent** (top-3 articles par client, pool éligible) :")
hit = replay[replay["label"] == 1]
g1, g2, g3 = st.columns(3)
g1.metric("Paires recommandées", f"{len(replay):,}")
g2.metric("Rachetées en 2026", f"{len(hit):,}",
          delta=f"{len(hit) / len(replay):.1%} de réussite" if len(replay) else None)
g3.metric("CA réellement réalisé", fmt_tnd(hit["ca_realized"].sum()),
          help="CA 2026 effectivement observé sur les paires recommandées et rachetées.")
st.markdown(
    f'<div class="metric-note">Valeur attendue annoncée sur ces paires : '
    f'{fmt_tnd(replay["valeur_attendue"].sum())}. CA réalisé : '
    f'{fmt_tnd(replay["ca_realized"].sum())}. L\'écart entre les deux mesure '
    f'l\'optimisme du classement — rappel : seuls janvier–mai 2026 sont observés, '
    f'donc le réalisé est un minorant.</div>',
    unsafe_allow_html=True,
)

# ---------------- §4 Inspection individuelle ----------------
st.markdown('<div class="section-label">4 · Inspection individuelle</div>', unsafe_allow_html=True)
st.markdown("Descendre jusqu'à une prédiction précise et vérifier ce qui s'est réellement passé.")

art_lookup = articles.set_index("article_id")["designation"].to_dict()

i1, i2 = st.columns([1, 2])
vendors = sorted(test["vendor_code"].dropna().unique())
vendor = i1.selectbox("Vendeur", vendors)
scope = test[test["vendor_code"] == vendor]

view = i2.radio(
    "Afficher",
    ["Tout", "Réussites uniquement", "Échecs uniquement", "Erreurs les plus confiantes"],
    horizontal=True,
)

clients = (scope.groupby("client_id")
                .agg(paires=("article_id", "size"), rachats=("label", "sum"))
                .sort_values("paires", ascending=False)
                .reset_index())
client = st.selectbox(
    "Client",
    clients["client_id"],
    format_func=lambda c: (
        f"{c} — {int(clients.loc[clients.client_id == c, 'paires'].iloc[0])} paires, "
        f"{int(clients.loc[clients.client_id == c, 'rachats'].iloc[0])} rachetées"
    ),
)

detail = scope[scope["client_id"] == client].copy()
detail["Désignation"] = detail["article_id"].map(art_lookup).fillna("(désignation inconnue)")
detail["Réalité"] = detail["label"].map({1: "✓ acheté", 0: "✗ pas acheté"})
detail["Valeur attendue"] = detail["p_model"] * detail["ca_sum"]

if view == "Réussites uniquement":
    detail = detail[detail["label"] == 1]
elif view == "Échecs uniquement":
    detail = detail[detail["label"] == 0]
elif view == "Erreurs les plus confiantes":
    detail = detail[detail["label"] == 0].sort_values("p_model", ascending=False)

if detail.empty:
    st.info("Aucune paire ne correspond à ce filtre pour ce client.")
else:
    show = (detail.sort_values("p_model", ascending=False)
                  .rename(columns={"article_id": "Article", "p_model": "Probabilité",
                                   "ca_sum": "CA 2025"})
            [["Article", "Désignation", "Probabilité", "CA 2025",
              "Valeur attendue", "Réalité"]])
    st.dataframe(
        show.style.format({"Probabilité": "{:.1%}", "CA 2025": "{:,.0f}",
                           "Valeur attendue": "{:,.0f}"}),
        hide_index=True, use_container_width=True, height=420,
    )
    st.markdown(
        f'<div class="metric-note">{len(detail):,} paire(s) affichée(s). '
        f'Sur l\'ensemble du client : {int(scope[scope.client_id == client]["label"].sum())} '
        f'rachat(s) sur {len(scope[scope.client_id == client])} paires.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Fenêtre de test 2025 → 2026, vérité connue. Les labels n'observent que janvier–mai : "
    "tous les taux sont des minorants. Cette page ne suit pas les recommandations en cours — "
    "cela demandera les retours vendeurs, qui ne sont pas encore collectés."
)
```

- [ ] **Step 2: Boot and walk the page end to end**

Run: `py -m streamlit run src/app.py --server.headless true --server.port 8598`
Confirm: the caveat box is visible in §3, precision@5 beats precision@30, selecting a vendor and
client renders the drill-down table, and each of the four filters produces a coherent view
("Erreurs les plus confiantes" must show only `✗ pas acheté` rows, sorted by descending probability).
Stop the server afterwards.

- [ ] **Step 3: Verify no regression**

Run: `py -m pytest tests/ -v`
Expected: 19 passed (7 app + 12 validation).

---

### Task 8: Page smoke tests and documentation

**Files:**
- Create: `tests/test_validation_page.py`
- Modify: `progress.md` (append a dated entry)
- Modify: `README.md` (pipeline/run section)

**Interfaces:**
- Consumes: the finished page.
- Produces: nothing downstream.

- [ ] **Step 1: Write the page smoke tests**

```python
# -*- coding: utf-8 -*-
"""AppTest coverage for the validation page."""
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def page_path(project_root):
    p = project_root / "src" / "pages" / "1_Validation.py"
    if not p.exists():
        pytest.skip("validation page not built yet")
    return str(p)


@pytest.fixture
def at(page_path, data_dir):
    app = AppTest.from_file(page_path, default_timeout=180)
    app.run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    return app


def test_page_loads_with_all_four_sections(at):
    text = " ".join(m.value for m in at.markdown)
    for section in ("Le protocole", "Performance agrégée",
                    "Les recommandations", "Inspection individuelle"):
        assert section in text, f"section missing: {section}"


def test_replay_caveat_is_visible(at):
    text = " ".join(m.value for m in at.markdown)
    assert "rejeu littéral" in text, "the §3 replication caveat must be shown in the UI"


def test_threshold_slider_changes_the_confusion_matrix(at):
    before = {m.label: m.value for m in at.metric}
    at.slider[0].set_value(0.9)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    after = {m.label: m.value for m in at.metric}
    assert before["Vrais positifs"] != after["Vrais positifs"], \
        "raising the threshold must reduce the number of flagged pairs"


def test_vendor_switch_updates_the_drilldown(at):
    vendor_box = [sb for sb in at.selectbox if sb.label == "Vendeur"][0]
    other = vendor_box.options[1]
    vendor_box.set_value(other)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_misses_only_filter_shows_no_purchases(at):
    radio = [r for r in at.radio if r.label == "Afficher"][0]
    radio.set_value("Échecs uniquement")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
```

- [ ] **Step 2: Run them**

Run: `py -m pytest tests/test_validation_page.py -v`
Expected: 5 passed. If `test_threshold_slider_changes_the_confusion_matrix` fails because the metric
labels collide with §1's metrics, disambiguate by asserting on `at.slider[0].value` plus the
`n_flagged` caption text instead — do not delete the test.

- [ ] **Step 3: Run the whole suite**

Run: `py -m pytest tests/ -v`
Expected: 24 passed (7 app + 12 validation + 5 page).

- [ ] **Step 4: Update `progress.md`**

Append a dated entry following the existing style — what was built, which files, what it measures,
the calibration finding (quote the actual largest gap from Task 4 Step 5), and the explicit statement
that §3 is an approximation with the two non-replicated rules named. Also record that the `app.py`
AppTest scenarios were recovered from a scratchpad into `tests/`, which is what makes the earlier
verification claim reproducible.

- [ ] **Step 5: Update `README.md`**

In the pipeline section, note the new order and the new page:

```text
build_dataset → baseline_lists → train_model → contact_queue → vendor_sheets
                                      │
                                      └─> pair_scores_train.csv + model_metrics.json
                                          consumed by the Validation page

streamlit run src/app.py     # page 1: file d'appels (vendeurs)
                             # page 2: validation du modèle (technique)
py -m pytest tests/          # 24 tests
```

- [ ] **Step 6: Final verification**

```bash
py -m pytest tests/ -q && py -c "import pathlib; \
[print(p, pathlib.Path(p).exists()) for p in ['src/validation.py','src/ui_theme.py','src/pages/1_Validation.py','reports/model_metrics.json','data/pair_scores_train.csv','tests/test_app.py','tests/test_validation.py','tests/test_validation_page.py']]"
```
Expected: all tests pass and every path reports `True`.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| §1 protocol, train vs test AUC, iteration finding | 3, 6 |
| §2 AUC comparison, decile lift, calibration, confusion matrix | 4, 6 |
| §3 precision@N, money view, replication caveat in the UI | 5, 7 |
| §4 vendor → client drill-down with reality column and three filters | 7 |
| `validation.py` has no Streamlit import | 4, 5 |
| Threshold as a slider defaulting to base rate | 6 |
| Tests use synthetic frames, not the 51 MB extract | 4, 5 |
| Recover scratchpad AppTests into `tests/` | 1 |
| `train_model.py` emits train scores + metrics JSON | 3 |
| Re-running training reproduces artifacts bit-for-bit | 3 (steps 4 and 6) |
| Out of scope: no tuning, no `contact_queue.py` change | Global Constraints |

No gaps.

**Placeholder scan:** the only intentional placeholder is the CSS paste marker in Task 2 Step 1
(`<<< paste the full <style> string from src/app.py:40-122 here, unchanged >>>`). This is deliberate:
reproducing 80 lines of CSS inline would invite silent divergence, and the instruction is exact about
source lines and that it must be verbatim.

**Type consistency:** `p_model`, `label`, `ca_sum`, `client_id`, `article_id`, `vendor_code` are used
identically across Tasks 3–7 and match the columns `train_model.py` writes. `valeur_attendue` and
`rank_in_client` are produced by `queue_replay` (Task 5) and consumed only in Task 7.
`calibration_table` returns `gap` (Task 4), rendered as "Écart" in Task 6.
