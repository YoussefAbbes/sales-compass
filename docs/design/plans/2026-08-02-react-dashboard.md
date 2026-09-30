# React Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A four-view React dashboard (overview, call queue, model validation, dataset profile) served by a FastAPI backend that reads `data/` at runtime so no confidential data ever enters the frontend bundle.

**Architecture:** FastAPI imports the already-tested `src/validation.py` and serves aggregates over `/api`. React + Vite + TypeScript consumes them through a dev proxy, so the frontend uses relative paths and needs no CORS or host configuration. Built in four vertical slices; each slice ends with runnable software.

**Tech Stack:** FastAPI, uvicorn, pandas (existing), React 18, Vite 5, TypeScript 5, React Router 6, Recharts 2, Vitest.

## Global Constraints

- **No git commits.** The user opted out for this session. Every task ends with a verification step. Do not run `git commit`, `git add`, or `git push`.
- **No confidential data under `web/`.** No client name, revenue figure, or exported dataset may be written into the frontend folder. The API is the only path from `data/` to the browser.
- **No endpoint returns a full table.** Aggregates or bounded slices only. `pair_scores_2026.csv` is 15 MB and must never cross the wire whole.
- **The API binds to `127.0.0.1` only.** Local tool, not a service.
- **Do not modify `src/app.py`, `src/pages/`, `src/contact_queue.py`, or any model logic.** The Streamlit app stays working; React is additive.
- **Reuse `src/validation.py` unchanged.** Do not reimplement calibration, decile lift, precision@N, or the queue replay.
- **Interface language is French.** Code, comments, and identifiers are English; user-visible strings are French.
- **Run Python as `py`** on this machine (`python` is not on PATH in the Bash tool).
- **Visual verification is manual.** Tests prove data correctness only; they never assert that a page looks right.

---

## File Structure

```text
api/
  __init__.py
  main.py               FastAPI app, mounts routers, binds 127.0.0.1
  deps.py               cached loaders — read parquet/CSV once per process
  routes/
    __init__.py
    overview.py         GET /api/overview
    queue.py            GET /api/vendors, /api/vendors/{code}/queue, /api/clients/{id}; POST /api/feedback
    validation.py       GET /api/validation/{protocol,aggregate,recommendations,pairs}
    dataset.py          GET /api/dataset/profile
src/
  dataset_profile.py    NEW — profiling computations, pure pandas
web/
  package.json, vite.config.ts, tsconfig.json, index.html
  src/
    main.tsx            React entry + router
    App.tsx             route table
    api.ts              typed fetch client + response types
    theme.css           design tokens + base styles
    components/
      Layout.tsx        sidebar nav + page frame
      StatTile.tsx      KPI tile
      DataTable.tsx     generic table with column formatters
      Caveat.tsx        the bordered warning box
      SectionLabel.tsx  the monospace section heading
    pages/
      Overview.tsx, Queue.tsx, Validation.tsx, Dataset.tsx
tests/
  test_api.py
  test_dataset_profile.py
```

---

# Slice 1 — API skeleton, app shell, Vue d'ensemble

### Task 1: FastAPI skeleton and the overview endpoint

**Files:**
- Create: `api/__init__.py`, `api/deps.py`, `api/main.py`, `api/routes/__init__.py`, `api/routes/overview.py`
- Modify: `requirements.txt`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `src/validation.py` (`load_metrics`, `load_scores`), `data/clients_to_contact.csv`, `data/ca_monthly.parquet`.
- Produces: `api.main.app` (FastAPI instance) used by every later task; `api.deps.queue_df()`, `api.deps.fact_df()`, `api.deps.articles_df()`, each returning a cached `pd.DataFrame`.

- [ ] **Step 1: Add the dependencies**

Append to `requirements.txt`:

```text
fastapi>=0.115
uvicorn[standard]>=0.32
httpx>=0.27
```

Run: `py -m pip install -r requirements.txt`

`httpx` is required by FastAPI's `TestClient`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_api.py`:

```python
# -*- coding: utf-8 -*-
"""Endpoint coverage. Every test asserts a real value, not just a 200."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def queue(project_root):
    return pd.read_csv(project_root / "data" / "clients_to_contact.csv",
                       dtype={"vendor_code": str, "client_id": str},
                       encoding="utf-8-sig")


def test_overview_totals_match_the_queue_file(queue):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["n_clients"] == len(queue)
    assert body["n_vendors"] == queue["vendor_code"].nunique()
    assert body["tnd_at_stake"] == pytest.approx(
        float(queue["priority_score"].sum()), rel=1e-6)


def test_overview_monthly_series_has_the_15_observed_months():
    body = client.get("/api/overview").json()
    months = [m["month"] for m in body["monthly"]]
    assert len(months) == 15
    assert months[0] == "2024-01"
    assert months[-1] == "2026-05"
    # The June-December gap must be absent, not interpolated.
    assert "2024-06" not in months


def test_overview_top_clients_are_sorted_by_priority():
    body = client.get("/api/overview").json()
    scores = [c["priority_score"] for c in body["top_clients"]]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) <= 10
```

- [ ] **Step 3: Run to verify it fails**

Run: `py -m pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 4: Create the cached loaders**

Create `api/__init__.py` (empty) and `api/routes/__init__.py` (empty).

Create `api/deps.py`:

```python
# -*- coding: utf-8 -*-
"""Cached data access. Files are read once per process, not once per request.

Everything here reads from data/, which is gitignored because it contains real
client names and revenue. Nothing in this module writes into web/.
"""
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT / "src"))


@lru_cache(maxsize=1)
def queue_df() -> pd.DataFrame:
    return pd.read_csv(DATA / "clients_to_contact.csv",
                       dtype={"vendor_code": str, "client_id": str},
                       encoding="utf-8-sig")


@lru_cache(maxsize=1)
def fact_df() -> pd.DataFrame:
    return pd.read_parquet(DATA / "ca_monthly.parquet")


@lru_cache(maxsize=1)
def articles_df() -> pd.DataFrame:
    return pd.read_csv(DATA / "articles.csv", dtype=str, encoding="utf-8-sig")


@lru_cache(maxsize=1)
def vendors_df() -> pd.DataFrame:
    return pd.read_csv(DATA / "vendors.csv", dtype=str, encoding="utf-8-sig")
```

- [ ] **Step 5: Create the overview route**

Create `api/routes/overview.py`:

```python
# -*- coding: utf-8 -*-
"""Executive overview: KPI tiles, monthly revenue series, cross-vendor top clients."""
from fastapi import APIRouter

import validation as v

from ..deps import fact_df, queue_df

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def overview() -> dict:
    q = queue_df()
    fact = fact_df()

    monthly = (fact.groupby(fact["month"].dt.strftime("%Y-%m"))["ca"]
                   .sum().reset_index(name="ca").sort_values("month"))

    try:
        auc = v.load_metrics()["test"]["auc"]
    except FileNotFoundError:
        auc = None

    top = (q.sort_values("priority_score", ascending=False)
             .head(10)[["client_id", "client_name", "vendor_code", "vendor_name",
                        "status", "priority_score", "resume"]])

    return {
        "n_clients": int(len(q)),
        "n_vendors": int(q["vendor_code"].nunique()),
        "tnd_at_stake": float(q["priority_score"].sum()),
        "test_auc": auc,
        "monthly": [{"month": m, "ca": float(c)} for m, c in
                    zip(monthly["month"], monthly["ca"])],
        "top_clients": top.to_dict(orient="records"),
    }
```

- [ ] **Step 6: Create the app**

Create `api/main.py`:

```python
# -*- coding: utf-8 -*-
"""FastAPI backend for the React dashboard.

Serves aggregates from data/ so that no confidential row ever enters the frontend
bundle. Local tool: bind 127.0.0.1 only.

Run:  py -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""
from fastapi import FastAPI

from .routes import overview

app = FastAPI(title="Recommandation vendeurs — API", version="1.0")
app.include_router(overview.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 7: Run the tests**

Run: `py -m pytest tests/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 8: Verify the server boots**

Run: `py -m uvicorn api.main:app --host 127.0.0.1 --port 8000` in the background, then
`py -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/overview'))['n_clients'])"`
Expected: `1204`. Stop the server.

---

### Task 2: React scaffold, theme, layout, and the Vue d'ensemble page

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/tsconfig.node.json`, `web/index.html`
- Create: `web/src/main.tsx`, `web/src/App.tsx`, `web/src/api.ts`, `web/src/theme.css`
- Create: `web/src/components/Layout.tsx`, `StatTile.tsx`, `SectionLabel.tsx`, `DataTable.tsx`, `Caveat.tsx`
- Create: `web/src/pages/Overview.tsx`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `GET /api/overview` from Task 1.
- Produces:
  - `api.ts` exports `getJSON<T>(path: string): Promise<T>`, `fmtTND(n: number): string`, `fmtPct(n: number, digits?: number): string`, and the `Overview` type.
  - `StatTile` props: `{ label: string; value: string; hint?: string }`
  - `SectionLabel` props: `{ n: number; children: React.ReactNode }`
  - `Caveat` props: `{ title: string; children: React.ReactNode }`
  - `DataTable` props: `{ columns: Column[]; rows: Record<string, unknown>[] }` where
    `Column = { key: string; label: string; format?: (v: any, row: any) => React.ReactNode; align?: 'left'|'right' }`
  - `Layout` renders the sidebar nav and an `<Outlet />`.
  Tasks 4, 6 and 8 use all of these.

- [ ] **Step 1: Scaffold the project**

```bash
cd web && npm create vite@latest . -- --template react-ts
npm install
npm install react-router-dom recharts
```

If `npm create` refuses because the directory is non-empty, create `web/` fresh first.

- [ ] **Step 2: Configure the dev proxy**

Replace `web/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API proxy is what keeps confidential data out of the bundle: the frontend
// only ever uses relative /api paths, and there is no host to misconfigure.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
```

- [ ] **Step 3: Protect the frontend folder**

Append to `.gitignore`:

```text
# --- Frontend ---
web/node_modules/
web/dist/
```

- [ ] **Step 4: Write the theme**

Create `web/src/theme.css` with the tokens copied from `src/ui_theme.py` so the two UIs match:
`--ink: #1B1A17`, `--paper: #F6F3EC`, `--line: #E2DBCB`, `--muted: #6B655A`, `--urgent: #D64B1F`,
`--watch: #8A6A16`, `--steady: #2F6B4C`. Import Archivo, Archivo Expanded and JetBrains Mono from
Google Fonts. Define classes matching the Streamlit ones: `.masthead`, `.eyebrow`, `.section-label`,
`.badge` (+ `-urgent`/`-watch`/`-steady`), `.caveat`, `.metric-note`, `.stat-tile`, `.data-table`.
Body background `var(--paper)`, text `var(--ink)`; sidebar background `#232220`, text `#EDE8DC`.

- [ ] **Step 5: Write the API client**

Create `web/src/api.ts`:

```typescript
export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`)
  return r.json() as Promise<T>
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`)
  return r.json() as Promise<T>
}

export const fmtTND = (n: number) =>
  `${Math.round(n).toLocaleString('fr-FR')} TND`

export const fmtPct = (n: number, digits = 1) =>
  `${(n * 100).toFixed(digits)} %`

export const fmtInt = (n: number) => Math.round(n).toLocaleString('fr-FR')

export type TopClient = {
  client_id: string; client_name: string; vendor_code: string
  vendor_name: string; status: string; priority_score: number; resume: string
}

export type Overview = {
  n_clients: number; n_vendors: number; tnd_at_stake: number
  test_auc: number | null
  monthly: { month: string; ca: number }[]
  top_clients: TopClient[]
}
```

- [ ] **Step 6: Build the shared components**

`StatTile.tsx` renders a bordered card: uppercase monospace `label`, large Archivo Expanded `value`,
optional muted `hint`. `SectionLabel.tsx` renders `<div className="section-label">{n} · {children}</div>`.
`Caveat.tsx` renders `<div className="caveat"><strong>{title}</strong><br/>{children}</div>`.
`DataTable.tsx` maps `columns` over `rows`, applying `format` when present and right-aligning numeric
columns. `Layout.tsx` renders a fixed sidebar with `NavLink`s to `/`, `/file`, `/validation`,
`/donnees` and an `<Outlet/>` in the main area.

- [ ] **Step 7: Wire the router**

`App.tsx` defines routes: `/` → `Overview`, `/file` → `Queue`, `/validation` → `Validation`,
`/donnees` → `Dataset`, all nested under `Layout`. For this task, `Queue`, `Validation` and `Dataset`
are one-line placeholder components rendering `<p>À venir</p>`; Tasks 4, 6 and 8 replace them.
`main.tsx` mounts `<BrowserRouter><App/></BrowserRouter>` and imports `./theme.css`.

- [ ] **Step 8: Build the Overview page**

`Overview.tsx` fetches `/api/overview` in a `useEffect`, holds `{data, error, loading}` state, and
renders: a masthead; four `StatTile`s (clients à contacter, TND en jeu, vendeurs, AUC test); a
Recharts `<BarChart>` of `monthly` with the axis labelled by month — the June–December months are
simply absent from the data, which is correct and must not be filled; and a `DataTable` of
`top_clients` with columns client, vendeur, statut (badge), priorité (TND), résumé.

Render `error` as a visible message, never a blank page.

- [ ] **Step 9: Run both processes and verify**

Terminal 1: `py -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`
Terminal 2: `cd web && npm run dev`
Open `http://localhost:5173`. Expected: four tiles with real numbers (1 204 clients, 44 vendors),
a 15-bar chart, and a 10-row table. Confirm in the browser devtools Network tab that data arrives
from `/api/overview` and is not present in the JS bundle.

- [ ] **Step 10: Verify no confidential data leaked into the frontend**

```bash
py -c "import pathlib,re; \
hits=[p for p in pathlib.Path('web/src').rglob('*') if p.is_file() and re.search(r'(?i)\b(mr|mme)\b|TND\s*\d', p.read_text(encoding='utf-8',errors='ignore'))]; \
print('files containing name/amount patterns:', hits)"
```
Expected: `[]`. If any file matches, remove the data from it — the API is the only allowed source.

---

# Slice 2 — File d'appels

### Task 3: Queue, dossier, and feedback endpoints

**Files:**
- Create: `api/routes/queue.py`
- Modify: `api/main.py` (register the router)
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `api.deps.queue_df()`, `fact_df()` from Task 1.
- Produces:
  - `GET /api/vendors` → `[{vendor_code, vendor_name, n_clients, tnd_at_stake}]`
  - `GET /api/vendors/{code}/queue` → `[{client_id, client_name, status, priority_score, valeur_attendue, manque_a_gagner, last_activity, resume, rank}]`
  - `GET /api/clients/{client_id}` → `{client_id, client_name, vendor_code, vendor_name, status, priority_score, resume, articles: [{rank, label, reason}], suggestion, monthly: [{month, ca}]}`
  - `POST /api/feedback` body `{vendor_code, client_id, client_name, action, comment}` → `{"saved": true}`
  Task 4 consumes all four.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_vendors_list_matches_the_queue(queue):
    body = client.get("/api/vendors").json()
    assert len(body) == queue["vendor_code"].nunique()
    total = sum(v["n_clients"] for v in body)
    assert total == len(queue)


def test_vendor_queue_is_ranked_and_scoped(queue):
    code = queue["vendor_code"].iloc[0]
    body = client.get(f"/api/vendors/{code}/queue").json()
    assert len(body) == int((queue["vendor_code"] == code).sum())
    scores = [c["priority_score"] for c in body]
    assert scores == sorted(scores, reverse=True)
    assert [c["rank"] for c in body] == list(range(1, len(body) + 1))


def test_unknown_vendor_returns_404():
    assert client.get("/api/vendors/ZZZZ/queue").status_code == 404


def test_client_dossier_has_articles_and_monthly_history(queue):
    cid = queue["client_id"].iloc[0]
    body = client.get(f"/api/clients/{cid}").json()
    assert body["client_id"] == cid
    assert 1 <= len(body["articles"]) <= 3
    assert all("label" in a and "reason" in a for a in body["articles"])
    assert len(body["monthly"]) > 0
    assert all(m["month"][:4] in {"2024", "2025", "2026"} for m in body["monthly"])


def test_unknown_client_returns_404():
    assert client.get("/api/clients/NOPE").status_code == 404


def test_feedback_appends_a_row(tmp_path, monkeypatch):
    import api.routes.queue as qroute
    target = tmp_path / "feedback.csv"
    monkeypatch.setattr(qroute, "FEEDBACK", target)
    payload = {"vendor_code": "007", "client_id": "0049",
               "client_name": "TEST", "action": "Contacté", "comment": "essai"}
    assert client.post("/api/feedback", json=payload).json() == {"saved": True}
    rows = pd.read_csv(target, dtype=str, encoding="utf-8-sig")
    assert len(rows) == 1
    assert rows.iloc[0]["client_id"] == "0049"
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_api.py -v -k "vendors or client_dossier or feedback or unknown"`
Expected: FAIL with 404s from the not-yet-registered router.

- [ ] **Step 3: Implement the router**

Create `api/routes/queue.py`. It reads `clients_to_contact.csv` (already built by
`contact_queue.py` — do not recompute the ranking) and reshapes it:

```python
# -*- coding: utf-8 -*-
"""Vendor call queue: vendors, per-vendor queue, client dossier, feedback capture."""
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..deps import DATA, fact_df, queue_df

router = APIRouter(prefix="/api", tags=["queue"])
FEEDBACK = DATA / "feedback.csv"


class Feedback(BaseModel):
    vendor_code: str
    client_id: str
    client_name: str
    action: str
    comment: str = ""


@router.get("/vendors")
def vendors() -> list[dict]:
    q = queue_df()
    g = (q.groupby(["vendor_code", "vendor_name"], dropna=False)
           .agg(n_clients=("client_id", "size"),
                tnd_at_stake=("priority_score", "sum"))
           .reset_index()
           .sort_values("tnd_at_stake", ascending=False))
    return [{"vendor_code": r.vendor_code,
             "vendor_name": r.vendor_name if pd.notna(r.vendor_name) else r.vendor_code,
             "n_clients": int(r.n_clients),
             "tnd_at_stake": float(r.tnd_at_stake)} for r in g.itertuples()]


@router.get("/vendors/{code}/queue")
def vendor_queue(code: str) -> list[dict]:
    q = queue_df()
    sub = q[q["vendor_code"] == code].sort_values("priority_score", ascending=False)
    if sub.empty:
        raise HTTPException(404, f"vendeur {code} inconnu")
    cols = ["client_id", "client_name", "status", "priority_score",
            "valeur_attendue", "manque_a_gagner", "last_activity", "resume"]
    out = sub[cols].to_dict(orient="records")
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


@router.get("/clients/{client_id}")
def client_dossier(client_id: str) -> dict:
    q = queue_df()
    row = q[q["client_id"] == client_id]
    if row.empty:
        raise HTTPException(404, f"client {client_id} inconnu")
    r = row.iloc[0]

    articles = []
    for i in (1, 2, 3):
        label = r.get(f"article_txt_{i}")
        if isinstance(label, str) and label.strip():
            articles.append({"rank": i, "label": label,
                             "reason": r.get(f"reason_{i}") or ""})

    fact = fact_df()
    hist = (fact[fact["client_id"] == client_id]
            .groupby(fact["month"].dt.strftime("%Y-%m"))["ca"].sum()
            .reset_index(name="ca").sort_values("month"))

    return {
        "client_id": client_id,
        "client_name": r["client_name"],
        "vendor_code": r["vendor_code"],
        "vendor_name": r["vendor_name"] if pd.notna(r["vendor_name"]) else r["vendor_code"],
        "status": r["status"],
        "priority_score": float(r["priority_score"]),
        "resume": r["resume"],
        "suggestion": r["suggestion_complementaire"]
                      if pd.notna(r["suggestion_complementaire"]) else None,
        "articles": articles,
        "monthly": [{"month": m, "ca": float(c)}
                    for m, c in zip(hist["month"], hist["ca"])],
    }


@router.post("/feedback")
def save_feedback(fb: Feedback) -> dict:
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "vendor_code": fb.vendor_code, "client_id": fb.client_id,
        "client_name": fb.client_name, "action": fb.action, "comment": fb.comment,
    }])
    row.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(),
               index=False, encoding="utf-8-sig")
    return {"saved": True}
```

Register it in `api/main.py`: `from .routes import overview, queue` and
`app.include_router(queue.router)`.

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_api.py -v`
Expected: 9 passed.

- [ ] **Step 5: Confirm the real feedback file was untouched**

Run: `py -c "from pathlib import Path; print('feedback.csv exists:', Path('data/feedback.csv').exists())"`
Expected: `False` — the test monkeypatched the path to a temp file.

---

### Task 4: File d'appels page

**Files:**
- Create: `web/src/pages/Queue.tsx`, `web/src/components/ClientDossier.tsx`, `web/src/components/FeedbackForm.tsx`
- Modify: `web/src/api.ts` (add types), `web/src/App.tsx` (replace the placeholder)

**Interfaces:**
- Consumes: the four endpoints from Task 3; `StatTile`, `DataTable`, `SectionLabel`, `fmtTND` from Task 2.
- Produces: nothing downstream.

- [ ] **Step 1: Add the types to `api.ts`**

```typescript
export type Vendor = {
  vendor_code: string; vendor_name: string; n_clients: number; tnd_at_stake: number
}

export type QueueRow = {
  rank: number; client_id: string; client_name: string; status: string
  priority_score: number; valeur_attendue: number; manque_a_gagner: number
  last_activity: string; resume: string
}

export type Dossier = {
  client_id: string; client_name: string; vendor_code: string; vendor_name: string
  status: string; priority_score: number; resume: string; suggestion: string | null
  articles: { rank: number; label: string; reason: string }[]
  monthly: { month: string; ca: number }[]
}
```

- [ ] **Step 2: Build `FeedbackForm.tsx`**

Props: `{ vendorCode: string; clientId: string; clientName: string }`. Renders a `<select>` with the
seven actions (`Contacté`, `Devis envoyé`, `Commande`, `À rappeler`, `Refus`, `Client fermé`,
`Injoignable`), a comment `<textarea>`, and a submit button that `postJSON('/api/feedback', …)`.
On success show "Enregistré" for a few seconds; on failure show the error text. Reset the form when
`clientId` changes — otherwise a comment typed for one client can be submitted against another.

- [ ] **Step 3: Build `ClientDossier.tsx`**

Props: `{ clientId: string }`. Fetches `/api/clients/{clientId}`, renders the name, a status badge,
priority in TND, the résumé, then the recommended articles as numbered cards showing `label` and
`reason`, then the `suggestion` if present, then a Recharts `<LineChart>` of `monthly`, then
`<FeedbackForm/>`.

- [ ] **Step 4: Build `Queue.tsx`**

Fetches `/api/vendors` once. Renders a vendor `<select>`; on change fetches
`/api/vendors/{code}/queue`. Layout mirrors the Streamlit page: rank 1 as a large featured card,
ranks 2–4 as a three-card row, the rest in a `DataTable`. Selecting any client sets `openClientId`,
which renders `<ClientDossier/>` below. A search `<input>` filters the table by client name or id and
collapses the featured cards while active — a "Top" framing on an ad-hoc filter would be misleading.

Guard the empty case: if a vendor has fewer than four clients, or the search matches nothing, render
a message rather than indexing into an empty array.

- [ ] **Step 5: Verify manually**

With both processes running, open `http://localhost:5173/file`. Check: vendor switch reloads the
queue; clicking a card opens that client's dossier; the chart shows only observed months; submitting
feedback shows a confirmation. Then confirm the row landed:

```bash
py -c "import pandas as pd; d=pd.read_csv('data/feedback.csv', encoding='utf-8-sig'); print(d.tail(1).to_string())"
```
Expected: the row matches the client whose dossier was open. Delete the test row afterwards.

---

# Slice 3 — Validation du modèle

### Task 5: Validation endpoints

**Files:**
- Create: `api/routes/validation.py`
- Modify: `api/main.py`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `src/validation.py` — `load_metrics`, `load_scores`, `calibration_table`, `decile_table`, `confusion_at`, `precision_at_n`, `eligible_pairs`, `queue_replay`, `realized_ca`.
- Produces:
  - `GET /api/validation/protocol` → `{train, validation, test, benchmarks, best_iteration, iterations_cap}`
  - `GET /api/validation/aggregate?threshold=` → `{base_rate, benchmarks, deciles[], calibration[], confusion{}}`
  - `GET /api/validation/recommendations` → `{precision_at_n[], replay{}}`
  - `GET /api/validation/pairs?vendor=&client=&view=` → `{rows[], summary{}}` — `view` ∈ `all|hits|misses|confident_errors`
  Task 6 consumes all four.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_protocol_reports_train_above_test_auc():
    b = client.get("/api/validation/protocol").json()
    assert b["train"]["auc"] > b["test"]["auc"]
    assert b["test"]["n_pairs"] == 251342


def test_aggregate_matches_validation_module_directly():
    import validation as v
    t = v.load_scores("test")
    expected = v.confusion_at(t["p_model"], t["label"], 0.5)
    b = client.get("/api/validation/aggregate?threshold=0.5").json()
    assert b["confusion"]["tp"] == expected["tp"]
    assert b["confusion"]["fp"] == expected["fp"]
    assert len(b["deciles"]) == 10
    assert len(b["calibration"]) <= 10


def test_recommendations_beat_the_manual_score():
    b = client.get("/api/validation/recommendations").json()
    at5 = next(r for r in b["precision_at_n"] if r["n"] == 5)
    assert at5["model"] > at5["manual"] > b["replay"]["base_rate"]


def test_pairs_drilldown_is_scoped_and_bounded():
    import validation as v
    t = v.load_scores("test")
    vendor = t["vendor_code"].dropna().iloc[0]
    cid = t[t["vendor_code"] == vendor]["client_id"].iloc[0]
    b = client.get(f"/api/validation/pairs?vendor={vendor}&client={cid}").json()
    assert len(b["rows"]) < 5000, "drill-down must never return the whole table"
    assert all(r["client_id"] == cid for r in b["rows"])
    assert set(b["rows"][0]) >= {"article_id", "designation", "p_model",
                                 "ca_sum", "label"}


def test_pairs_misses_view_returns_only_failures():
    import validation as v
    t = v.load_scores("test")
    vendor = t["vendor_code"].dropna().iloc[0]
    cid = t[(t["vendor_code"] == vendor) & (t["label"] == 0)]["client_id"].iloc[0]
    b = client.get(
        f"/api/validation/pairs?vendor={vendor}&client={cid}&view=misses").json()
    assert all(r["label"] == 0 for r in b["rows"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_api.py -v -k "protocol or aggregate or recommendations or pairs"`
Expected: FAIL with 404.

- [ ] **Step 3: Implement the router**

Create `api/routes/validation.py`:

```python
# -*- coding: utf-8 -*-
"""Model validation endpoints.

Every number here comes from src/validation.py, which is unit-tested. This module
reshapes results into JSON; it computes no metric of its own, so the React page and
the Streamlit page can never disagree.
"""
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

import validation as v

from ..deps import articles_df, fact_df

router = APIRouter(prefix="/api/validation", tags=["validation"])


@lru_cache(maxsize=1)
def _test():
    return v.load_scores("test")


@lru_cache(maxsize=1)
def _replay():
    scores = _test()
    el = v.eligible_pairs(fact_df(), (2024, 2025))
    q = v.queue_replay(scores, el, top_k=3)
    real = v.realized_ca(fact_df(), 2026)
    q = q.merge(real, on=["client_id", "article_id"], how="left")
    q["ca_realized"] = q["ca_realized"].fillna(0.0)
    return q


@router.get("/protocol")
def protocol() -> dict:
    m = v.load_metrics()
    return {k: m[k] for k in ("train", "validation", "test", "benchmarks",
                              "best_iteration", "iterations_cap")}


@router.get("/aggregate")
def aggregate(threshold: float = Query(default=None, ge=0.0, le=1.0)) -> dict:
    t = _test()
    base = float(t["label"].mean())
    thr = base if threshold is None else threshold
    dec = v.decile_table(t["p_model"], t["label"])
    cal = v.calibration_table(t["p_model"], t["label"], n_bins=10)
    return {
        "base_rate": base,
        "n_pairs": int(len(t)),
        "benchmarks": v.load_metrics()["benchmarks"],
        "threshold": thr,
        "deciles": [{"decile": int(r.decile), "n": int(r.n), "rate": float(r.rate)}
                    for r in dec.itertuples()],
        "calibration": [{"bin_low": float(r.bin_low), "bin_high": float(r.bin_high),
                         "n": int(r.n), "mean_pred": float(r.mean_pred),
                         "observed": float(r.observed), "gap": float(r.gap)}
                        for r in cal.itertuples()],
        "confusion": v.confusion_at(t["p_model"], t["label"], thr),
    }


@router.get("/recommendations")
def recommendations() -> dict:
    t = _test()
    q = _replay()
    base = float(t["label"].mean())
    rows = []
    for n in (5, 10, 30):
        m = v.precision_at_n(t, "p_model", n)
        man = v.precision_at_n(t, "manual_score", n)
        rows.append({"n": n, "n_pairs": m["n_pairs"], "n_groups": m["n_groups"],
                     "model": m["precision"], "manual": man["precision"]})
    hit = q[q["label"] == 1]
    return {
        "precision_at_n": rows,
        "replay": {
            "base_rate": base,
            "n_pairs": int(len(q)),
            "n_clients": int(q["client_id"].nunique()),
            "n_hits": int(len(hit)),
            "hit_rate": float(q["label"].mean()) if len(q) else 0.0,
            "expected_tnd": float(q["valeur_attendue"].sum()),
            "realized_tnd": float(hit["ca_realized"].sum()),
        },
    }


@router.get("/pairs")
def pairs(vendor: str, client: str,
          view: str = Query(default="all",
                            pattern="^(all|hits|misses|confident_errors)$")) -> dict:
    t = _test()
    sub = t[(t["vendor_code"] == vendor) & (t["client_id"] == client)].copy()
    if sub.empty:
        raise HTTPException(404, "aucune paire pour ce vendeur/client")

    names = articles_df().set_index("article_id")["designation"].to_dict()
    sub["designation"] = sub["article_id"].map(names).fillna("(désignation inconnue)")
    sub["valeur_attendue"] = sub["p_model"] * sub["ca_sum"]
    summary = {"n_total": int(len(sub)), "n_hits": int(sub["label"].sum())}

    if view == "hits":
        sub = sub[sub["label"] == 1]
    elif view in ("misses", "confident_errors"):
        sub = sub[sub["label"] == 0]

    sub = sub.sort_values("p_model", ascending=False).head(500)
    cols = ["article_id", "designation", "p_model", "ca_sum",
            "valeur_attendue", "label", "client_id"]
    return {"rows": sub[cols].to_dict(orient="records"), "summary": summary}
```

Register it in `api/main.py`.

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_api.py -v`
Expected: 14 passed.

---

### Task 6: Validation page

**Files:**
- Create: `web/src/pages/Validation.tsx`
- Modify: `web/src/api.ts` (types), `web/src/App.tsx`

**Interfaces:**
- Consumes: the four validation endpoints; `StatTile`, `Caveat`, `DataTable`, `SectionLabel`.
- Produces: nothing downstream.

- [ ] **Step 1: Add the types to `api.ts`**

```typescript
export type Protocol = {
  train: { window: string; n_pairs: number; base_rate: number; auc: number }
  validation: { window: string; n_pairs: number; base_rate: number; auc: number }
  test: { window: string; n_pairs: number; base_rate: number; auc: number }
  benchmarks: { manual_score: number; logistic_regression: number; catboost: number }
  best_iteration: number; iterations_cap: number
}

export type Aggregate = {
  base_rate: number; n_pairs: number; threshold: number
  benchmarks: Protocol['benchmarks']
  deciles: { decile: number; n: number; rate: number }[]
  calibration: { bin_low: number; bin_high: number; n: number
                 mean_pred: number; observed: number; gap: number }[]
  confusion: { tp: number; fp: number; fn: number; tn: number
               n_flagged: number; precision: number; recall: number }
}

export type Recommendations = {
  precision_at_n: { n: number; n_pairs: number; n_groups: number
                    model: number; manual: number }[]
  replay: { base_rate: number; n_pairs: number; n_clients: number; n_hits: number
            hit_rate: number; expected_tnd: number; realized_tnd: number }
}

export type PairRow = {
  article_id: string; designation: string; p_model: number
  ca_sum: number; valeur_attendue: number; label: number; client_id: string
}
```

- [ ] **Step 2: Build the page in four sections**

§1 Protocole — a `DataTable` of the three splits, three `StatTile`s (train AUC, test AUC, écart), and
a `Caveat` titled "Ce qu'on a appris en tuning" carrying the 4000-iteration finding (validation
0.798→0.810 while the chronological test fell 0.762→0.748).

§2 Performance agrégée — three benchmark tiles; a Recharts `<BarChart>` of `deciles`; a
`<LineChart>` of `calibration` plotting `observed` against `mean_pred` **with a dashed diagonal
reference line** so miscalibration is visible rather than inferred; the calibration table; a
`<input type="range">` threshold slider defaulting to `base_rate` that refetches
`/api/validation/aggregate?threshold=` (debounced 200 ms) and four confusion tiles. Below the
calibration chart, state that the gap is an upper bound because labels observe only January–May.

§3 Les recommandations — a `Caveat` titled "Portée de ce rejeu" naming exactly what is and is not
replicated (replicated: expected-value ranking, both-years restriction, top-3 per client; not
replicated: the lapsed filter and the client status boost, both needing a 2023 basis); the
precision@N table with model vs manual vs base rate; three tiles for the money view.

§4 Inspection individuelle — vendor `<select>`, client `<select>`, a four-way view toggle
(`Tout` / `Réussites` / `Échecs` / `Erreurs les plus confiantes`), and a `DataTable` whose
`Réalité` column renders `✓ acheté` in `--steady` and `✗ pas acheté` in `--urgent`.

The vendor and client option lists come from `/api/vendors` and `/api/vendors/{code}/queue`, which
already exist — do not add an endpoint for them.

- [ ] **Step 3: Cross-check against Streamlit**

Open `http://localhost:5173/validation` and `streamlit run src/app.py` → Validation side by side.
Test AUC, decile-10 rate, precision@5, and the confusion counts at the default threshold must be
identical. They share `validation.py`, so any difference is a display bug in one of the two.

---

# Slice 4 — Le jeu de données

### Task 7: Dataset profile module and endpoint

**Files:**
- Create: `src/dataset_profile.py`, `api/routes/dataset.py`
- Modify: `api/main.py`
- Test: `tests/test_dataset_profile.py`, `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `api.deps.fact_df()`, `articles_df()`, `clients.csv`, `vendors.csv`.
- Produces: `dataset_profile.profile(fact, articles, clients, vendors) -> dict` with keys
  `coverage[]`, `entities{}`, `naming{}`, `sparsity{}`, `concentration{}`, `returns{}`;
  and `GET /api/dataset/profile` returning it. Task 8 consumes the endpoint.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_dataset_profile.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for src/dataset_profile.py — synthetic frames only."""
import pandas as pd
import pytest

import dataset_profile as dp


def _fact(rows):
    return pd.DataFrame(
        [{"vendor_code": v, "client_id": c, "article_id": a,
          "month": pd.Timestamp(year=y, month=m, day=1), "ca": ca}
         for v, c, a, y, m, ca in rows])


def test_coverage_lists_每_observed_month_once():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 100.0),
                  ("V1", "C1", "A1", 2024, 2, 50.0),
                  ("V1", "C2", "A2", 2024, 2, 25.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": ["A1"], "designation": ["x"]}),
                   pd.DataFrame({"client_id": ["C1"]}),
                   pd.DataFrame({"vendor_code": ["V1"]}))
    months = [c["month"] for c in p["coverage"]]
    assert months == ["2024-01", "2024-02"]
    assert p["coverage"][1]["ca"] == pytest.approx(75.0)


def test_naming_gap_counts_unnamed_entities():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 10.0),
                  ("V1", "C2", "A2", 2024, 1, 10.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": ["A1"], "designation": ["x"]}),
                   pd.DataFrame({"client_id": ["C1"]}),
                   pd.DataFrame({"vendor_code": ["V1"]}))
    assert p["naming"]["articles"]["total"] == 2
    assert p["naming"]["articles"]["named"] == 1
    assert p["naming"]["articles"]["unnamed"] == 1


def test_sparsity_reports_single_month_pairs():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 10.0),
                  ("V1", "C1", "A1", 2024, 2, 10.0),
                  ("V1", "C2", "A2", 2024, 1, 10.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": [], "designation": []}),
                   pd.DataFrame({"client_id": []}), pd.DataFrame({"vendor_code": []}))
    assert p["sparsity"]["n_pairs"] == 2
    assert p["sparsity"]["single_month_pairs"] == 1


def test_returns_are_counted_separately():
    fact = _fact([("V1", "C1", "A1", 2024, 1, 100.0),
                  ("V1", "C1", "A2", 2024, 1, -20.0)])
    p = dp.profile(fact, pd.DataFrame({"article_id": [], "designation": []}),
                   pd.DataFrame({"client_id": []}), pd.DataFrame({"vendor_code": []}))
    assert p["returns"]["n_rows"] == 1
    assert p["returns"]["total_ca"] == pytest.approx(-20.0)
```

Rename the first test to `test_coverage_lists_every_observed_month_once` — the name above contains a
non-ASCII character by accident and must be corrected before running.

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_dataset_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_profile'`.

- [ ] **Step 3: Implement `src/dataset_profile.py`**

```python
# -*- coding: utf-8 -*-
"""Profiling of the monthly CA extract — what the dataset actually contains.

Pure pandas, no Streamlit and no FastAPI, so it is unit-testable on synthetic frames.
"""
import pandas as pd


def profile(fact: pd.DataFrame, articles: pd.DataFrame,
            clients: pd.DataFrame, vendors: pd.DataFrame) -> dict:
    pos_neg = fact["ca"] < 0
    monthly = (fact.groupby(fact["month"].dt.strftime("%Y-%m"))
                   .agg(rows=("ca", "size"), ca=("ca", "sum"))
                   .reset_index().sort_values("month"))

    pair_months = fact.groupby(["client_id", "article_id"])["month"].nunique()
    n_clients = int(fact["client_id"].nunique())
    n_articles = int(fact["article_id"].nunique())
    grid = n_clients * n_articles

    art_ca = fact.groupby("article_id")["ca"].sum().sort_values(ascending=False)
    cli_ca = fact.groupby("client_id")["ca"].sum().sort_values(ascending=False)

    def _named(tx_ids: set, ref_ids: set) -> dict:
        named = len(tx_ids & ref_ids)
        return {"total": len(tx_ids), "named": named,
                "unnamed": len(tx_ids) - named}

    def _top_share(s: pd.Series, frac: float = 0.1) -> float:
        if s.empty or s.sum() == 0:
            return 0.0
        return float(s.head(max(1, int(len(s) * frac))).sum() / s.sum())

    return {
        "coverage": [{"month": m, "rows": int(r), "ca": float(c)}
                     for m, r, c in zip(monthly["month"], monthly["rows"], monthly["ca"])],
        "entities": {
            "rows": int(len(fact)),
            "vendors": int(fact["vendor_code"].nunique()),
            "clients": n_clients,
            "articles": n_articles,
            "months": int(fact["month"].nunique()),
            "total_ca": float(fact["ca"].sum()),
        },
        "naming": {
            "articles": _named(set(fact["article_id"]), set(articles.get("article_id", []))),
            "clients": _named(set(fact["client_id"]), set(clients.get("client_id", []))),
            "vendors": _named(set(fact["vendor_code"]), set(vendors.get("vendor_code", []))),
        },
        "sparsity": {
            "n_pairs": int(len(pair_months)),
            "dense_grid": int(grid),
            "fill_rate": float(len(pair_months) / grid) if grid else 0.0,
            "single_month_pairs": int((pair_months == 1).sum()),
        },
        "concentration": {
            "top10pct_articles_share": _top_share(art_ca),
            "top10pct_clients_share": _top_share(cli_ca),
        },
        "returns": {
            "n_rows": int(pos_neg.sum()),
            "share": float(pos_neg.mean()) if len(fact) else 0.0,
            "total_ca": float(fact.loc[pos_neg, "ca"].sum()),
        },
    }
```

- [ ] **Step 4: Run the unit tests**

Run: `py -m pytest tests/test_dataset_profile.py -v`
Expected: 4 passed.

- [ ] **Step 5: Add the endpoint**

Create `api/routes/dataset.py`:

```python
# -*- coding: utf-8 -*-
"""Dataset profile — what the extract contains and where its limits are."""
import pandas as pd
from fastapi import APIRouter

import dataset_profile as dp

from ..deps import DATA, articles_df, fact_df, vendors_df

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/profile")
def profile() -> dict:
    clients = pd.read_csv(DATA / "clients.csv", dtype=str, encoding="utf-8-sig")
    return dp.profile(fact_df(), articles_df(), clients, vendors_df())
```

Register it in `api/main.py`.

- [ ] **Step 6: Add the endpoint test**

Append to `tests/test_api.py`:

```python
def test_dataset_profile_matches_the_real_extract():
    b = client.get("/api/dataset/profile").json()
    assert b["entities"]["rows"] == 896312
    assert b["entities"]["months"] == 15
    assert b["entities"]["articles"] == 40321
    assert b["naming"]["articles"]["unnamed"] == 15638
    assert 0.0 < b["sparsity"]["fill_rate"] < 0.01
    assert b["returns"]["n_rows"] == 35111
```

Run: `py -m pytest tests/test_api.py -v`
Expected: 15 passed.

---

### Task 8: Dataset page and documentation

**Files:**
- Create: `web/src/pages/Dataset.tsx`
- Modify: `web/src/api.ts`, `web/src/App.tsx`, `README.md`, `progress.md`

**Interfaces:**
- Consumes: `GET /api/dataset/profile`.
- Produces: nothing downstream.

- [ ] **Step 1: Add the type to `api.ts`**

```typescript
export type NamingGap = { total: number; named: number; unnamed: number }

export type DatasetProfile = {
  coverage: { month: string; rows: number; ca: number }[]
  entities: { rows: number; vendors: number; clients: number; articles: number
              months: number; total_ca: number }
  naming: { articles: NamingGap; clients: NamingGap; vendors: NamingGap }
  sparsity: { n_pairs: number; dense_grid: number; fill_rate: number
              single_month_pairs: number }
  concentration: { top10pct_articles_share: number; top10pct_clients_share: number }
  returns: { n_rows: number; share: number; total_ca: number }
}
```

- [ ] **Step 2: Build `Dataset.tsx`**

Sections: entity tiles (rows, vendors, clients, articles, months, total CA); a coverage
`<BarChart>` over the 15 observed months **with the seven missing months of each year rendered as
empty slots**, so the June–December hole is visually obvious rather than hidden by a continuous axis;
a naming-gap table (articles 38.8% unnamed, clients, vendors); sparsity and concentration tiles; and
a `Caveat` titled "Mai 2026 — à confirmer" noting that May 2026 has 29% less revenue than May 2025
while April 2026 was near a record, which reads like a mid-month extract and is an open question for
the company, not an established fact.

- [ ] **Step 3: Run the whole suite**

Run: `py -m pytest tests/ -q`
Expected: 39 passed (24 existing + 15 new).

- [ ] **Step 4: Add a single dev command**

Create `run-dev.ps1` at the repo root:

```powershell
# Starts the API and the Vite dev server together.
Start-Process -FilePath "py" -ArgumentList "-m","uvicorn","api.main:app","--reload","--host","127.0.0.1","--port","8000"
Set-Location web
npm run dev
```

- [ ] **Step 5: Update `README.md`**

Document the dashboard: `.\run-dev.ps1`, the four views, and the statement that the API is the only
path from `data/` to the browser and binds to `127.0.0.1` only.

- [ ] **Step 6: Update `progress.md`**

Append a dated entry: what was built, the API surface, the reuse of `validation.py` (so the React and
Streamlit numbers cannot diverge), the confidentiality reasoning behind choosing a backend over a
static export, the test count, and an explicit note that visual verification is manual and was or was
not performed.

- [ ] **Step 7: Final verification**

```bash
py -m pytest tests/ -q && py -c "import pathlib; \
[print(('  OK  ' if pathlib.Path(p).exists() else 'MISSING'), p) for p in \
['api/main.py','api/deps.py','api/routes/overview.py','api/routes/queue.py', \
 'api/routes/validation.py','api/routes/dataset.py','src/dataset_profile.py', \
 'web/src/pages/Overview.tsx','web/src/pages/Queue.tsx', \
 'web/src/pages/Validation.tsx','web/src/pages/Dataset.tsx','run-dev.ps1']]"
```
Expected: all tests pass and every path reports `OK`.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| FastAPI backend, no data in bundle | 1 (route), 2 step 10 (verified) |
| Vite proxy, no CORS | 2 |
| Vue d'ensemble | 1, 2 |
| File d'appels + dossier + feedback | 3, 4 |
| Validation, 4 sections, caveat in UI | 5, 6 |
| Le jeu de données | 7, 8 |
| Reuse `validation.py` unchanged | 5 (imports only) |
| `web/dist`, `web/node_modules` gitignored | 2 step 3 |
| API binds 127.0.0.1 | 1 step 8, 8 step 4 |
| No endpoint returns a full table | 5 (`.head(500)`), tested in Task 5 |
| Feedback writes the same schema as Streamlit | 3 |
| Tests assert real values, not just 200 | every API test |
| React/Streamlit numbers must match | 6 step 3 |
| Visual verification is manual | Global Constraints; 4 step 5, 6 step 3 |

No gaps.

**Placeholder scan:** Task 2 steps 4, 6, 7 and Tasks 4, 6, 8 specify component contracts, props, and
content rather than full JSX. This is deliberate and bounded: every prop type, endpoint, section, and
required behaviour is named, and the visual layer is where `frontend-design` guidance applies at
implementation time. No step says "TBD", "handle edge cases", or "similar to Task N".

**Type consistency:** `vendor_code`, `client_id`, `article_id`, `priority_score`, `p_model`, `label`,
`ca_sum`, `valeur_attendue` are spelled identically across the Python routes and the TypeScript types.
`Overview`, `Vendor`, `QueueRow`, `Dossier`, `Protocol`, `Aggregate`, `Recommendations`, `PairRow`,
`DatasetProfile` are each defined once in `api.ts` and match their endpoint's response shape.
`getJSON` / `postJSON` / `fmtTND` / `fmtPct` / `fmtInt` are defined in Task 2 and used unchanged
thereafter.

**One correction applied:** Task 7 Step 1's first test name contained a non-ASCII character; the step
now instructs renaming it to `test_coverage_lists_every_observed_month_once`.
