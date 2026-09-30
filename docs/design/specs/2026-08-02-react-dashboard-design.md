# Tableau de bord React — design

**Date:** 2026-08-02
**Status:** Approved, pending implementation plan

## Problem

The project's three deliverables live in three places: Excel call sheets, a Streamlit vendor app, and
a Streamlit validation page. Nothing shows the whole system at once, and nothing presents the dataset's
own characteristics — which is where most of the project's analytical work actually went.

## Goal

A single React dashboard covering four views: an executive overview, the vendor call queue, the model
validation evidence, and a profile of the dataset itself.

**Audience:** dual. The overview and call queue are for the company; the validation and dataset views
are for the encadrant/jury. Interface language is French throughout, matching the existing tooling.

## Decisions settled during brainstorming

1. **Data reaches the browser through a FastAPI backend**, never through the bundle. The extract
   contains named individuals and their revenue; `data/` is gitignored precisely for that reason.
   A static export would place personal data inside the frontend folder and, after `vite build`,
   inside a `dist/` directory that is easy to copy or commit by accident. The API keeps the data
   where `.gitignore` already protects it.
2. **The Streamlit app stays.** It is working, tested software with 24 passing tests. React is
   additive. Retiring Streamlit is a later decision, made once React has proven itself — not a
   precondition of this work.
3. **All four views are in scope.**
4. **The existing visual language carries over** rather than starting fresh, so the Excel fiches,
   the Streamlit app, and this dashboard read as one system.

## Architecture

```text
api/
  main.py               FastAPI app; mounts routers
  deps.py               cached data loaders (read parquet/CSV once per process)
  routes/
    overview.py         KPIs, monthly trend, cross-vendor top clients
    queue.py            vendors, per-vendor queue, client dossier, feedback POST
    validation.py       protocol, aggregate metrics, recommendations, pair drill-down
    dataset.py          dataset profile
src/
  validation.py         REUSED unchanged by the API
  dataset_profile.py    NEW — profiling computations, pure pandas, testable
web/
  package.json, vite.config.ts, tsconfig.json, index.html
  src/
    main.tsx, App.tsx, api.ts, theme.css
    components/  Layout, StatTile, DataTable, Caveat, SectionLabel, charts
    pages/       Overview, Queue, Validation, Dataset
tests/
  test_api.py           FastAPI TestClient coverage per endpoint
  test_dataset_profile.py
```

**Reuse is the core of this design.** `src/validation.py` already computes calibration, decile lift,
precision@N and the queue replay, all under test. `contact_queue.py` already produced
`clients_to_contact.csv`. The API imports and serves these; it does not reimplement them. The only
new Python is `dataset_profile.py`.

**No CORS handling.** Vite proxies `/api/*` to `http://localhost:8000`, so the frontend uses relative
paths and there is no host configuration to get wrong or to leak into a build.

### Data flow

`data/*.parquet|csv` → FastAPI (aggregate + cache) → JSON over `/api` → React state → render.

Payload discipline matters: `pair_scores_2026.csv` is 15 MB and must never cross the wire whole. The
drill-down endpoint filters by vendor and client server-side and returns tens of rows. Every endpoint
returns aggregates or a bounded slice, never a raw table.

### API surface

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/overview` | KPI tiles, monthly CA series, top-priority clients across vendors |
| GET | `/api/vendors` | vendor codes, names, client counts, TND at stake |
| GET | `/api/vendors/{code}/queue` | ranked clients for one vendor |
| GET | `/api/clients/{id}` | dossier: status, articles + reasons + confidence, monthly CA series |
| POST | `/api/feedback` | appends to `data/feedback.csv`, same schema as the Streamlit app |
| GET | `/api/validation/protocol` | train/validation/test windows, pair counts, AUC |
| GET | `/api/validation/aggregate?threshold=` | benchmarks, decile table, calibration table, confusion |
| GET | `/api/validation/recommendations` | precision@N vs manual score, replay stats, money view |
| GET | `/api/validation/pairs?vendor=&client=&view=` | drill-down rows with the reality flag |
| GET | `/api/dataset/profile` | coverage, entities, naming gaps, sparsity, concentration |

## Views

**Vue d'ensemble.** KPI tiles (clients to contact, TND at stake, vendors, test AUC), monthly revenue
trend with the June–December gap visibly absent rather than silently interpolated, and the highest-
priority clients across all vendors.

**File d'appels.** Vendor selector → ranked client queue → dossier with recommended articles, their
reasons, confidence labels, a monthly revenue chart, and the feedback form. Feedback POSTs to the API
and appends to the same `data/feedback.csv` the Streamlit app writes, so the two never diverge.

**Validation du modèle.** The four sections already built and validated in Streamlit: protocol
(train vs test), aggregate metrics including the calibration curve, precision@N against the manual
rule, and the per-prediction drill-down with a reality column. The §3 replication caveat must appear
in this UI too — it is a property of the measurement, not of the Streamlit page.

**Le jeu de données.** The profile rendered visually: 896,312 rows, 15 months of coverage with the
7-month annual hole drawn explicitly, the naming gaps (38.8% of articles unnamed), sparsity (0.75%
fill, 74.3% of pairs seen in a single month), concentration (top 10% of articles = 73.2% of CA), and
the May-2026 completeness anomaly flagged as an open question rather than presented as fact.

## Visual language

Carried over from `.streamlit/config.toml` and `src/ui_theme.py`: warm paper background, dark graphite
chrome, Archivo Expanded for headings, JetBrains Mono for figures and codes, safety-orange accent.
Charts via Recharts. The palette lives in `web/src/theme.css` as CSS custom properties so it is
declared once.

## Confidentiality requirements

- `data/` stays gitignored. The API reads from it at runtime; nothing is copied into `web/`.
- `web/dist/` and `web/node_modules/` are added to `.gitignore`.
- The API binds to `127.0.0.1` only. It is a local tool, not a service.
- No endpoint returns a full table. Aggregates or bounded slices only.

## Testing

- `tests/test_api.py` — FastAPI `TestClient` per endpoint: status, response shape, and one asserted
  value per endpoint that must match what `validation.py` computes directly. Fast, no browser.
- `tests/test_dataset_profile.py` — synthetic frames, as with `test_validation.py`.
- `web` — Vitest over the data-shaping helpers in `api.ts` and any chart-formatting utilities.
- Feedback POST is tested against a temp file, never the real `data/feedback.csv`.

**Explicitly not covered by tests: whether the pages look right.** Two prior UI bugs in this project
(invisible text, contrast regression) passed every test. A hand-written CSS layer has more surface for
this than Streamlit's themed components did. Visual verification is a manual step, not a test claim.

## Build order

One system, built in vertical slices so that each slice is runnable and demoable:

1. API skeleton + app shell + **Vue d'ensemble** — proves the full stack end to end
2. **File d'appels** + dossier + feedback
3. **Validation du modèle**
4. **Le jeu de données**

If time runs out, every completed slice is still working software.

## Out of scope

- Retiring or modifying the Streamlit app.
- Authentication, deployment, hosting. This is a local tool.
- Any change to model logic, `contact_queue.py`, or the recommendation rules.
- Server-side rendering, PWA, offline mode.

## Success criteria

- `npm run dev` starts both processes; all four views render with real data.
- No client name or revenue figure exists anywhere under `web/`.
- Every endpoint is covered by a test asserting a real value, not just a 200.
- The validation view's numbers match the Streamlit page's numbers exactly — same module, same data.
