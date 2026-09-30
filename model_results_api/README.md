# Model Results REST API

Fresh REST handoff for the two models in the vendor-recommendation project:

1. **Repurchase model** — ranks articles a client bought before and may buy again.
2. **Article-prospect model** — ranks a vendor's clients who never bought a selected article.

The frontend receives bounded JSON responses. It never receives Parquet files, full score
tables, real model artifacts, or unrestricted access to `data/`.

The pure feature/ranking engine required by the prospect endpoint is bundled under `engine/`.
The only external runtime input is a source root containing compatible `data/` and `reports/`
directories.

## Run locally with the real owner-side data

From the repository root:

```powershell
py -m uvicorn model_results_api.app.main:app --host 127.0.0.1 --port 8001
```

Open:

- Swagger UI: <http://127.0.0.1:8001/docs>
- OpenAPI: <http://127.0.0.1:8001/openapi.json>
- Health: <http://127.0.0.1:8001/api/v1/health>

The default source is the parent project, which contains `data/` and `reports/` locally.

## Frontend-developer mode

Use the committed anonymised fixtures:

```powershell
$env:MODEL_RESULTS_SOURCE_ROOT="fixtures"
py -m uvicorn model_results_api.app.main:app --host 127.0.0.1 --port 8001
```

The response contracts are identical. The fixtures include repurchase scores. They do not
include the real prospect-model artifact, so `/prospects` honestly reports
`source: "heuristic"`; the real owner-side backend reports `source: "model"` when
`data/prospect_model.cbm` is present.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Data and artifact readiness, plus the serving window |
| GET | `/api/v1/meta` | Model versions and score semantics |
| GET | `/api/v1/vendors` | Vendor selector |
| GET | `/api/v1/dashboard/overview` | Executive statistics, model summaries, top priorities and top brands |
| GET | `/api/v1/vendors/{vendor}/dashboard` | One vendor's KPIs, brand portfolio, history and ranked client list |
| GET | `/api/v1/vendors/{vendor}/clients/{client}` | Full client dossier, brand affinity, evidence and recommendations |
| GET | `/api/v1/brands?query=bosch` | Searchable brand directory with CA, share, clients, references and evolution |
| GET | `/api/v1/brands/detail?brand=BOSCH` | One brand's history plus top vendors, clients and articles |
| GET | `/api/v1/articles?query=huile` | Article search |
| GET | `/api/v1/vendors/{vendor}/repurchases` | Model 1 results, grouped by client |
| GET | `/api/v1/vendors/{vendor}/articles/{article}/prospects` | Model 2 results |
| GET | `/api/v1/vendors/{vendor}/articles/{article}/recommendations` | Both article blocks in one response |
| POST | `/api/v1/feedback` | Record one call outcome |
| GET | `/api/v1/vendors/{vendor}/feedback` | Replay that vendor's recorded outcomes |

See [`examples/requests.http`](examples/requests.http) for ready-to-run requests.

`/health` and `/meta` both carry `repurchase_window` (`snapshot_date`, `window_end`,
`expired`, `days_remaining`). An expired window flips `status` to `degraded` while every
route keeps answering — the numbers stay well-formed and stop being forecasts. A missing
record reads as unknown, not expired. Operational detail is in
[`docs/model-lifecycle.md`](../docs/model-lifecycle.md).


The repository's `web/` application consumes these contracts. From the repository root,
`scripts/run-dev.ps1` (Windows) or `scripts/run-dev.sh` (macOS/Linux) starts both the API and the React dashboard at <http://localhost:5173>.

## Semantics the frontend must preserve

For the article page, prefer the combined `/recommendations` endpoint. Its `repurchases`
object feeds **Ils l'achetaient**, and its `prospects` object feeds **À prospecter**. The
same `limit` applies independently to both lists.

- **Every ranked list is fully reachable.** `limit` on the article blocks accepts up
  to 200 and `per_client` on `/repurchases` and the client dossier up to 50 — enough
  that no real vendor hits the ceiling (the deepest article in the live data has 99
  buyers), while keeping responses bounded. Defaults are unchanged (5, and 3
  articles per client), so a caller only gets more by asking.
- **Articles past the third come from a different source, by necessity.**
  `clients_to_contact.csv` pivots articles into `article_txt_1..3`; that shape was
  the old `le=3` ceiling. Ranks 1–3 still come from it — `contact_queue.py` fills
  model coverage gaps the pair table has no row for, so those three stay
  authoritative and their `reason` wording is unchanged. Ranks 4+ come from the
  pair-level score table, ranked by damped expected value, and their `reason` is
  generated live, so the two styles coexist in one list by design.
- `rank` is contiguous and absolute across the whole list. When revealing a list
  progressively, keep showing the server's `rank` — do not renumber per page.
- **Lists are ordered by a damped figure, not by the money column beside it.** A
  client is demoted while they sit early in their own purchase cycle:
  `r = days_since_last / median_interval`, and `damping = 1.0` when `r >= 1.0`,
  otherwise `max(0.05, r**2)`. Squared so that a client one day early is barely
  moved while one served yesterday drops out of the visible top. Applied to
  `priority_score` and to block A's ranking. `expected_value_tnd` and
  `previous_ca_tnd` stay undamped, because they are amounts a vendor quotes out
  loud. So **do not re-sort a list client-side by `expected_value_tnd`** — that
  restores the exact defect: the repurchase label spans three months, so a client
  on a 12-day cadence scores ~91% the day after they order, and an undamped sort
  puts them at the top of a call list the morning after their delivery.
- `signals.overdue_ratio` is the field that explains a rank the money column does
  not. Below `1.0` the client is inside their cycle and was demoted for it; the
  `reason` string says so. Display it — it is the only visible account of why a
  large client can sit low.
- `repurchase_score_raw` is an uncalibrated model score. If displayed, label it as an
  optimistic estimate. Do not silently call it a guaranteed probability.
- The repurchase endpoint serves a **forecast**, not a backtest. `data_window` names the
  period the scores answer for and `evaluation_mode` classifies them: `live_forecast`
  while that period is open, `expired_forecast` once it has closed, `unknown` when no
  serving record is present. Both are derived from the record written beside the scores,
  so neither can drift from the file being served. Do not cache `evaluation_mode`
  alongside a score and assume it stays true.
- A forecast has no observed outcome yet, so no accuracy figure accompanies it. If you
  need one, read the model card, not the endpoint.
- The prospect endpoint exposes a validated ordering and factual signals. It deliberately
  exposes no percentage.
- Always display the `source` field for prospects during development: `model` or `heuristic`.
- When `items` is empty, use `empty_reason`, `current_window_buyers`, and
  `eligible_candidates` to explain why. Do not infer that an empty list means the model is
  missing.
- The NAV ledger is continuous and no month is truncated; `/dashboard/overview` reports its
  real span rather than a hardcoded one. Read the coverage from the response.
- Brand rankings use positive purchases only. Unknown brands stay in the coverage denominator
  but do not appear as a named brand.
- Brand `evolution_pct` compares equal spans: both years are restricted to the months the
  most recent year actually covers. A partial latest year would otherwise be weighed against
  a full one and report a collapse for every brand.

## Access control

**Serving `data/` without authentication is refused at startup.** The process stops
before it accepts a connection, and prints the three ways forward. Fixtures are exempt:
they are anonymised by construction, so nothing there needs a key.

| Source root | `MODEL_RESULTS_API_KEY` | Result |
| --- | --- | --- |
| `fixtures` | unset | starts, no key required |
| real `data/` | set | starts, key enforced on all `/api/v1` routes |
| real `data/` | unset | **refuses to start** |
| real `data/` | unset, `MODEL_RESULTS_ALLOW_ANONYMOUS=1` | starts, unauthenticated |

For a controlled shared deployment:

```powershell
$env:MODEL_RESULTS_API_KEY="a-long-random-secret"
```

The frontend must then send:

```http
X-API-Key: a-long-random-secret
```

For local work against real data with no key, opt in explicitly — and keep the bind on
loopback:

```powershell
$env:MODEL_RESULTS_ALLOW_ANONYMOUS="1"
py -m uvicorn model_results_api.app.main:app --host 127.0.0.1 --port 8001
```

The variable exists so that anonymous access to real data is something a person typed,
never something a deploy reached by leaving a variable unset.

### Schema endpoints

`/docs`, `/redoc` and `/openapi.json` are mounted on the app rather than behind the API-key
dependency, so they answer without a key. They expose no client data but do publish every
route, field and parameter — so **they are disabled whenever an API key is configured**.
With no key set (local development) they stay available. `/api/v1/health` is always
reachable, so a platform probe still works.

### Still missing

This shared key is not a replacement for production authentication and **vendor-level
authorization**: any key holder reads all 54 vendors. Until that exists, bind the API to
`127.0.0.1` only.

- The feedback log lives on the container filesystem and does not survive a redeploy;
  durable storage is pending the deployment decision.
- Feedback idempotency is check-then-act with no lock, so two concurrent posts of the
  same `event_id` could both write under multiple workers.

## Tests

Tests always use anonymised fixtures:

```powershell
py -m pytest model_results_api/tests -q
```

They verify the response contracts, useful OpenAPI schemas, bounded result sizes, 404s,
and the rule that the prospect response never contains a probability.

The owner can additionally verify both real local artifacts without printing client data:

```powershell
py model_results_api/tools/smoke_real.py
```
