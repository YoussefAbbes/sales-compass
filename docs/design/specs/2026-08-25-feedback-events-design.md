# Feedback events — design

**Status:** approved for planning, 2026-08-25.
**Scope:** the React dashboard and the model-results API only.

## Why

Every input the system has today is **passive**: the NAV ledger records transactions
that would have happened anyway. Nothing can tell the system it was wrong.

Two consequences, and the first matters more than the second:

1. **The ledger cannot record why a client stopped.** A garage that closed down and a
   garage that is merely dormant are identical in NAV data — both simply stop appearing.
   No volume of transaction data will ever separate them. A vendor knows within one
   phone call, and that knowledge currently dies in their head while the model keeps
   ranking the closed garage every quarter.

2. **There is no business number.** The model reports `AUC 0.774` and
   `precision@30 61.9%`, both from a backtest. Nobody can say how many orders the
   recommended calls produced, and no mechanism exists that could ever answer it.

The React dashboard already collects exactly this. `SellerExperiencePage.tsx` has the
action buttons, the note field and a progress counter — all held in `useState`, so a
refresh discards them. A vendor cannot work a 200-client queue across two sessions.

## Decisions taken

| Question | Decision |
| --- | --- |
| What is it for? | Both: operational now, measurement later |
| Where does it run? | Local now, behind a seam that swaps later |
| Does it change the queue? | No. Record faithfully; act on it in a later iteration |
| Cross-vendor visibility? | **None.** Strictly per-vendor, even for shared clients |
| Streamlit | **Out of scope.** Not read, not written, not modified |

### Strictly per-vendor

90.9% of clients appear under more than one vendor code (up to 37). Feedback is
nonetheless scoped to the vendor who recorded it: vendor A's events are invisible to
vendor B, including for a client they share. This is a deliberate product decision, not
an oversight. It keeps every read path trivially scoped and defers the question of who
owns a shared client, which the data does not answer either.

## Storage

`data/feedback_events.jsonl` — one JSON object per line, append-only, never mutated.

Chosen over SQLite and over extending a CSV:

- **An event log is both answers at once.** Operational state and the measurement
  dataset are two views derived from the same immutable events, rather than two stores
  that can disagree.
- **Schema evolution is free.** A field added next month leaves last month's lines
  parseable. A CSV with a changing header does not.
- **Concurrent appends are safe** for whole lines opened `O_APPEND`; a CSV carrying
  free-text notes is exactly where quoting corrupts.
- **One seam to swap.** A `FeedbackStore` protocol (`append`, `read_for_vendor`) with a
  JSONL implementation today and a database implementation later. Nothing above the
  protocol changes.

A malformed line must never poison a read: parse failures are skipped, not raised. A log
that cannot be read because of one bad line is worse than a log missing one event.

## The event

```json
{"event_id": "8f2c...", "recorded_at": "2026-08-25T21:53:11",
 "vendor_code": "007", "client_id": "0508",
 "action": "ORDER", "note": "2 palettes, livraison jeudi",
 "score_snapshot": "2026-07-01"}
```

| Field | Why it is there |
| --- | --- |
| `event_id` | Idempotency. **Minted by the client**, so a retry after a dropped response is the same event rather than a second one |
| `recorded_at` | ISO-8601 local, matching the rest of the project |
| `vendor_code` | The scope key. Claimed, not verified — see Identity |
| `client_id` | Joins to the name at read time |
| `action` | One of the seven below |
| `note` | Free text, may be empty |
| `score_snapshot` | Which forecast window was on screen |

**`score_snapshot` is the field most easily forgotten and the one the measurement half
depends on.** Without it an outcome cannot be joined back to the recommendation that
produced it, and the log degrades into a call journal.

**`client_name` is deliberately absent.** Storing it would copy confidential names into
a second file for no benefit. `client_id` joins to the name whenever it is displayed.

## Actions

Seven. The existing four are kept unchanged; three are added because they carry
information no other source can produce.

| value | label | detail | why it exists |
| --- | --- | --- | --- |
| `CONTACTED` | Client contacté | L'échange a eu lieu | existing |
| `ORDER` | Commande passée | Vente conclue | **the outcome measurement needs** |
| `QUOTE` | Préparer un devis | Opportunité confirmée | existing |
| `SNOOZE` | Reporter | À rappeler plus tard | existing |
| `UNREACHABLE` | Injoignable | Aucun contact établi | **not derivable from the ledger** |
| `CLOSED` | Client fermé | L'activité a cessé | **not derivable from the ledger** |
| `DISMISS` | Non pertinent | Recommandation à revoir | existing — judges the *model*, not the client |

`DISMISS` and `CLOSED` are distinct and both matter: the first says the recommendation
was wrong, the second says the client is gone. Collapsing them would lose the only
signal that criticises the model directly.

## API

The first write path into an API that is otherwise entirely `GET`.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/feedback` | Record one event; returns the stored record |
| `GET` | `/api/v1/vendors/{vendor_code}/feedback` | Replay that vendor's events |

- CORS `allow_methods` becomes `["GET", "POST"]`.
- Both routes sit behind the existing API-key dependency.
- Unknown `vendor_code` or `client_id` returns 404, consistent with existing routes.
- Re-posting a known `event_id` returns the existing record rather than appending.

## Frontend

`SellerExperiencePage.tsx` already renders the interaction; the work is mostly removing
local state.

- `postJSON` beside `getJSON` in `api.ts`, carrying `X-API-Key` when configured.
- `setAction` / `setNote` POST, then reflect what the server stored.
- Replay `GET .../feedback` on vendor load, so progress survives a refresh.
- Three new action buttons, markers `01`–`07`.
- A failed POST must surface. Silent loss of a recorded call is worse than an error.

## Identity — recorded as claimed, not verified

With a single shared API key the server cannot verify which vendor is submitting; it can
only record what the browser asserts. This spec states that rather than implying an
authentication that does not exist.

`vendor_code` therefore arrives in the request body and is stored as claimed. The check
lives in one function so that when per-vendor keys are decided, verification lands in
exactly one place and every route inherits it.

## Testing

TDD, following the existing suites.

**Store:** appends without mutating prior lines; a malformed line is skipped, not raised;
`read_for_vendor` returns only that vendor's events; a repeated `event_id` appends once.

**API:** POST round-trips; replay is vendor-scoped, including for a client shared with
another vendor; unknown vendor and unknown client 404; `client_name` never appears in a
stored record.

**Frontend:** `tsc --noEmit`.

## Explicitly out of scope

- `contact_queue.py`, `clients_to_contact.csv` and cooldown rules — the next iteration,
  once real events show what durations are actually right.
- Cross-vendor visibility.
- Any Streamlit code.
- Retraining on feedback. Outcomes exist only for clients that were called, and those
  were chosen by the model — training on them naively reinforces its existing
  preferences rather than correcting them. Evaluation first, and carefully.
- Durable cloud storage. The protocol is the seam; the implementation waits for the
  deployment decision.
