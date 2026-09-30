# Feedback Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist vendor feedback from the React dashboard so it survives a refresh, and accumulate it as an immutable event log that can later answer whether recommended calls converted.

**Architecture:** An append-only JSONL log behind a `FeedbackStore` protocol, written through the first `POST` route in an otherwise read-only API. The React page already renders the interaction; its `useState` is replaced by a POST plus a replay fetch. Events are stored, never mutated — "current status" is a view derived by taking the latest event per client.

**Tech Stack:** FastAPI, Pydantic v2, pytest, React 18 + TypeScript, Vite.

**Spec:** `docs/superpowers/specs/2026-08-25-feedback-events-design.md`

## Global Constraints

- Python: this repo runs `py` (Windows launcher), not `python`.
- All API tests run against fixtures: `model_results_api/tests/conftest.py` sets `MODEL_RESULTS_SOURCE_ROOT=fixtures` at import.
- Tests that write must use the `scratch_dir` fixture from that conftest. `tmp_path` raises `PermissionError` on this machine.
- Never store `client_name` in the event log. `client_id` only.
- Feedback is strictly per-vendor. A read for vendor A must never return vendor B's events, including for a shared client.
- No Streamlit code is read, written, or modified.
- `contact_queue.py`, `clients_to_contact.csv` and cooldown rules are out of scope.
- French user-facing strings, matching the existing API (`vendeur {code} inconnu`).
- Engine modules are generated: never hand-edit `model_results_api/engine/`. This plan touches none.

### Deviation from the spec, deliberate

The spec says `postJSON` should carry `X-API-Key` "when configured". There is no
frontend mechanism for configuring a key, and `getJSON` does not send one either.
`postJSON` therefore mirrors `getJSON` exactly. Adding key support to only the write
helper would be inconsistent and dead code. Revisit when the per-vendor auth decision
lands.

### Refinement of the spec

`score_snapshot` is filled **by the server** from `repo.live_window()["snapshot_date"]`,
not supplied by the client. The server knows the served window authoritatively; asking
the browser to report it invites a stale or forged value on the one field the
measurement half depends on.

## File Structure

| File | Responsibility |
| --- | --- |
| `model_results_api/app/feedback_store.py` | **Create.** `FeedbackStore` protocol + `JsonlFeedbackStore`. Pure I/O, no HTTP, no Pydantic. |
| `model_results_api/app/schemas.py` | **Modify.** `FeedbackAction`, `FeedbackEventIn`, `FeedbackEvent`, `FeedbackEventList`. |
| `model_results_api/app/services.py` | **Modify.** `record_feedback`, `vendor_feedback` — validation and the store instance. |
| `model_results_api/app/main.py` | **Modify.** `POST /feedback`, `GET /vendors/{v}/feedback`, CORS methods. |
| `model_results_api/tests/test_feedback_store.py` | **Create.** Store behaviour against `scratch_dir`. |
| `model_results_api/tests/test_feedback_api.py` | **Create.** Contract + vendor scoping. |
| `web/src/api.ts` | **Modify.** `postJSON`, endpoints, types. |
| `web/src/pages/SellerExperiencePage.tsx` | **Modify.** 7 actions, POST, replay. |
| `.gitignore` | **Modify.** Ignore the fixture-mode log. |

---

### Task 1: The feedback store

The store knows nothing about HTTP or Pydantic — it takes and returns plain dicts. That
keeps it testable without a client and swappable for a database later without touching
anything above it.

**Files:**
- Create: `model_results_api/app/feedback_store.py`
- Test: `model_results_api/tests/test_feedback_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class FeedbackStore(Protocol)` with `append(event: dict) -> dict` and `read_for_vendor(vendor_code: str) -> list[dict]`
  - `class JsonlFeedbackStore(path: Path)` implementing it

- [ ] **Step 1: Write the failing tests**

Create `model_results_api/tests/test_feedback_store.py`:

```python
# -*- coding: utf-8 -*-
"""The log is append-only, per-vendor, and survives a corrupt line.

Events are stored rather than state. Overwriting a per-client status would collapse
SNOOZE -> CONTACTED -> ORDER into ORDER and destroy the sequence that says how many
touches a sale took. State is derived from events; events cannot be derived from state.
"""
from model_results_api.app.feedback_store import JsonlFeedbackStore


def _event(event_id="e1", vendor_code="007", client_id="0508", action="CONTACTED"):
    return {"event_id": event_id, "recorded_at": "2026-08-25T21:53:11",
            "vendor_code": vendor_code, "client_id": client_id,
            "action": action, "note": "", "score_snapshot": "2026-07-01"}


def test_append_then_read_returns_the_event(scratch_dir):
    store = JsonlFeedbackStore(scratch_dir / "events.jsonl")

    store.append(_event())

    assert store.read_for_vendor("007")[0]["client_id"] == "0508"


def test_a_second_event_does_not_replace_the_first(scratch_dir):
    """The whole point of an event log: history is kept, not overwritten."""
    store = JsonlFeedbackStore(scratch_dir / "events.jsonl")
    store.append(_event(event_id="e1", action="SNOOZE"))
    store.append(_event(event_id="e2", action="ORDER"))

    actions = [e["action"] for e in store.read_for_vendor("007")]

    assert actions == ["SNOOZE", "ORDER"]


def test_reads_are_scoped_to_one_vendor_even_for_a_shared_client(scratch_dir):
    """90.9% of clients sit under more than one vendor code. Scoping is the
    product decision, so it is pinned here rather than left to a caller."""
    store = JsonlFeedbackStore(scratch_dir / "events.jsonl")
    store.append(_event(event_id="e1", vendor_code="007", client_id="0508"))
    store.append(_event(event_id="e2", vendor_code="011", client_id="0508"))

    assert len(store.read_for_vendor("007")) == 1
    assert store.read_for_vendor("007")[0]["event_id"] == "e1"


def test_a_corrupt_line_is_skipped_rather_than_raised(scratch_dir):
    """A log unreadable because of one bad line is worse than a log missing
    one event."""
    path = scratch_dir / "events.jsonl"
    store = JsonlFeedbackStore(path)
    store.append(_event(event_id="e1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    store.append(_event(event_id="e2"))

    assert [e["event_id"] for e in store.read_for_vendor("007")] == ["e1", "e2"]


def test_reposting_the_same_event_id_does_not_append_twice(scratch_dir):
    """A dropped response makes the browser retry. The retry is the same event."""
    store = JsonlFeedbackStore(scratch_dir / "events.jsonl")
    store.append(_event(event_id="e1", action="CONTACTED"))

    store.append(_event(event_id="e1", action="ORDER"))

    events = store.read_for_vendor("007")
    assert len(events) == 1
    assert events[0]["action"] == "CONTACTED"


def test_reading_a_log_that_does_not_exist_yet_is_empty_not_an_error(scratch_dir):
    store = JsonlFeedbackStore(scratch_dir / "never-written.jsonl")

    assert store.read_for_vendor("007") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest model_results_api/tests/test_feedback_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_results_api.app.feedback_store'`

- [ ] **Step 3: Write the store**

Create `model_results_api/app/feedback_store.py`:

```python
# -*- coding: utf-8 -*-
"""Append-only storage for vendor feedback events.

Plain dicts in, plain dicts out: no HTTP, no Pydantic, no pandas. The protocol is the
seam that lets a database replace the file later without anything above it changing.

Events are never mutated. "Current status per client" is a view derived by taking the
latest event, because state can always be computed from events and events can never be
recovered from state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class FeedbackStore(Protocol):
    def append(self, event: dict) -> dict: ...
    def read_for_vendor(self, vendor_code: str) -> list[dict]: ...


class JsonlFeedbackStore:
    """One JSON object per line, opened O_APPEND so whole-line writes do not interleave."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # One corrupt line must not cost the whole log. Skipping keeps every
                # readable event available; raising would hide all of them.
                continue
        return events

    def append(self, event: dict) -> dict:
        existing = {e.get("event_id") for e in self._all()}
        if event.get("event_id") in existing:
            # The browser retried after a dropped response. Same event, not a new one.
            return next(e for e in self._all() if e.get("event_id") == event["event_id"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def read_for_vendor(self, vendor_code: str) -> list[dict]:
        return [e for e in self._all() if str(e.get("vendor_code")) == str(vendor_code)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest model_results_api/tests/test_feedback_store.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add model_results_api/app/feedback_store.py model_results_api/tests/test_feedback_store.py
git commit -m "feat(api): append-only feedback event store"
```

---

### Task 2: Schemas and the service layer

**Files:**
- Modify: `model_results_api/app/schemas.py`
- Modify: `model_results_api/app/services.py`
- Test: `model_results_api/tests/test_feedback_api.py` (created here, extended in Task 3)

**Interfaces:**
- Consumes: `JsonlFeedbackStore` from Task 1.
- Produces:
  - `FeedbackAction = Literal["CONTACTED", "ORDER", "QUOTE", "SNOOZE", "UNREACHABLE", "CLOSED", "DISMISS"]`
  - `FeedbackEventIn` — `event_id, vendor_code, client_id, action, note`
  - `FeedbackEvent` — the above plus `recorded_at`, `score_snapshot`
  - `FeedbackEventList` — `vendor_code: str`, `items: list[FeedbackEvent]`
  - `services.record_feedback(payload: FeedbackEventIn) -> FeedbackEvent`
  - `services.vendor_feedback(vendor_code: str) -> FeedbackEventList`

- [ ] **Step 1: Write the failing tests**

Create `model_results_api/tests/test_feedback_api.py`:

```python
# -*- coding: utf-8 -*-
"""Recording a call, and getting it back after a refresh.

The React page held feedback in useState, so closing the tab discarded a vendor's
progress through a 200-client queue. These tests pin the round trip and the two rules
that are easy to break silently: vendor scoping, and keeping client names out of the log.
"""
import pytest

from model_results_api.app import services
from model_results_api.app.feedback_store import JsonlFeedbackStore
from model_results_api.app.schemas import FeedbackEventIn


@pytest.fixture(autouse=True)
def isolated_store(scratch_dir, monkeypatch):
    """Never append to fixtures/data/ during a test run."""
    monkeypatch.setattr(services, "feedback_store",
                        JsonlFeedbackStore(scratch_dir / "events.jsonl"))


def _known_ids():
    """Derived from exactly what the validators check, so a fixture regeneration
    cannot make these tests fail for the wrong reason."""
    from model_results_api.app import repository as repo
    fact = repo.fact()
    vendor = str(fact[fact["ca"] > 0]["vendor_code"].iloc[0])
    client = str(repo.clients()["client_id"].iloc[0])
    return vendor, client


def _payload(event_id="e1", vendor_code=None, client_id=None, action="CONTACTED"):
    vendor, client = _known_ids()
    return FeedbackEventIn(event_id=event_id, vendor_code=vendor_code or vendor,
                           client_id=client_id or client, action=action, note="")


def test_recording_returns_the_stored_event():
    stored = services.record_feedback(_payload())

    assert stored.action == "CONTACTED"
    assert stored.recorded_at


def test_the_server_stamps_the_window_the_vendor_was_looking_at():
    """score_snapshot is server-filled. Without it an outcome cannot be joined back
    to the recommendation that produced it, and the log is only a call journal."""
    from model_results_api.app import repository as repo

    stored = services.record_feedback(_payload())

    assert stored.score_snapshot == repo.live_window()["snapshot_date"]


def test_a_recorded_event_comes_back_on_replay():
    payload = _payload()
    services.record_feedback(payload)

    listed = services.vendor_feedback(payload.vendor_code)

    assert [e.event_id for e in listed.items] == ["e1"]


def test_an_unknown_vendor_is_rejected():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        services.record_feedback(_payload(vendor_code="ZZZ"))

    assert excinfo.value.status_code == 404


def test_the_client_name_is_never_stored():
    """The log holds client_id only. Copying names into a second file duplicates
    confidential data for no benefit."""
    stored = services.record_feedback(_payload())

    assert "client_name" not in stored.model_dump()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest model_results_api/tests/test_feedback_api.py -q`
Expected: FAIL — `ImportError: cannot import name 'FeedbackEventIn'`

- [ ] **Step 3: Add the schemas**

In `model_results_api/app/schemas.py`, append at the end of the file:

```python
FeedbackAction = Literal[
    "CONTACTED", "ORDER", "QUOTE", "SNOOZE", "UNREACHABLE", "CLOSED", "DISMISS",
]


class FeedbackEventIn(BaseModel):
    """What the browser sends.

    `event_id` is minted by the client so a retry after a dropped response is
    recognised as the same event rather than becoming a duplicate.

    `vendor_code` is CLAIMED, not verified: a single shared API key cannot identify the
    caller. Stated rather than implied, so nobody reads this as authenticated.
    """
    event_id: str = Field(min_length=1, max_length=64)
    vendor_code: str = Field(min_length=1, max_length=32)
    client_id: str = Field(min_length=1, max_length=32)
    action: FeedbackAction
    note: str = Field(default="", max_length=2000)


class FeedbackEvent(FeedbackEventIn):
    """What was stored. Server-filled fields are added here, never accepted from
    the browser."""
    recorded_at: str
    score_snapshot: str | None = None


class FeedbackEventList(BaseModel):
    vendor_code: str
    items: list[FeedbackEvent]
```

- [ ] **Step 4: Add the service functions**

In `model_results_api/app/services.py`, add to the `.schemas` import block:
`FeedbackEvent, FeedbackEventIn, FeedbackEventList,`

Add a new import beside the other relative imports at the top of the file:

```python
from .feedback_store import JsonlFeedbackStore
```

Then append at the end of the file:

```python
# Module-level so tests can swap it for a scratch path, and so a database
# implementation later replaces one line rather than every call site.
feedback_store = JsonlFeedbackStore(repo.settings.data_dir / "feedback_events.jsonl")


def _known_client(client_id: str) -> None:
    if client_id not in set(repo.clients()["client_id"].astype(str)):
        raise HTTPException(404, f"client {client_id} inconnu")


def record_feedback(payload: FeedbackEventIn) -> FeedbackEvent:
    """Store one call outcome.

    `score_snapshot` is stamped here rather than accepted from the browser: the server
    knows the served window authoritatively, and it is the single field the measurement
    half depends on.
    """
    _known_vendor(payload.vendor_code)
    _known_client(payload.client_id)
    event = FeedbackEvent(
        **payload.model_dump(),
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        score_snapshot=repo.live_window().get("snapshot_date"),
    )
    stored = feedback_store.append(event.model_dump())
    return FeedbackEvent(**stored)


def vendor_feedback(vendor_code: str) -> FeedbackEventList:
    _known_vendor(vendor_code)
    return FeedbackEventList(
        vendor_code=vendor_code,
        items=[FeedbackEvent(**e) for e in feedback_store.read_for_vendor(vendor_code)],
    )
```

If `datetime` is not already imported in `services.py`, add `from datetime import date, datetime` (the module already imports `date`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -m pytest model_results_api/tests/test_feedback_api.py -q`
Expected: PASS, 5 tests

- [ ] **Step 6: Run the whole API suite for regressions**

Run: `py -m pytest model_results_api/tests -q`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add model_results_api/app/schemas.py model_results_api/app/services.py model_results_api/tests/test_feedback_api.py
git commit -m "feat(api): record and replay vendor feedback events"
```

---

### Task 3: The API routes

**Files:**
- Modify: `model_results_api/app/main.py`
- Modify: `model_results_api/tests/test_feedback_api.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `services.record_feedback`, `services.vendor_feedback` from Task 2.
- Produces: `POST /api/v1/feedback`, `GET /api/v1/vendors/{vendor_code}/feedback`.

- [ ] **Step 1: Write the failing tests**

Append to `model_results_api/tests/test_feedback_api.py`:

```python
from fastapi.testclient import TestClient

from model_results_api.app.main import app

client = TestClient(app)


def _body(event_id="h1", action="CONTACTED"):
    vendor, client = _known_ids()
    return {"event_id": event_id, "vendor_code": vendor, "client_id": client,
            "action": action, "note": "rappeler lundi"}


def test_post_feedback_round_trips_through_the_endpoint():
    response = client.post("/api/v1/feedback", json=_body())

    assert response.status_code == 200
    assert response.json()["action"] == "CONTACTED"


def test_the_endpoint_replays_what_was_posted():
    body = _body(event_id="h2")
    client.post("/api/v1/feedback", json=body)

    listed = client.get(f"/api/v1/vendors/{body['vendor_code']}/feedback").json()

    assert "h2" in [e["event_id"] for e in listed["items"]]


def test_an_action_outside_the_vocabulary_is_rejected():
    """A free-text action would make the log unqueryable a month later."""
    response = client.post("/api/v1/feedback", json=_body() | {"action": "PEUT_ETRE"})

    assert response.status_code == 422


def test_replay_for_an_unknown_vendor_is_404():
    assert client.get("/api/v1/vendors/ZZZ/feedback").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest model_results_api/tests/test_feedback_api.py -q`
Expected: FAIL — 405 Method Not Allowed or 404, because the routes do not exist

- [ ] **Step 3: Add the routes**

In `model_results_api/app/main.py`, add to the `.schemas` import block:
`FeedbackEvent, FeedbackEventIn, FeedbackEventList,`

Add to the `.services` import block:
`record_feedback, vendor_feedback,`

Change the CORS middleware line:

```python
    allow_methods=["GET", "POST"],
```

Append after the last route, before `app.include_router(router)`:

```python
@router.post("/feedback", response_model=FeedbackEvent, tags=["feedback"])
def post_feedback(payload: FeedbackEventIn) -> FeedbackEvent:
    """Record one call outcome.

    The first write path in this API. `vendor_code` is stored as claimed: a shared API
    key cannot identify the caller, and pretending otherwise would be worse than saying
    so. Verification lands here when per-vendor keys are decided.
    """
    return record_feedback(payload)


@router.get(
    "/vendors/{vendor_code}/feedback",
    response_model=FeedbackEventList,
    tags=["feedback"],
)
def get_vendor_feedback(vendor_code: str) -> FeedbackEventList:
    """This vendor's events only -- never another vendor's, even on a shared client."""
    return vendor_feedback(vendor_code)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest model_results_api/tests/test_feedback_api.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Keep the fixture-mode log out of git**

The API appends while it runs against fixtures. That is per-developer noise, exactly
like the existing `fixtures/data/feedback.csv` rule. Add below that line in `.gitignore`:

```
fixtures/data/feedback_events.jsonl
```

- [ ] **Step 6: Run both suites**

Run: `py -m pytest model_results_api/tests -q && py -m pytest tests -q`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add model_results_api/app/main.py model_results_api/tests/test_feedback_api.py .gitignore
git commit -m "feat(api): POST /feedback and per-vendor replay"
```

---

### Task 4: Wire the dashboard

**Files:**
- Modify: `web/src/api.ts`
- Modify: `web/src/pages/SellerExperiencePage.tsx`

**Interfaces:**
- Consumes: both routes from Task 3.
- Produces: no downstream consumers.

- [ ] **Step 1: Add the write helper and types**

In `web/src/api.ts`, after `getJSON`:

```ts
// Mirrors getJSON deliberately, including sending no X-API-Key: the dashboard has no
// mechanism for configuring one, and a write helper that authenticates while the read
// helper does not would be inconsistent and untested.
export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const parsed = (await response.json()) as { detail?: string }
      if (parsed.detail) detail = parsed.detail
    } catch {
      // Keep the status when the backend did not return JSON.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export type FeedbackAction =
  | 'CONTACTED' | 'ORDER' | 'QUOTE' | 'SNOOZE' | 'UNREACHABLE' | 'CLOSED' | 'DISMISS'

export type FeedbackEvent = {
  event_id: string
  vendor_code: string
  client_id: string
  action: FeedbackAction
  note: string
  recorded_at: string
  score_snapshot: string | null
}
```

Add to the `endpoints` object:

```ts
  feedback: `${API}/feedback`,
  vendorFeedback: (vendor: string) => `${API}/vendors/${encodeURIComponent(vendor)}/feedback`,
```

- [ ] **Step 2: Extend the action vocabulary**

In `web/src/pages/SellerExperiencePage.tsx`, delete the local
`type FeedbackAction = ...` line and import the type from `../api` instead.

Replace the `actions` array with all seven:

```tsx
const actions: Array<{
  value: FeedbackAction
  label: string
  detail: string
  marker: string
}> = [
  { value: 'CONTACTED', label: 'Client contacté', detail: 'L’échange a eu lieu', marker: '01' },
  { value: 'ORDER', label: 'Commande passée', detail: 'Vente conclue', marker: '02' },
  { value: 'QUOTE', label: 'Préparer un devis', detail: 'Opportunité confirmée', marker: '03' },
  { value: 'SNOOZE', label: 'Reporter', detail: 'À rappeler plus tard', marker: '04' },
  { value: 'UNREACHABLE', label: 'Injoignable', detail: 'Aucun contact établi', marker: '05' },
  { value: 'CLOSED', label: 'Client fermé', detail: 'L’activité a cessé', marker: '06' },
  { value: 'DISMISS', label: 'Non pertinent', detail: 'Recommandation à revoir', marker: '07' },
]

const actionLabels: Record<FeedbackAction, string> = {
  CONTACTED: 'Contacté',
  ORDER: 'Commande',
  QUOTE: 'Devis',
  SNOOZE: 'Reporté',
  UNREACHABLE: 'Injoignable',
  CLOSED: 'Fermé',
  DISMISS: 'Écarté',
}
```

- [ ] **Step 3: Replay on load and POST on action**

First fix the imports at the top of the file. They are currently
`import { useMemo, useState } from 'react'` and an `../api` block without `getJSON`:

```tsx
import { useEffect, useMemo, useState } from 'react'
```

and add `getJSON`, `postJSON`, `type FeedbackAction` and `type FeedbackEvent` to the
existing `../api` import block (`endpoints` is already there).

Then, in the component, replace the `feedback` state block with a replay effect plus a
POST.

```tsx
  const [feedback, setFeedback] = useState<Record<string, SessionFeedback>>({})
  const [feedbackError, setFeedbackError] = useState<string | null>(null)

  // Replay: a vendor's progress through a 200-client queue must survive a refresh.
  // Latest event per client wins -- the log keeps every event, the UI shows the last.
  useEffect(() => {
    if (!activeVendor) return
    let cancelled = false
    getJSON<{ items: FeedbackEvent[] }>(endpoints.vendorFeedback(activeVendor))
      .then(data => {
        if (cancelled) return
        const latest: Record<string, SessionFeedback> = {}
        for (const event of data.items) {
          latest[`${event.vendor_code}:${event.client_id}`] =
            { action: event.action, note: event.note }
        }
        setFeedback(latest)
      })
      .catch(() => { /* an empty queue state is better than blocking the page */ })
    return () => { cancelled = true }
  }, [activeVendor])

  const record = (action: FeedbackAction, note: string) => {
    if (!activeVendor || !activeClient) return
    setFeedbackError(null)
    postJSON<FeedbackEvent>(endpoints.feedback, {
      event_id: crypto.randomUUID(),
      vendor_code: activeVendor,
      client_id: activeClient,
      action,
      note,
    }).catch((error: Error) => setFeedbackError(error.message))
  }

  const setAction = (action: FeedbackAction) => {
    if (!feedbackKey) return
    const note = feedback[feedbackKey]?.note ?? ''
    setFeedback(previous => ({ ...previous, [feedbackKey]: { action, note } }))
    record(action, note)
  }

  const setNote = (note: string) => {
    if (!feedbackKey) return
    setFeedback(previous => ({
      ...previous,
      [feedbackKey]: { action: previous[feedbackKey]?.action, note },
    }))
  }
```

`setNote` deliberately does not POST on every keystroke. The note travels with the next
action, which is when the vendor has finished typing.

- [ ] **Step 4: Surface a failed write**

Silently losing a recorded call is worse than an error. Immediately after the opening
`<>` of the returned JSX, add:

```tsx
      {feedbackError ? (
        <aside className="notice-panel" role="alert">
          <div className="notice-icon">!</div>
          <div><strong>Enregistrement impossible</strong>
            <p>{feedbackError}</p></div>
        </aside>
      ) : null}
```

- [ ] **Step 5: Typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: exit 0, no output

- [ ] **Step 6: Verify by hand**

Run: `.\run-dev.ps1`, open <http://localhost:5173>, go to *Expérience vendeur*, mark a
client **Commande passée**, then **refresh the page**. The action must still be shown.

Confirm the log: `py -c "print(open('data/feedback_events.jsonl',encoding='utf-8').read())"`

- [ ] **Step 7: Commit**

```bash
git add web/src/api.ts web/src/pages/SellerExperiencePage.tsx
git commit -m "feat(web): persist vendor feedback across sessions"
```

---

## Done when

- `py -m pytest model_results_api/tests -q` and `py -m pytest tests -q` both pass.
- `npx tsc --noEmit` exits 0.
- An action recorded in the dashboard survives a page refresh.
- `data/feedback_events.jsonl` holds one line per recorded action, with no `client_name`.
