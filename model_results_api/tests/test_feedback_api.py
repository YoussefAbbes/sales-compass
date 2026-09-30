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
    services.record_feedback(_payload())

    raw = services.feedback_store.path.read_text(encoding="utf-8")

    assert "client_name" not in raw


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
