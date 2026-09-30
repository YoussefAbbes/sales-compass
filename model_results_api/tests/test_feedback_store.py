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


def test_a_unicode_line_separator_in_a_note_does_not_split_the_event(scratch_dir):
    """U+2028 arrives by pasting from Word or a PDF. str.splitlines() treats it as a
    line break; the JSON writer does not escape it under ensure_ascii=False, so a
    naive read turns one event into two unparseable fragments and loses it silently."""
    store = JsonlFeedbackStore(scratch_dir / "events.jsonl")
    store.append(_event(event_id="u1") | {"note": "ligne un\u2028ligne deux"})

    events = store.read_for_vendor("007")

    assert len(events) == 1
    assert events[0]["note"] == "ligne un\u2028ligne deux"
