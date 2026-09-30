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
    """One JSON object per line, opened O_APPEND.

    Append mode keeps a single process's whole-line writes intact. It does not guard
    against concurrent writes from multiple worker processes interleaving -- that is
    not currently handled.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        # split("\n") rather than splitlines(): splitlines() also breaks on the
        # Unicode line separators U+2028/U+2029 and U+0085, which json.dumps(...,
        # ensure_ascii=False) writes raw when a note contains one (e.g. pasted from
        # Word or a PDF). splitlines() would then cut one event into two
        # unparseable fragments, silently losing it to the corrupt-line skip below.
        for line in self.path.read_text(encoding="utf-8").split("\n"):
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
