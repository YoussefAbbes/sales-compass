# -*- coding: utf-8 -*-
"""The API must serve, and describe, the window it actually answers for.

Two failures hide behind one symptom here, and both are silent:

  wrong table   pair_scores_rolling.csv is the BACKTEST. Every row carries a
                `label` and a `snapshot` whose horizon has already elapsed, and
                the shipped ledger observes it. Served as a recommendation it
                answers a question the data already answers.

  wrong caption the window text was read off rolling_metrics.json, which records
                the TRAINING splits. A training span is not a serving span, so
                the endpoint described live recommendations with the backtest's
                snapshot dates and nothing contradicted it.

The fix for the first without the second re-arms the same trap on a later fuse:
the live window closes too, and nothing said so.
"""
from datetime import date, timedelta
from fastapi.testclient import TestClient

from model_results_api.app import repository as repo
from model_results_api.app.config import Settings
from model_results_api.app import services
from model_results_api.app.main import app


client = TestClient(app)


# --- the score table ------------------------------------------------------------

def test_repurchase_scores_carry_the_live_schema_not_the_backtest_one():
    """`days_since_last` exists only in the live file and `snapshot` only in the
    rolling one, so the columns identify which table was loaded without reaching
    for the filename."""
    frame = repo.repurchase_scores()

    assert "days_since_last" in frame.columns
    assert "snapshot" not in frame.columns


def test_no_backtest_label_survives_into_the_score_frame():
    """The live file has no label by construction. The drop stays anyway: it is
    the one guard that cannot be defeated by a later change of source."""
    assert "label" not in repo.repurchase_scores().columns


def _rooted_at(path) -> Settings:
    """Settings is a frozen dataclass whose dirs are derived properties, so a test
    swaps the whole instance rather than patching an attribute."""
    return Settings(source_root=path, api_key=None, cors_origins=())


def test_repurchase_readiness_requires_the_live_score_file(scratch_dir, monkeypatch):
    """Readiness that probes the wrong filename reports a healthy service that
    cannot answer, or an unhealthy one that can."""
    data = scratch_dir / "data"
    data.mkdir()
    (data / "clients_to_contact.csv").write_text("x", encoding="utf-8")
    (data / "pair_scores_rolling.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(repo, "settings", _rooted_at(scratch_dir))

    assert repo.repurchase_ready() is False

    (data / "pair_scores_live.csv").write_text("x", encoding="utf-8")
    assert repo.repurchase_ready() is True


# --- the recorded window --------------------------------------------------------

def test_live_window_reports_the_record_written_beside_the_scores():
    record = repo.live_window()

    assert record["snapshot_date"]
    assert record["window_end"] > record["snapshot_date"]


def test_live_window_is_empty_rather_than_guessed_when_the_record_is_absent(
        scratch_dir, monkeypatch):
    """An absent record means unknown. Deriving one from the ledger instead would
    describe the DATA's window, which is not the SCORES' window the moment a stale
    score file is shipped against a fresh ledger."""
    (scratch_dir / "reports").mkdir()
    monkeypatch.setattr(repo, "settings", _rooted_at(scratch_dir))
    repo.live_window.cache_clear()
    try:
        assert repo.live_window() == {}
    finally:
        repo.live_window.cache_clear()


# --- expiry ---------------------------------------------------------------------

def test_window_status_flags_a_window_whose_last_day_has_passed():
    """Probe date derived from the record rather than hardcoded: regenerating the
    fixtures at a later T must not turn this red without a code change."""
    end = date.fromisoformat(repo.live_window()["window_end"])

    status = services.window_status(today=end + timedelta(days=1))

    assert status.expired is True
    assert status.days_remaining == 0


def test_window_status_counts_the_days_still_left_in_an_open_window():
    record = repo.live_window()
    end = date.fromisoformat(record["window_end"])
    status = services.window_status(today=end - timedelta(days=14))

    assert status.expired is False
    assert status.days_remaining == 14


def test_the_last_day_of_the_window_is_still_inside_it():
    """Half-open [T, T+H) is the label convention, but `window_end` is already the
    last day INSIDE it, so the boundary must not be off by one against itself."""
    record = repo.live_window()
    end = date.fromisoformat(record["window_end"])

    status = services.window_status(today=end)

    assert status.expired is False


# --- what the endpoints say -----------------------------------------------------

def test_health_exposes_the_serving_window():
    body = client.get("/api/v1/health").json()

    assert body["repurchase_window"]["snapshot_date"] == \
        repo.live_window()["snapshot_date"]
    assert body["repurchase_window"]["expired"] is False


def test_meta_exposes_the_serving_window():
    body = client.get("/api/v1/meta").json()

    assert body["repurchase_window"]["window_end"] == repo.live_window()["window_end"]


def test_repurchase_window_text_describes_the_served_window(monkeypatch):
    """Not the training snapshots. The old text read splits.test out of
    rolling_metrics.json and printed 2026-01-01..2026-04-01 beside forward-looking
    recommendations."""
    text = services._repurchase_window_text()

    assert repo.live_window()["window_end"] in text
    assert "2026-01-01" not in text


def test_repurchase_notices_no_longer_claim_a_retrospective_demonstration():
    """A live window has no observed outcome to compare against; claiming one
    invites a reader to look for an accuracy that cannot exist yet."""
    vendor = client.get("/api/v1/vendors").json()["items"][0]["vendor_code"]

    body = client.get(f"/api/v1/vendors/{vendor}/repurchases").json()

    assert not any("rétrospective" in n.lower() for n in body["notices"])
    assert body["data_window"] == services._repurchase_window_text()


# --- how the served window is CLASSIFIED ----------------------------------------
#
# `data_window` and the notices were repointed to the serving record, but
# `evaluation_mode` stayed pinned to the literal "retrospective_demo" — a fact the
# response asserts about itself and that no data feeds. The scores are now a
# forecast, so the field contradicted the very window printed beside it.
#
# The replacement is derived rather than pinned: a forecast whose window has closed
# IS a retrospective record, and only the serving record knows which it is.

def test_evaluation_mode_declares_a_forecast_while_the_window_is_open():
    end = date.fromisoformat(repo.live_window()["window_end"])

    assert services.evaluation_mode(today=end - timedelta(days=14)) == "live_forecast"


def test_evaluation_mode_declares_an_expired_forecast_once_the_window_closes():
    """The scores do not change when the window shuts; what they mean does. A
    pinned literal could never say so."""
    end = date.fromisoformat(repo.live_window()["window_end"])

    assert services.evaluation_mode(today=end + timedelta(days=1)) == "expired_forecast"


def test_evaluation_mode_is_unknown_when_no_serving_record_exists(
        scratch_dir, monkeypatch):
    """Absence of the record is not evidence of expiry — the same three-valued
    reading /health already applies to the window itself."""
    (scratch_dir / "reports").mkdir()
    monkeypatch.setattr(repo, "settings", _rooted_at(scratch_dir))
    repo.live_window.cache_clear()
    try:
        assert services.evaluation_mode() == "unknown"
    finally:
        repo.live_window.cache_clear()


def test_repurchase_endpoint_no_longer_declares_a_retrospective_demo():
    vendor = client.get("/api/v1/vendors").json()["items"][0]["vendor_code"]

    body = client.get(f"/api/v1/vendors/{vendor}/repurchases").json()

    assert body["evaluation_mode"] == "live_forecast"


def test_article_recommendations_declare_the_same_mode_as_the_repurchase_endpoint():
    """One response embeds the repurchase block inside another. Two independently
    pinned literals is exactly how the two drifted apart in the first place."""
    vendor = client.get("/api/v1/vendors").json()["items"][0]["vendor_code"]
    article = client.get("/api/v1/articles?limit=1").json()["items"][0]["article_id"]

    body = client.get(
        f"/api/v1/vendors/{vendor}/articles/{article}/recommendations").json()

    assert body["repurchases"]["evaluation_mode"] == "live_forecast"
