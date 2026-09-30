# -*- coding: utf-8 -*-
"""AppTest coverage for src/app.py (the vendor queue page).

Recovered from a session scratchpad and converted from a bare assertion script into
pytest: repo-relative paths, isolated test functions, and a fixture that protects the
real data/feedback.csv from test writes.
"""
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


def test_clicking_a_queue_row_opens_that_client(app_path, data_dir):
    """Master/detail: clicking a row in the left list must swap the right-hand file.

    The second row is used deliberately — the first is already open by default, so
    clicking it would prove nothing.
    """
    at = run_app(app_path)
    rows = [b for b in at.button if b.key and b.key.startswith("open_top_")]
    assert len(rows) >= 2, "expected at least two clients in the queue"
    target = rows[1]
    expected_client_id = target.key.replace("open_top_", "")
    target.click()
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    dossier = " ".join(m.value for m in at.markdown if "dossier-head" in m.value
                       or "pane-title" in m.value)
    assert expected_client_id in dossier, \
        "the open file did not follow the clicked row"


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


def test_recommended_articles_show_a_repurchase_probability(app_path, data_dir):
    """The probability must be visible per article, with its optimism stated.

    Showing a bare percentage would be misleading: the v4 probabilities are not
    calibrated (reports/model_rolling.md), so they are a ranking rather than a
    real likelihood, and the UI has to say so.
    """
    at = run_app(app_path)
    top_buttons = [b for b in at.button if b.key and b.key.startswith("open_top_")]
    assert top_buttons, "no top-card buttons rendered"
    top_buttons[0].click()
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    blocks = " ".join(m.value for m in at.markdown)
    captions = " ".join(c.value for c in at.caption)
    page = blocks + " " + captions

    # Assert the intent, not the exact wording: rephrasing the copy is fine, dropping
    # the number or its caveat is not.
    assert "proba-line" in blocks, "no per-article probability rendered"
    assert "chance" in blocks, "the probability is not explained in words"
    assert "optimiste" in page, \
        "the probability must always be labelled as optimistic — never shown bare"


def test_step_numbers_are_consecutive_even_when_a_section_is_absent(app_path, data_dir):
    """Steps are issued by a counter, never hardcoded.

    A client with no cross-sell suggestion must still read 1, 2, 3 — not 1, 2, 4.
    A hardcoded number survived an earlier refactor and produced exactly that.
    """
    import re

    at = run_app(app_path)
    rows = [b for b in at.button if b.key and b.key.startswith("open_top_")]

    def step_numbers(app):
        found = []
        for m in app.markdown:
            found += re.findall(r'class="step"><span class="num">(\d+)</span>', m.value)
        return [int(n) for n in found]

    # Pick one client WITH a cross-sell suggestion and one WITHOUT, so the
    # varying-section case is exercised deliberately rather than by luck.
    queue = pd.read_csv(data_dir / "clients_to_contact.csv",
                        dtype={"client_id": str, "vendor_code": str},
                        encoding="utf-8-sig")
    shown_ids = [b.key.replace("open_top_", "") for b in rows]
    scope = queue[queue["client_id"].isin(shown_ids)]
    with_sugg = scope[scope["suggestion_complementaire"].notna()]["client_id"].tolist()
    without_sugg = scope[scope["suggestion_complementaire"].isna()]["client_id"].tolist()
    assert with_sugg and without_sugg, \
        "this vendor has only one case; pick a vendor that has both"

    seen_lengths = set()
    for client_id in (with_sugg[0], without_sugg[0]):
        {b.key: b for b in at.button}[f"open_top_{client_id}"].click()
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        nums = step_numbers(at)
        assert nums, "no numbered steps rendered"
        assert nums == list(range(1, len(nums) + 1)), \
            f"step numbers skip or repeat for {client_id}: {nums}"
        seen_lengths.add(len(nums))

    assert len(seen_lengths) == 2, (
        "both clients rendered the same number of steps — the cross-sell section is "
        "not actually conditional, so this test proves nothing"
    )
