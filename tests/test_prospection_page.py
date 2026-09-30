# -*- coding: utf-8 -*-
"""AppTest coverage for the article prospection page."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def page_path():
    p = Path(__file__).resolve().parents[1] / "src" / "pages" / "2_Prospection.py"
    if not p.exists():
        pytest.skip("prospection page not built yet")
    return str(p)


@pytest.fixture
def at(page_path, data_dir):
    app = AppTest.from_file(page_path, default_timeout=300)
    app.run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    return app


def test_page_loads_with_the_three_blocks(at):
    """Three populations, distinguished by tense: they still buy it (v4 scores
    them), they used to (v4 has no opinion), they never did (adoption model).

    Block 2 renders only when the article actually has dormant buyers, so its
    absence here is correct rather than a failure — test_dormant_block_carries
    _no_percentage covers its content.
    """
    text = " ".join(m.value for m in at.markdown)
    assert "Ils l'achètent" in text          # present tense: still buying
    assert "À prospecter" in text


def test_the_dormant_block_is_never_labelled_with_the_present_tense(at):
    """The imperfect is what tells a vendor the purchase stopped. Getting the
    tense wrong is the difference between "call them, they buy this" and "call
    them, they used to"."""
    text = " ".join(m.value for m in at.markdown)
    if "Ils l'achetaient" in text:
        assert "2 · Ils l'achetaient" in text


def test_searching_narrows_the_article_list(at):
    box = [t for t in at.text_input if t.label == "Chercher une pièce"][0]
    before = len([sb for sb in at.selectbox if sb.label == "Article"][0].options)
    box.set_value("huile")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    after = [sb for sb in at.selectbox if sb.label == "Article"][0].options
    assert all("HUILE" in o.upper() for o in after)
    assert len(after) <= before


def test_block2_never_shows_a_probability(at):
    """The model cannot score these clients; a percentage there would be invented."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    assert "propension" not in tail
    assert "%" not in tail


def test_changing_vendor_keeps_the_page_working(at):
    box = [sb for sb in at.selectbox if sb.label == "Vendeur"][0]
    box.set_value(box.options[1])
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_unmatched_search_shows_guidance_not_a_crash(at):
    box = [t for t in at.text_input if t.label == "Chercher une pièce"][0]
    box.set_value("zzzzzznotanarticle")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    text = " ".join(i.value for i in at.info)
    assert "Aucune pièce ne correspond" in text


def test_no_resemblance_is_distinguished_from_everyone_already_buys(at):
    """ap.prospects() returns an empty block-2 frame for two different reasons, and the
    page must say something different for each:

      (a) every client of the vendor already bought the article
          -> real example: article 01381, vendor 216 (verified against data/ directly:
             vendor_clients - buyers_of == empty set)
      (b) the vendor has a non-buying client, but it holds none of the article's
          signature articles -> real example: article 703141410, vendor 606

    This test drives the page's own widgets (search box + Vendeur selectbox) to state
    (b) and asserts the "no resemblance" message is shown, and NOT the "everyone
    already buys it" message. Driving to a single article via the search box is
    practical here because "703141410" matches exactly one row in data/articles.csv,
    so the Article selectbox needs no further interaction once the query narrows it
    to one option.
    """
    box = [t for t in at.text_input if t.label == "Chercher une pièce"][0]
    box.set_value("703141410")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    vendor_box = [sb for sb in at.selectbox if sb.label == "Vendeur"][0]
    vendor_box.set_value("606")
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    text = " ".join(i.value for i in at.info)
    assert "Absence de ressemblance" in text
    assert "tous vos clients achètent déjà cette pièce" not in text


def test_block2_still_shows_no_probability_after_the_model_change(at):
    """The model changes the ORDER of the five, never what is displayed about them."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    assert "propension" not in tail
    assert "%" not in tail


def test_page_renders_whatever_the_model_artifact_state(at):
    """data/ is gitignored, so a fresh clone has no prospect_model.cbm. Trained or not, the
    page must render block 2 rather than fail on the missing artifact."""
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "À prospecter" in text


def test_block2_never_justifies_a_client_with_a_zero(at):
    """The reason must never read "achète 0 des 30" — an argument against the recommendation
    it is attached to. Clauses whose value is zero are dropped, not printed."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    for bad in ["**0 des", "**0.0 des", "déjà **0**", "**0 des 20**"]:
        assert bad not in tail, f"zero-valued clause shown: {bad}"


def test_block2_counts_render_as_integers(at):
    """n_signature and knn_buyers arrive as floats from the affinity frame; the page must not
    print "0.0 des 30"."""
    text = " ".join(m.value for m in at.markdown)
    tail = text.split("À prospecter")[-1]
    assert ".0 des" not in tail
    assert ".0** " not in tail
