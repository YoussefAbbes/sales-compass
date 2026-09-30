# -*- coding: utf-8 -*-
"""Unit tests for src/prospect_model.py. Synthetic frames only — no dependency on data/."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import prospect_features as pf
import prospect_model as pm
from tests.test_prospect_features import make_articles, make_fact


@pytest.fixture
def ctx():
    return pf.build_context(make_fact(), make_articles(), 2024)


@pytest.fixture
def win_next():
    return pf.window(make_fact(), 2025)


def test_adopters_are_candidates_who_bought_it_next_window(ctx, win_next):
    cands = pf.candidates_for(ctx, "X")
    got = pm.adopters(win_next, "X", cands)
    assert got == {"P1"}                      # B1 also bought X in 2025, but is not a candidate


def test_sampling_labels_adopters_positive(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=10, rng=np.random.default_rng(0))
    assert out.loc[out["client_id"] == "P1", "label"].iloc[0] == 1
    assert out["label"].sum() == 1


def test_negatives_come_only_from_this_articles_candidate_pool(ctx, win_next):
    """The whole point of within-article sampling — a negative must be a client who COULD
    have bought this part, not a random client-article pair from elsewhere."""
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=10, rng=np.random.default_rng(0))
    pool = set(pf.candidates_for(ctx, "X"))
    assert set(out["client_id"]) <= pool
    for holder in ["B1", "B2", "B3", "B4", "P4"]:
        assert holder not in set(out["client_id"])


def test_sampling_honours_the_negative_ratio(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "X", neg_ratio=2, rng=np.random.default_rng(0))
    assert (out["label"] == 0).sum() <= 2 * (out["label"] == 1).sum()


def test_sampling_is_deterministic_under_a_seed(ctx, win_next):
    a = pm.sample_article(ctx, win_next, "X", neg_ratio=3, rng=np.random.default_rng(42))
    b = pm.sample_article(ctx, win_next, "X", neg_ratio=3, rng=np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)


def test_article_with_no_adopters_contributes_nothing(ctx, win_next):
    out = pm.sample_article(ctx, win_next, "S1", neg_ratio=5, rng=np.random.default_rng(0))
    assert out.empty


def test_training_set_carries_the_article_id_for_grouping(ctx, win_next):
    out = pm.build_training_set(ctx, win_next, ["X"], neg_ratio=5, seed=1)
    assert "article_id" in out.columns
    assert set(out["article_id"]) == {"X"}
    assert list(out.columns) == pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"]


def test_precision_at_k_counts_hits_in_the_top_k():
    assert pm.precision_at_k(["a", "b", "c"], {"a", "c"}, k=2) == pytest.approx(0.5)
    assert pm.precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
    assert pm.precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_divides_by_k_not_by_the_list_length():
    """Three candidates and one hit is 1/5, not 1/3 — the page shows five slots either way."""
    assert pm.precision_at_k(["a", "b", "c"], {"a"}, k=5) == pytest.approx(0.2)


def test_recall_at_k_is_relative_to_what_was_available():
    assert pm.recall_at_k(["a", "b"], {"a", "c"}, k=2) == pytest.approx(0.5)
    assert pm.recall_at_k(["a"], set(), k=5) == 0.0


def test_evaluate_skips_articles_with_no_adoption(ctx, win_next):
    """S1 gains no new buyer in 2025; including it would measure the catalogue, not the ranker."""
    out = pm.evaluate_ranker(ctx, win_next, ["X", "S1"], pm.heuristic_ranking)
    assert out["n_articles"] == 1


def test_heuristic_ranking_puts_the_strongest_basket_overlap_first(ctx):
    ranked = pm.heuristic_ranking(ctx, "X", ["P3", "P2", "P1"])
    assert ranked[0] == "P1"
    assert ranked.index("P2") < ranked.index("P3") if "P3" in ranked else True


def test_size_ranking_ignores_the_article_entirely(ctx):
    """The trap baseline: it returns the biggest clients whatever the part."""
    a = pm.size_ranking(ctx, "X", ["P1", "P2", "P3"])
    b = pm.size_ranking(ctx, "S1", ["P1", "P2", "P3"])
    assert a == b


def test_popularity_and_random_baselines_return_every_candidate(ctx):
    """The two remaining floors. Random is seeded so a run is reproducible."""
    cands = ["P1", "P2", "P3"]
    assert sorted(pm.popularity_ranking(ctx, "X", cands)) == sorted(cands)
    assert sorted(pm.random_ranking(ctx, "X", cands)) == sorted(cands)
    assert pm.random_ranking(ctx, "X", cands) == pm.random_ranking(ctx, "X", cands)


def test_evaluate_reports_the_heuristic_finding_the_adopter(ctx, win_next):
    out = pm.evaluate_ranker(ctx, win_next, ["X"], pm.heuristic_ranking)
    assert out["precision_at_k"] == pytest.approx(1 / pm.TOP_N)
    assert out["recall_at_k"] == pytest.approx(1.0)


def test_load_model_returns_none_when_no_artifact_exists(scratch_dir):
    assert pm.load_model(scratch_dir / "absent.cbm") is None


def test_rank_prospects_falls_back_to_the_heuristic_without_a_model(ctx):
    out, sig, source = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert source == "heuristic"
    assert list(out["client_id"]) == ["P1", "P2"]
    assert sig


def test_rank_prospects_keeps_the_vendor_scope(ctx):
    """O1 belongs to V2 and must never appear in V1's list."""
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert "O1" not in set(out["client_id"])


def test_rank_prospects_excludes_clients_buying_it_through_another_vendor(ctx):
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None)
    assert "P4" not in set(out["client_id"])


def test_rank_prospects_returns_a_typed_empty_frame_when_the_signature_is_refused(ctx):
    out, sig, _ = pm.rank_prospects(ctx, "Z", "V1", model=None)
    assert sig == []
    assert out.empty
    assert list(out.columns) == pm.RANKED_COLUMNS


def test_rank_prospects_honours_the_top_limit(ctx):
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None, top=1)
    assert list(out["client_id"]) == ["P1"]


def test_rank_prospects_drops_clients_the_caller_excludes(ctx):
    """A context is one window. The page excludes all-time buyers so that a client who took
    the part in an earlier year does not resurface under a "never bought it" heading."""
    out, _, _ = pm.rank_prospects(ctx, "X", "V1", model=None, exclude={"P1"})
    assert "P1" not in set(out["client_id"])
    assert list(out["client_id"]) == ["P2"]


class _FakeModel:
    """Scores P2 above P1, the reverse of the heuristic — proves the model is in charge."""

    def __init__(self, order=None):
        self.order = order if order is not None else {"P1": 0.1, "P2": 0.9, "P3": 0.5}

    def predict_proba(self, pool):
        ids = pool if isinstance(pool, list) else list(pool.index)
        return np.array([[1 - self.order.get(c, 0.0), self.order.get(c, 0.0)] for c in ids])


def test_model_ranking_orders_by_predicted_probability(ctx):
    ranked = pm.model_ranking(ctx, "X", ["P1", "P2", "P3"], _FakeModel())
    assert ranked[0] == "P2"


def test_rank_prospects_reports_the_model_as_its_source(ctx):
    _, _, source = pm.rank_prospects(ctx, "X", "V1", model=_FakeModel())
    assert source == "model"


def _row(**kw):
    """A block-2 output row. Defaults to all-zero so each test sets only what it exercises."""
    base = dict(client_id="C1", n_signature=0, client_ca=0.0, score=1,
                knn_buyers=0, brand_n_held=0, family_n_held=0, client_n_articles=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_reason_never_states_a_zero_valued_fact():
    """The bug this guards: a client with no signature overlap was justified as
    « achète 0 des 30 articles » — an argument against calling them."""
    text = pm.prospect_reason(_row(client_n_articles=340), ["S1"] * 30, "MECO", "FILTRE")
    assert "0 des" not in text
    assert "0.0" not in text
    assert "déjà **0**" not in text


def test_reason_falls_back_to_a_verifiable_fact_when_nothing_overlaps():
    text = pm.prospect_reason(_row(client_n_articles=340), ["S1"] * 30, "MECO", "FILTRE")
    assert "340" in text
    assert "mais jamais cette pièce" in text


def test_reason_leads_with_the_models_strongest_signal():
    """knn_buyers outranks the signature count in feature importance, so it comes first."""
    text = pm.prospect_reason(_row(knn_buyers=3, n_signature=7), ["S1"] * 30)
    assert text.index("garages au panier") < text.index("articles-repères")


def test_reason_renders_counts_as_integers_not_floats():
    """affinity_features is a float frame; 0.0/1.0 must never reach the page."""
    text = pm.prospect_reason(_row(knn_buyers=3.0, n_signature=1.0), ["S1"] * 30)
    assert "3.0" not in text and "1.0" not in text
    assert "**3 des 20**" in text
    assert "**1 des 30**" in text


def test_reason_skips_brand_and_family_when_the_article_has_none():
    """38.6% of articles carry no name, so brand and family are often blank."""
    text = pm.prospect_reason(_row(brand_n_held=5, family_n_held=5, client_n_articles=12),
                              ["S1"] * 30, brand="", family="")
    assert "pièces" not in text
    assert "12" in text


def test_reason_agrees_in_number_in_french():
    """« déjà 1 pièces FEBI » reads as a typo to a vendor."""
    one = pm.prospect_reason(_row(brand_n_held=1, family_n_held=1), [], "FEBI", "PATE")
    assert "**1** pièce FEBI" in one
    assert "**1** autre PATE" in one
    many = pm.prospect_reason(_row(brand_n_held=14, family_n_held=3), [], "FEBI", "PATE")
    assert "**14** pièces FEBI" in many
    assert "**3** autres PATE" in many
