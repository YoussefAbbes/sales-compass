# -*- coding: utf-8 -*-
"""Training-set construction, evaluation and scoring for the block-2 prospect model.

Deliberately free of any Streamlit import: the page is a display layer, and this module has to
be unit-testable and callable from a training script or an API route without a browser.

The sampling rule here is the one design decision most able to invalidate the whole model, and
it is the reason this file exists separately from prospect_features.py — see sample_article().
"""
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import prospect_features as pf

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"
REPORTS = DATA_ROOT / "reports"

MODEL_PATH = DATA / "prospect_model.cbm"

NEG_RATIO = 20          # negatives drawn per positive, within the same article
SEED = 7
TOP_N = pf.ap.TOP_N


def adopters(win_next: pd.DataFrame, article_id: str, candidates) -> set:
    """Candidates who went on to buy the article in the label window.

    Intersected with `candidates` on purpose: a client who already held the article is a repeat
    buyer, not an adoption, and counting them would train the model on block 1's population.
    """
    bought = pf.ap.buyers_of(win_next, article_id)
    return bought & set(candidates)


def sample_article(ctx: pf.WindowContext, win_next: pd.DataFrame, article_id: str,
                   neg_ratio: int, rng) -> pd.DataFrame:
    """Positives and sampled negatives for ONE article.

    Negatives are drawn from this article's own candidate pool, never from the global grid.
    The page ranks clients within a single article, so training must pose the same question:
    "among clients who could buy X, which ones will". Sampling across articles instead teaches
    the model to recognise plausible articles, which inflates AUC and ranks nothing usefully.

    Articles with no adoption in the label window are skipped: they contribute negatives only,
    and a group with no positive cannot inform a within-article ranking.
    """
    cands = pf.candidates_for(ctx, article_id)
    if not cands:
        return pd.DataFrame()

    pos = adopters(win_next, article_id, cands)
    if not pos:
        return pd.DataFrame()

    neg_pool = [c for c in cands if c not in pos]
    n_neg = min(len(neg_pool), neg_ratio * len(pos))
    neg = list(rng.choice(neg_pool, size=n_neg, replace=False)) if n_neg else []

    chosen = sorted(pos) + sorted(neg)
    feats = pf.build_features(ctx, article_id, chosen)
    feats = feats.reset_index()
    feats["article_id"] = article_id
    feats["label"] = feats["client_id"].isin(pos).astype(int)
    return feats[pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"]]


def build_training_set(ctx: pf.WindowContext, win_next: pd.DataFrame, article_ids,
                       neg_ratio: int = NEG_RATIO, seed: int = SEED) -> pd.DataFrame:
    """Stack per-article samples into one frame, keeping article_id so groups stay visible."""
    rng = np.random.default_rng(seed)
    parts = [sample_article(ctx, win_next, a, neg_ratio, rng) for a in article_ids]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=pf.FEATURE_COLUMNS + ["client_id", "article_id", "label"])
    return pd.concat(parts, ignore_index=True)


def precision_at_k(ranked, actual, k: int = TOP_N) -> float:
    """Hits in the top k, divided by k.

    Divided by k rather than by len(ranked): the page always offers five slots, so a short
    list that fills three of them has still spent five slots' worth of the vendor's attention.
    """
    if not k:
        return 0.0
    return len([c for c in list(ranked)[:k] if c in actual]) / k


def recall_at_k(ranked, actual, k: int = TOP_N) -> float:
    """Share of the window's actual adopters that the top k captured."""
    if not actual:
        return 0.0
    return len([c for c in list(ranked)[:k] if c in actual]) / len(actual)


def heuristic_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """The incumbent: rank by signature overlap, ties broken by client CA.

    Reproduces article_prospects.prospects() over an explicit client list so the two can be
    scored on identical candidates. This is the bar the model has to clear.
    """
    sig = pf.signature_with_lift(ctx.win, article_id)
    aff = pf.affinity_features(ctx, article_id, client_ids, sig)
    ca = ctx.client_feats["client_ca"].reindex(list(client_ids)).fillna(0.0)
    order = pd.DataFrame({"n": aff["n_signature_held"].values, "ca": ca.values},
                         index=list(client_ids))
    order = order[order["n"] > 0].sort_values(["n", "ca"], ascending=False)
    return list(order.index)


def size_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: biggest clients first, article ignored.

    Named explicitly because it is the specific failure the signature was built to avoid —
    a ranking that returns the same large clients for every part in the catalogue.
    """
    ca = ctx.client_feats["client_ca"].reindex(list(client_ids)).fillna(0.0)
    return list(ca.sort_values(ascending=False).index)


def popularity_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: clients who buy the widest range of articles first.

    Distinct from size_ranking: a client can spend heavily on few references, or little across
    many. Breadth is the more plausible naive predictor of trying something new, so it is the
    tougher of the two floors.
    """
    n = ctx.client_feats["client_n_articles"].reindex(list(client_ids)).fillna(0.0)
    return list(n.sort_values(ascending=False).index)


def random_ranking(ctx: pf.WindowContext, article_id: str, client_ids) -> list:
    """Floor baseline: shuffle. Seeded per article so a run is reproducible.

    Seeding on the article id rather than a global counter keeps each article's shuffle
    independent of the order articles happen to be evaluated in.

    crc32, not hash(): Python randomises string hashing per process unless PYTHONHASHSEED is
    set, so hash() would give a different "reproducible" baseline on every run.
    """
    ids = list(client_ids)
    rng = np.random.default_rng(zlib.crc32(str(article_id).encode("utf-8")))
    return [ids[i] for i in rng.permutation(len(ids))]


def evaluate_ranker(ctx: pf.WindowContext, win_next: pd.DataFrame, article_ids,
                    rank_fn, k: int = TOP_N) -> dict:
    """Average precision@k and recall@k over articles that actually gained a new buyer.

    Articles with no adoption are excluded: precision@k is 0 for them however good the
    ranking, so averaging them in would measure the catalogue's adoption rate instead of the
    ranker. Their count is reported so the coverage stays visible.
    """
    precisions, recalls, used = [], [], 0
    for article_id in article_ids:
        cands = pf.candidates_for(ctx, article_id)
        if not cands:
            continue
        actual = adopters(win_next, article_id, cands)
        if not actual:
            continue
        ranked = rank_fn(ctx, article_id, cands)
        precisions.append(precision_at_k(ranked, actual, k))
        recalls.append(recall_at_k(ranked, actual, k))
        used += 1
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "n_articles": used,
    }


# Everything the page needs to state WHY a client is here. The ranking and the explanation
# used to come from one function, so they could not disagree; once the model took over the
# ordering, a reason built only from n_signature started contradicting it — clients with zero
# signature overlap ranked highly and were explained as "buys 0 of the 30 related articles".
RANKED_COLUMNS = ["client_id", "n_signature", "client_ca", "score",
                  "knn_buyers", "brand_n_held", "family_n_held", "client_n_articles"]


def prospect_reason(r, signature, brand: str = "", family: str = "") -> str:
    """Why this client is in the list, stated only in facts that are actually true of them.

    Lives here rather than in the page for the same reason every other rule does: it is pure
    logic, and putting it behind a Streamlit import would make it untestable except by driving
    the whole page, where whichever article the fixture happens to pick decides the coverage.

    Every clause is skipped when its value is zero. Before this, the reason was always the
    signature count, which produced « achète 0 des 30 articles » as the justification for
    calling someone — an argument against the recommendation it was attached to.

    Clauses are ordered by the model's own feature importance, so the vendor reads the
    strongest reason the ranking actually used first. No percentage appears: the ordering was
    validated on held-out data, the calibration was not.
    """
    bits = []
    if r.knn_buyers:
        bits.append(f"**{int(r.knn_buyers)} des {pf.KNN_K}** garages au panier le plus proche "
                    f"prennent cette pièce")
    if r.n_signature:
        bits.append(f"achète **{int(r.n_signature)} des {len(signature)}** articles-repères "
                    f"de ses acheteurs")
    if r.brand_n_held and brand:
        n = int(r.brand_n_held)
        bits.append(f"déjà **{n}** pièce{'s' if n > 1 else ''} {brand}")
    if r.family_n_held and family:
        n = int(r.family_n_held)
        bits.append(f"déjà **{n}** autre{'s' if n > 1 else ''} {family}")
    if not bits:
        # Nothing specific overlaps, so the model ranked them on the shape of the basket as a
        # whole. State the one thing that is verifiable rather than inventing a closer link.
        bits.append(f"panier large — **{int(r.client_n_articles)}** références — au profil "
                    f"proche de celui des acheteurs")
    return " · ".join(bits) + ", mais jamais cette pièce"


def load_model(path=MODEL_PATH):
    """The fitted CatBoost model, or None when it has not been trained yet.

    Returning None rather than raising keeps a fresh clone working: the page falls back to the
    heuristic and renders normally instead of failing on a missing artifact in data/, which is
    gitignored and therefore absent for everyone who has not run training.
    """
    from catboost import CatBoostClassifier

    path = Path(path)
    if not path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def model_ranking(ctx: pf.WindowContext, article_id: str, client_ids, model) -> list:
    """Rank candidates by the model's probability of adoption, highest first."""
    client_ids = list(client_ids)
    if not client_ids:
        return []
    feats = pf.build_features(ctx, article_id, client_ids)
    proba = model.predict_proba(feats)[:, 1]
    return list(pd.Series(proba, index=feats.index).sort_values(ascending=False).index)


def rank_prospects(ctx: pf.WindowContext, article_id: str, vendor_code: str,
                   model=None, top: int = TOP_N, exclude=()):
    """Block 2, model-ranked when a model exists and heuristic-ranked when it does not.

    Returns (frame, signature, source). `source` is surfaced so the page — and any future API
    route — can tell the two apart, and so a screenshot of the page is self-describing.

    The vendor filter is applied AFTER scoring: the model answers "who wants this part", and
    ownership is a separate question about who is allowed to be shown it.

    `exclude` removes clients the caller knows about but the window does not. The page passes
    every all-time buyer of the article: a context is one window, so a client who bought the
    part in 2024 but not in the current one would otherwise resurface as a prospect, and the
    page promises clients who have *never* taken it.
    """
    signature = list(pf.signature_with_lift(ctx.win, article_id).index)
    empty = pd.DataFrame(columns=RANKED_COLUMNS)
    if not signature:
        return empty, [], "heuristic"

    owned = pf.ap.vendor_clients(ctx.win, vendor_code)
    exclude = set(exclude)
    candidates = [c for c in pf.candidates_for(ctx, article_id)
                  if c in owned and c not in exclude]
    if not candidates:
        return empty, signature, "heuristic"

    source = "model" if model is not None else "heuristic"
    ranked = (model_ranking(ctx, article_id, candidates, model) if model is not None
              else heuristic_ranking(ctx, article_id, candidates))
    if not ranked:
        return empty, signature, source

    sig = pf.signature_with_lift(ctx.win, article_id)
    aff = pf.affinity_features(ctx, article_id, ranked, sig).reindex(ranked)
    feats = ctx.client_feats.reindex(ranked).fillna(0.0)
    out = pd.DataFrame({
        "client_id": ranked,
        "n_signature": aff["n_signature_held"].values.astype(int),
        "client_ca": feats["client_ca"].values,
        "score": range(1, len(ranked) + 1),
        "knn_buyers": aff["knn_buyers"].values.astype(int),
        "brand_n_held": aff["brand_n_held"].values.astype(int),
        "family_n_held": aff["family_n_held"].values.astype(int),
        "client_n_articles": feats["client_n_articles"].values.astype(int),
    })
    return out.head(top).reset_index(drop=True), signature, source
