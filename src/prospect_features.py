# -*- coding: utf-8 -*-
"""Features for the block-2 prospect model — one builder shared by training and inference.

The page and the training script call the SAME functions, so a feature can never be computed
one way when fitting and another way when serving. That failure (train/serve skew) is silent:
the model scores well offline and disappoints in the field.

No feature may look outside the snapshot window it is given. src/train_model.py enforces the
same rule, because 2023 does not exist for the training split and the two halves of the split
must see feature distributions built the same way.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_prospects as ap

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"

FAMILY_UNKNOWN = "INCONNU"
KNN_K = 20               # neighbours considered by the knn_buyers feature


def family_of(designation) -> str:
    """Product family = first token of the designation, upper-cased.

    Verified on the catalogue: 616 distinct first tokens, the top 200 covering 97% of named
    articles. Two tokens is far too sparse (3,260 values, 63% of them singletons). 38.6% of
    articles carry no designation at all, but they are only 8.2% of revenue.
    """
    if designation is None or (isinstance(designation, float) and np.isnan(designation)):
        return FAMILY_UNKNOWN
    tokens = str(designation).strip().upper().split()
    return tokens[0] if tokens else FAMILY_UNKNOWN


def family_map(articles: pd.DataFrame) -> dict:
    """article_id -> family, for the 61.4% of articles that appear in articles.csv.

    The rest resolve to FAMILY_UNKNOWN at lookup time via .get(), so callers never need to
    know which articles are named.
    """
    return {a: family_of(d)
            for a, d in zip(articles["article_id"], articles["designation"])}


def window(fact: pd.DataFrame, period) -> pd.DataFrame:
    """The positive rows of one observation window.

    `period` is a calendar year (2024) or a half-year ("2024-H1", "2024-H2").

    Half-years exist because the NAV ledger ends 30 June 2026. Calendar years
    give a 12 / 12 / 6 split, so the test label window would be HALF the length
    of the training one — a 6-month adoption rate is structurally lower than a
    12-month one, which makes it a different question rather than merely less
    data. Half-years make every window the same size.

    `ca > 0` is the project-wide definition of a purchase: returns are netted into CA by the
    source system, so a negative row is not a purchase.
    """
    pos = fact[fact["ca"] > 0]
    if isinstance(period, (int, np.integer)):
        return pos[pos["month"].dt.year == int(period)]

    year, _, half = str(period).partition("-H")
    if half not in ("1", "2"):
        raise ValueError(f"period must be a year or 'YYYY-H1'/'YYYY-H2', got {period!r}")
    lo, hi = (1, 6) if half == "1" else (7, 12)
    m = pos["month"]
    return pos[(m.dt.year == int(year)) & (m.dt.month.between(lo, hi))]


def latest_complete_half(fact: pd.DataFrame) -> str:
    """The newest half-year the data fully covers, as "YYYY-H1" / "YYYY-H2".

    The adoption model is trained on complete six-month windows. Serving it a
    partial one would hand it half the history it learned thresholds on — the
    same train/serve skew that feeding it twelve-month windows would cause,
    just in the other direction. Derived, never hardcoded, so this stays right
    when the next NAV export lands.
    """
    last = fact["month"].max()
    if pd.isna(last):
        raise ValueError("no complete half-year window in the data")
    last_year, last_month = int(last.year), int(last.month)

    # "Complete" means the observation period reaches the window's end, not that
    # every month inside it traded. A month with no sales is absent from a
    # ca_monthly-shaped frame, and treating that as an incomplete window would
    # discard a perfectly good window for a quiet December.
    for year in range(last_year, last_year - 3, -1):
        for half, end_month in (("2", 12), ("1", 6)):
            if (year, end_month) <= (last_year, last_month):
                return f"{year}-H{half}"
    raise ValueError("no complete half-year window in the data")


def basket_matrix(win: pd.DataFrame):
    """Sparse binary client x article matrix for one window.

    Sparse because the dense form is ~1,600 x 40,000 floats (512 MB); as CSR it is a few MB.
    Binary rather than CA-weighted: the question is whether the client buys the part at all,
    and weighting by amount would let a handful of expensive lines dominate the similarity.

    Returns (matrix, clients, articles) — the two Index objects map ids to row/column numbers.
    """
    pairs = win[["client_id", "article_id"]].drop_duplicates()
    clients = pd.Index(sorted(pairs["client_id"].unique()))
    articles = pd.Index(sorted(pairs["article_id"].unique()))
    rows = clients.get_indexer(pairs["client_id"])
    cols = articles.get_indexer(pairs["article_id"])
    matrix = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (rows, cols)),
        shape=(len(clients), len(articles)))
    return matrix, clients, articles


def client_similarity(matrix) -> np.ndarray:
    """Dense pairwise cosine similarity between client baskets.

    Dense is affordable here and nowhere else: the result is clients x clients (~1,600^2,
    about 10 MB), not clients x articles. The diagonal is zeroed so a client is never its own
    nearest neighbour in knn_buyers.
    """
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    return sim


CLIENT_COLUMNS = ["client_ca", "client_n_articles", "client_n_months",
                  "client_n_vendors", "client_n_families", "client_mean_ca_per_article"]

ARTICLE_COLUMNS = ["article_n_buyers", "article_ca", "article_mean_ca_per_buyer",
                   "article_n_vendors", "enseigne", "family"]


def client_features(win: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """One row per client active in the window, indexed by client_id.

    Every feature is behavioural because clients.csv carries only a name — there is no sector,
    region or size attribute anywhere in the extract to use instead.
    """
    w = win.copy()
    w["family"] = w["article_id"].map(fam).fillna(FAMILY_UNKNOWN)
    g = w.groupby("client_id")
    out = pd.DataFrame({
        "client_ca": g["ca"].sum(),
        "client_n_articles": g["article_id"].nunique(),
        "client_n_months": g["month"].nunique(),
        "client_n_vendors": g["vendor_code"].nunique(),
        "client_n_families": g["family"].nunique(),
    })
    out["client_mean_ca_per_article"] = out["client_ca"] / out["client_n_articles"]
    return out[CLIENT_COLUMNS]


def article_features(win: pd.DataFrame, articles: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """One row per article sold in the window, indexed by article_id.

    `mean_ca_per_buyer` stands in for a price band: the extract carries no unit price, and the
    monthly CA of a part divided by its buyers is the closest available proxy.
    """
    g = win.groupby("article_id")
    out = pd.DataFrame({
        "article_n_buyers": g["client_id"].nunique(),
        "article_ca": g["ca"].sum(),
        "article_n_vendors": g["vendor_code"].nunique(),
    })
    out["article_mean_ca_per_buyer"] = out["article_ca"] / out["article_n_buyers"]
    ens = dict(zip(articles["article_id"], articles["enseigne"].fillna("")))
    ids = pd.Series(out.index, index=out.index)
    out["enseigne"] = ids.map(ens).fillna("")
    out["family"] = ids.map(fam).fillna(FAMILY_UNKNOWN)
    return out[ARTICLE_COLUMNS]


@dataclass
class WindowContext:
    """Everything derived from one snapshot window, computed once and reused.

    Training loops over thousands of articles and the page re-runs on every interaction; both
    would otherwise rebuild the basket matrix and the similarity grid per article. Holding them
    here makes the expensive work happen once per window instead.

    `brand_holdings` and `family_holdings` exist for the same reason: without them, every call
    to affinity_features re-maps all ~273,000 window rows to their brand and family, which
    measured at 720 ms per article — 24 minutes per ranker over an evaluation run.
    """
    win: pd.DataFrame
    fam: dict
    clients: pd.Index
    articles: pd.Index
    matrix: object            # scipy.sparse.csr_matrix
    similarity: np.ndarray
    client_feats: pd.DataFrame
    article_feats: pd.DataFrame
    brand_holdings: pd.DataFrame
    family_holdings: pd.DataFrame


def _holding_aggregates(win: pd.DataFrame, key_by_article: dict, missing: str) -> pd.DataFrame:
    """Per (client, key) count of distinct articles and total CA, where key is brand or family.

    Indexed by a (client_id, key) MultiIndex so a lookup for one brand is a reindex rather than
    a scan.
    """
    per_pair = win.groupby(["client_id", "article_id"], as_index=False)["ca"].sum()
    per_pair["key"] = per_pair["article_id"].map(key_by_article).fillna(missing)
    return per_pair.groupby(["client_id", "key"]).agg(n=("article_id", "nunique"),
                                                      ca=("ca", "sum"))


def build_context(fact: pd.DataFrame, articles_ref: pd.DataFrame, period) -> WindowContext:
    """Slice one window and precompute every window-level structure.

    Nothing outside `period` is read. A context built for the training window physically cannot
    see the label window, which is what makes the leakage test meaningful rather than
    aspirational.
    """
    win = window(fact, period)
    fam = family_map(articles_ref)
    matrix, clients, articles = basket_matrix(win)
    art_feats = article_features(win, articles_ref, fam)
    ens = art_feats["enseigne"].to_dict()
    return WindowContext(
        win=win,
        fam=fam,
        clients=clients,
        articles=articles,
        matrix=matrix,
        similarity=client_similarity(matrix),
        client_feats=client_features(win, fam),
        article_feats=art_feats,
        brand_holdings=_holding_aggregates(win, ens, ""),
        family_holdings=_holding_aggregates(win, fam, FAMILY_UNKNOWN),
    )


AFFINITY_COLUMNS = ["n_signature_held", "share_signature_held", "lift_weighted_overlap",
                    "brand_n_held", "brand_ca_share", "family_n_held", "family_ca_share",
                    "cosine_basket_sim", "knn_buyers"]


def signature_with_lift(win: pd.DataFrame, article_id: str,
                        min_share: float = ap.MIN_SHARE, min_lift: float = ap.MIN_LIFT,
                        top: int = ap.TOP_SIGNATURE) -> pd.Series:
    """Signature articles WITH their lift, rather than just their ids.

    article_prospects.signature_articles() discards the lift values; the model wants them, so
    that a client holding three strongly-associated articles outranks one holding three
    marginal ones. Same thresholds, same ordering — only the return type differs.
    """
    buyers = ap.buyers_of(win, article_id)
    if len(buyers) < ap.MIN_BUYERS:
        return pd.Series(dtype=float)

    pairs = win[["client_id", "article_id"]].drop_duplicates()
    among = pairs[pairs["client_id"].isin(buyers)]
    share_buyers = among["article_id"].value_counts() / len(buyers)
    share_all = pairs["article_id"].value_counts() / pairs["client_id"].nunique()

    lift = (share_buyers / share_all.reindex(share_buyers.index)).dropna()
    keep = lift[(share_buyers.reindex(lift.index) >= min_share) & (lift > min_lift)]
    keep = keep.drop(labels=[article_id], errors="ignore")
    return keep.sort_values(ascending=False).head(top)


def affinity_features(ctx: WindowContext, article_id: str, client_ids,
                      sig: pd.Series, k: int = KNN_K) -> pd.DataFrame:
    """Pair features that need NO history for the pair itself — the heart of the model.

    Every one of these is computable for a client who has never bought the article, which is
    exactly the population the CatBoost pair model cannot score. `n_signature_held` is today's
    entire block-2 ranking; here it is one input among nine, with its weight learned.
    """
    client_ids = list(client_ids)
    out = pd.DataFrame(0.0, index=pd.Index(client_ids, name="client_id"),
                       columns=AFFINITY_COLUMNS)
    sig_set = set(sig.index)

    # ---- signature overlap, plain and lift-weighted ----
    # Filtered by article BEFORE client: the signature is at most 30 articles, so this
    # intermediate stays tiny. Filtering by client first materialises nearly the whole window.
    if sig_set:
        rows = ctx.win[ctx.win["article_id"].isin(sig_set)]
        ov = (rows[rows["client_id"].isin(set(client_ids))]
              [["client_id", "article_id"]].drop_duplicates())
        counts = ov.groupby("client_id").size()
        weights = ov.assign(w=ov["article_id"].map(sig)).groupby("client_id")["w"].sum()
        out["n_signature_held"] = counts.reindex(client_ids).fillna(0.0).values
        out["share_signature_held"] = out["n_signature_held"] / len(sig_set)
        out["lift_weighted_overlap"] = weights.reindex(client_ids).fillna(0.0).values

    # ---- brand and family holdings, read from the precomputed context aggregates ----
    # The target article's own contribution is subtracted, so "holds 3 articles of this brand"
    # never counts the article being prospected. Candidates never hold it, but the function
    # stays correct if called with someone who does.
    target_brand = ctx.article_feats["enseigne"].get(article_id, "")
    target_family = ctx.fam.get(article_id, FAMILY_UNKNOWN)

    tgt_ca = (ctx.win[ctx.win["article_id"] == article_id]
              .groupby("client_id")["ca"].sum().reindex(client_ids).fillna(0.0).values)
    holds = (tgt_ca > 0).astype(float)
    total_ca = (ctx.client_feats["client_ca"].reindex(client_ids).fillna(0.0).values - tgt_ca)

    for holdings, col, value, skip_blank in [
            (ctx.brand_holdings, "brand", target_brand, True),
            (ctx.family_holdings, "family", target_family, False)]:
        if skip_blank and not value:
            continue
        idx = pd.MultiIndex.from_arrays([client_ids, [value] * len(client_ids)])
        sub = holdings.reindex(idx)
        n = sub["n"].fillna(0.0).values - holds
        ca = sub["ca"].fillna(0.0).values - tgt_ca
        out[f"{col}_n_held"] = np.maximum(n, 0.0)
        out[f"{col}_ca_share"] = np.divide(ca, total_ca, out=np.zeros_like(ca),
                                           where=total_ca > 0)

    # ---- basket cosine against the article's buyer profile, and knn ----
    buyers = ap.buyers_of(ctx.win, article_id)
    known = [c for c in client_ids if c in ctx.clients]
    if buyers and known:
        buyer_rows = ctx.clients.get_indexer([b for b in buyers if b in ctx.clients])
        buyer_rows = buyer_rows[buyer_rows >= 0]
        if len(buyer_rows):
            profile = np.asarray(ctx.matrix[buyer_rows].mean(axis=0))
            rows = ctx.clients.get_indexer(known)
            out.loc[known, "cosine_basket_sim"] = cosine_similarity(
                ctx.matrix[rows], profile).ravel()

            is_buyer = np.zeros(len(ctx.clients), dtype=bool)
            is_buyer[buyer_rows] = True
            take = min(k, len(ctx.clients) - 1)
            # One batched argpartition over all candidates at once. Looping per client instead
            # costs ~1.2 s per article on the real data (1,569 candidates x 1,577 clients),
            # which the page pays on every interaction, not just training.
            nearest = np.argpartition(-ctx.similarity[rows], take, axis=1)[:, :k]
            out.loc[known, "knn_buyers"] = is_buyer[nearest].sum(axis=1).astype(float)

    return out[AFFINITY_COLUMNS]


FEATURE_COLUMNS = CLIENT_COLUMNS + ARTICLE_COLUMNS + AFFINITY_COLUMNS
CATEGORICAL_COLUMNS = ["enseigne", "family"]


def candidates_for(ctx: WindowContext, article_id: str) -> list:
    """Clients active in the window who do NOT already hold the article.

    Vendor scoping is deliberately absent: the model scores a client-article couple, and the
    vendor filter is applied afterwards at display time. Training on vendor-filtered candidates
    would make the model learn each vendor's book rather than the article's appeal.
    """
    holders = ap.buyers_of(ctx.win, article_id)
    return sorted(set(ctx.clients) - holders)


def build_features(ctx: WindowContext, article_id: str, client_ids=None) -> pd.DataFrame:
    """The single feature builder, called identically by training and by the page.

    Article-level values are broadcast across the rows: CatBoost sees them as constants within
    one article's group, which is exactly the comparison the page asks for — rank these clients
    against each other for THIS part.
    """
    if client_ids is None:
        client_ids = candidates_for(ctx, article_id)
    client_ids = list(client_ids)
    if not client_ids:
        return pd.DataFrame(columns=FEATURE_COLUMNS,
                            index=pd.Index([], name="client_id"))

    sig = signature_with_lift(ctx.win, article_id)
    out = ctx.client_feats.reindex(client_ids).fillna(0.0)
    out.index.name = "client_id"
    out = out.join(affinity_features(ctx, article_id, client_ids, sig))

    art = ctx.article_feats.reindex([article_id]).iloc[0]
    for col in ARTICLE_COLUMNS:
        value = art[col]
        if col in CATEGORICAL_COLUMNS:
            out[col] = FAMILY_UNKNOWN if pd.isna(value) else str(value)
        else:
            out[col] = 0.0 if pd.isna(value) else float(value)

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    return out[FEATURE_COLUMNS]
