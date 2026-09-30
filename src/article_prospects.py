# -*- coding: utf-8 -*-
"""Article-first prospection: given an article, which clients are worth calling?

The reverse of the call queue. Deliberately free of any Streamlit import so it can be
unit-tested without a browser — src/pages/2_Prospection.py is only a display layer.

Two populations, two signals, never mixed:
  block 1  clients who already bought the article -> the CatBoost model scores them
  block 2  clients who never did -> the model has NO pair features for them, so they
           are ranked by basket similarity instead, and carry no probability.
"""
import math
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data"

# ---- Business thresholds (to validate with stakeholders) ----
MIN_SHARE = 0.15        # an article is in the signature if >= 15% of buyers take it
MIN_LIFT = 1.2          # ... and it is 1.2x more common among buyers than catalogue-wide
TOP_SIGNATURE = 30      # signature articles kept, by lift
MIN_BUYERS = 3          # below this, a signature is noise rather than a pattern
TOP_N = 5               # clients shown per block
RECENCY_FLOOR = 0.05    # a client served today keeps a twentieth of their priority
RECENCY_CURVE = 2       # squared: punishes early-cycle hard, near-due barely at all

UNKNOWN_DESIGNATION = "(désignation inconnue)"


def search_articles(articles: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame:
    """Free-text search over reference, brand and designation.

    Reference is searched too: 15,638 of 40,321 articles carry no designation and would
    otherwise be unreachable.
    """
    out = articles.copy()
    out["designation"] = out["designation"].fillna(UNKNOWN_DESIGNATION)
    out["enseigne"] = out["enseigne"].fillna("")
    q = (query or "").strip().lower()
    if q:
        hay = (out["article_id"].str.lower() + " " + out["enseigne"].str.lower()
               + " " + out["designation"].str.lower())
        out = out[hay.str.contains(q, regex=False, na=False)]
    out["label"] = (out["article_id"] + " " + out["enseigne"]
                    + " — " + out["designation"]).str.replace("  ", " ", regex=False)
    return out[["article_id", "enseigne", "designation", "label"]].head(limit)


def _positive(fact: pd.DataFrame) -> pd.DataFrame:
    """A purchase is a month with net CA > 0 — the project-wide definition.

    Returns are netted into CA by the source system, so a negative row is not a purchase
    and must not make someone a buyer.
    """
    return fact[fact["ca"] > 0]


def vendor_clients(fact: pd.DataFrame, vendor_code: str) -> set:
    """Clients this vendor has actually traded with.

    Ownership comes from the fact table, never from the scored-pair files: their
    vendor_code is the pair's DOMINANT vendor, and 78% of clients are served by several
    vendors, so the two disagree constantly.
    """
    pos = _positive(fact)
    return set(pos.loc[pos["vendor_code"] == vendor_code, "client_id"])


def buyers_of(fact: pd.DataFrame, article_id: str) -> set:
    """Clients who bought this article under ANY vendor.

    Vendor-blind on purpose: a client already buying the part through a colleague is not
    a prospect, and pitching it to them would expose that colleague's business.
    """
    pos = _positive(fact)
    return set(pos.loc[pos["article_id"] == article_id, "client_id"])


BLOCK1_COLUMNS = ["client_id", "p_model", "ca_sum", "expected_value"]

# Evidence, not ranking inputs: they let the page say why THIS client ranks rather
# than restate the formula, and they are what the model ranks on --
# client_days_since_last is its strongest feature -- so the explanation is phrased in
# the terms that produced the order.
#
# Carried only when the score table has them. Requiring them would make an
# explanation a precondition of a ranking that never reads it, and force every
# caller to supply columns it does not use.
EVIDENCE_COLUMNS = ["n_orders", "days_since_last", "median_interval"]


def recency_damping(days_since_last, median_interval) -> float:
    """How much of a client's priority survives having just been served.

    The repurchase model answers "does this pair transact within three months?".
    A client on a 12-day cadence answers yes at ~91% the day after they order, and
    the ranking used to read that as "call them". It is the right answer to the
    wrong question: the label's horizon is a quarter, the call list's horizon is
    today, and nothing converted between the two.

    Returns a multiplier in [RECENCY_FLOOR, 1.0], applied to the money term:

        ratio >= 1.0   due or overdue        -> 1.0, untouched
        ratio  < 1.0   served early in cycle -> ratio ** RECENCY_CURVE, floored

    Squared, not linear. Linear damping cannot separate "a day or two early" from
    "served yesterday" without a floor low enough to be indistinguishable from
    exclusion: at ratio 0.9 and 0.08 it gives 0.9 and 0.2, a 4.5x gap for two cases
    a vendor treats completely differently. Squared gives 0.81 and 0.05 -- a 16x
    gap, leaving a nearly-due client essentially where they were while a
    just-served one drops out of the visible top. Measured on the live scores, it
    takes the just-served share of the global top-30 from 12 to 4.

    It only ever removes priority. Lateness is already the model's strongest
    feature (client_days_since_last, 18% importance) and must not be paid twice.

    A floor rather than zero: the largest accounts are usually the most frequent
    ones, and dropping them off the vendor's screen the day they order trades one
    wrong answer for another. Demoted stays reachable; excluded does not.

    No cadence means no damping. median_interval is NaN for a client with a single
    purchase -- snapshots.py returns NaN rather than 0 so "no cadence yet" stays
    distinct from "bought twice the same day" -- and with no cadence there is no
    "too soon" to measure. Inventing one would penalise exactly the new accounts
    a vendor most needs to see.
    """
    if median_interval is None:
        return 1.0
    try:
        interval = float(median_interval)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(interval) or interval <= 0:
        return 1.0
    days = float(days_since_last or 0)
    ratio = days / interval
    if ratio >= 1.0:
        return 1.0
    return max(RECENCY_FLOOR, ratio ** RECENCY_CURVE)


def _damping_series(frame: pd.DataFrame) -> pd.Series:
    """Per-row damping, or all-ones when the frame carries no cadence columns.

    Evidence columns are optional by design (see EVIDENCE_COLUMNS): a score table
    without them must still rank rather than raise. Absent them, damping is a
    no-op instead of a precondition.
    """
    if not {"days_since_last", "median_interval"} <= set(frame.columns):
        return pd.Series(1.0, index=frame.index)
    return pd.Series(
        [recency_damping(d, m) for d, m
         in zip(frame["days_since_last"], frame["median_interval"])],
        index=frame.index)


def existing_customers(scores: pd.DataFrame, fact: pd.DataFrame, article_id: str,
                       vendor_code: str, top: int = TOP_N) -> pd.DataFrame:
    """Block 1 — this vendor's clients who already bought the article, model-ranked.

    Ranked by expected value (p x prior-year CA), the same rule as the call queue, so
    that "#1" means the same thing on every screen: the most money at stake, not the
    surest bet.
    """
    owned = vendor_clients(fact, vendor_code)
    sub = scores[(scores["article_id"] == article_id)
                 & (scores["client_id"].isin(owned))].copy()
    if sub.empty:
        return pd.DataFrame(columns=BLOCK1_COLUMNS)
    sub["expected_value"] = sub["p_model"] * sub["ca_sum"]
    # Rank by the damped figure, report the undamped one. expected_value is money a
    # vendor can say out loud; the damped product is a blended number that
    # corresponds to nothing physical, the same line contact_queue.py draws for
    # priority_score. The demotion is explained on the page by overdue_ratio, which
    # is already displayed, rather than hidden inside a TND amount.
    ranking = sub["expected_value"] * _damping_series(sub)
    columns = BLOCK1_COLUMNS + [c for c in EVIDENCE_COLUMNS if c in sub.columns]
    return (sub.assign(_ranking=ranking)
               .sort_values("_ranking", ascending=False)
               .head(top)[columns]
               .reset_index(drop=True))


DORMANT_COLUMNS = ["client_id", "article_ca", "last_month"]


def dormant_buyers(fact: pd.DataFrame, scored_pairs: pd.DataFrame, article_id: str,
                   vendor_code: str, top: int = TOP_N) -> pd.DataFrame:
    """Clients of this vendor who bought the article and then stopped.

    The v4 snapshot only scores pairs active inside its 12-month feature window,
    which is half of all buyer pairs. Without this block the other half vanishes
    from the page: no score drops them out of the model-ranked block, and the
    prospect block excludes them because they ARE buyers.

    Ranked by what they used to spend on the part, since that is the size of what
    was lost. `last_month` lets the vendor say when it stopped.

    No probability is produced, and none must be displayed: the model has no
    features for a pair outside its window, so any number here would be invented
    — the same rule `prospects()` already follows.
    """
    owned = vendor_clients(fact, vendor_code)
    buyers = buyers_of(fact, article_id)

    scored: set = set()
    if len(scored_pairs) and "article_id" in scored_pairs.columns:
        scored = set(scored_pairs.loc[scored_pairs["article_id"] == article_id,
                                      "client_id"])

    candidates = (owned & buyers) - scored
    if not candidates:
        return pd.DataFrame(columns=DORMANT_COLUMNS)

    pos = _positive(fact)
    sub = pos[(pos["article_id"] == article_id) & (pos["client_id"].isin(candidates))]
    if sub.empty:
        return pd.DataFrame(columns=DORMANT_COLUMNS)

    out = (sub.groupby("client_id")
              .agg(article_ca=("ca", "sum"), last_month=("month", "max"))
              .reset_index())
    return (out.sort_values("article_ca", ascending=False)
               .head(top)[DORMANT_COLUMNS]
               .reset_index(drop=True))


def signature_articles(fact: pd.DataFrame, article_id: str,
                       min_share: float = MIN_SHARE, min_lift: float = MIN_LIFT,
                       top: int = TOP_SIGNATURE) -> list:
    """Articles over-represented among the buyers of `article_id`.

    Lift, not raw co-occurrence: without dividing by the catalogue-wide rate, the answer
    is always the best-selling articles, which say nothing about this one in particular.
    Below MIN_BUYERS the ratio is arithmetic noise, so the signature is refused outright
    rather than returned weak — the page then says so instead of showing a bad list.
    """
    pos = _positive(fact)
    buyers = buyers_of(fact, article_id)
    if len(buyers) < MIN_BUYERS:
        return []

    pairs = pos[["client_id", "article_id"]].drop_duplicates()
    among = pairs[pairs["client_id"].isin(buyers)]
    share_buyers = among["article_id"].value_counts() / len(buyers)
    n_clients = pairs["client_id"].nunique()
    share_all = pairs["article_id"].value_counts() / n_clients

    lift = (share_buyers / share_all.reindex(share_buyers.index)).dropna()
    keep = lift[(share_buyers.reindex(lift.index) >= min_share) & (lift > min_lift)]
    keep = keep.drop(labels=[article_id], errors="ignore")
    return list(keep.sort_values(ascending=False).head(top).index)


BLOCK2_COLUMNS = ["client_id", "n_signature", "client_ca"]


def prospects(fact: pd.DataFrame, article_id: str, vendor_code: str,
              top: int = TOP_N) -> tuple:
    """Block 2 — this vendor's clients who never bought the article, by basket similarity.

    Returns (frame, signature) so the page can phrase the reason as "23 of the 30
    articles that buyers of this part usually take".

    No probability is produced, and none must be displayed: the model has no pair
    features for a couple that has never traded, so any number here would be invented.
    """
    signature = signature_articles(fact, article_id)
    if not signature:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), []

    owned = vendor_clients(fact, vendor_code)
    buyers = buyers_of(fact, article_id)
    candidates = owned - buyers
    if not candidates:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), signature

    pos = _positive(fact)
    held = (pos[pos["client_id"].isin(candidates) & pos["article_id"].isin(signature)]
            [["client_id", "article_id"]].drop_duplicates())
    if held.empty:
        return pd.DataFrame(columns=BLOCK2_COLUMNS), signature

    out = held.groupby("client_id").size().rename("n_signature").reset_index()
    ca = pos.groupby("client_id")["ca"].sum().rename("client_ca")
    out = out.merge(ca, on="client_id", how="left")
    return (out.sort_values(["n_signature", "client_ca"], ascending=False)
               .head(top)[BLOCK2_COLUMNS]
               .reset_index(drop=True),
            signature)
