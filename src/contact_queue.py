# -*- coding: utf-8 -*-
"""
"Clients to contact today" — one ranked queue per vendor, combining:
  1. lapsed reorders     (article-level: bought 2024 & 2025, nothing in 2026)
  2. client reactivation (client-level: inactive or declining vs 2025)
  3. cross-sell          (one complementary suggestion for the top article)

Inputs : data/ca_monthly.parquet, lapsed_reorders.csv, client_reactivation.csv,
         cross_sell_pairs.csv, vendors/clients/articles lookups
Output : data/clients_to_contact.csv   (one row per client, sorted by vendor + priority)

Priority logic (all in TND, deliberately simple — to validate with stakeholders):
  valeur_attendue  = sum over the client's top-3 lapsed articles of
                     p_model x CA_2025   (expected repurchase value)
  manque_a_gagner  = max(0, CA Jan-Apr 2025 - CA Jan-Apr 2026)   [flagged clients only]
  priority_score   = max(valeur_attendue, manque_a_gagner) * status boost
Max, not sum: the revenue gap already partly contains the lapsed articles
(summing would double-count the same missed revenue).

p_model comes from the v4 embargoed rolling model (src/train_rolling.py, test AUC
0.7743, converged at iteration 1345 — reports/v4-embargoed/), scoring the LIVE snapshot
written by src/score_live.py: T = 2026-07-01, predicting purchases in 2026-07..09.
It falls back to the v1 pair_scores_2026.csv, then to the manual score. Probabilities
are shown to vendors only as confidence labels (Élevée/Moyenne/Faible) — calibration
not yet validated (docs/background/project.md §9). Explanations stay rule-based.

BASIS: on the v4 path both halves of valeur_attendue = p_model x pair_ca come from the
NAV ledger — pair_ca is the pair's own revenue over the snapshot's 12-month feature
window. The v1 fallback path still multiplies by ca_2025 from the older Jan-May extract,
which reports/reconciliation.md measures disagreeing with the ledger by up to 9.89% per
month (11 of 15 months outside the 1% tolerance). Read valeur_attendue as reconciled
only when the v4 universe is in use, which the run prints on startup.

Ownership rule: one row per client. The vendor with the largest historical CA on the
lapsed pairs owns the row; reactivation-only clients go to their 2025 main vendor.
Clients flagged as closed/blocked (***) are excluded.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from article_prospects import recency_damping  # noqa: E402
from ledger import COLUMNS as _LEDGER_COLS, monthly_ca  # noqa: E402


def ledger_columns():
    return _LEDGER_COLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---- Tunable business weights ----
TOP_ARTICLES = 3                 # articles shown per client
RECENT_DAYS = 90                 # beyond this, never call a purchase "récent"
BOOST_INACTIVE = 1.2             # client with zero 2026 activity
BOOST_DECLINING = 1.1            # client in strong decline


def attach_model_scores(lapsed: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add p_model, preferring the v4 live scores.

    Order: pair_scores_live.csv (v4, future window) -> pair_scores_2026.csv
    (v1, backtest basis) -> the manual score. Each fallback is a genuine
    degradation, so the winner is printed rather than left to be inferred.
    """
    live = data_dir / "pair_scores_live.csv"
    v1 = data_dir / "pair_scores_2026.csv"

    if live.exists():
        scores = pd.read_csv(live, dtype={"client_id": str, "article_id": str},
                             encoding="utf-8-sig").rename(columns={"p": "p_model"})
        source = "pair_scores_live.csv (v4, live)"
    elif v1.exists():
        scores = pd.read_csv(v1, dtype={"client_id": str, "article_id": str},
                             encoding="utf-8-sig")
        source = "pair_scores_2026.csv (v1)"
    else:
        print("[model] no score file -> falling back to manual score")
        lapsed = lapsed.copy()
        lapsed["p_model"] = (lapsed["score"] / lapsed["score"].max()).clip(0, 1)
        return lapsed

    lapsed = lapsed.merge(scores[["client_id", "article_id", "p_model"]],
                          on=["client_id", "article_id"], how="left")
    joined = lapsed["p_model"].notna().mean()
    lapsed["p_model"] = lapsed["p_model"].fillna(lapsed["p_model"].median())
    print(f"[model] {source}: p_model joined on {joined:.1%} of lapsed pairs")
    return lapsed


def _as_blocked(col) -> "pd.Series":
    """Parse a blocked column that may be bool OR text.

    csv has no boolean type and the queue reads clients.csv with dtype=str, so
    this arrives as "True"/"False". astype(bool) on the string "False" is True,
    which flags every client and empties the queue entirely - observed, not
    hypothetical. Parse the value; never cast it.
    """
    if col.dtype == bool:
        return col.fillna(False)
    return col.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _flagged(df: pd.DataFrame) -> pd.Series:
    """Is this client excluded from the queue?

    NAV's `blocked` column when the lookup carries it, otherwise the old "***"
    in the name. That convention existed only because the January-May Excel had
    nowhere else to put the flag, and it saw 56 blocked clients where NAV sees
    233 - it cannot see a non-payer whose name looks ordinary.
    """
    if "blocked" in df.columns:
        return _as_blocked(df["blocked"])
    return df["client_name"].str.contains(r"\*", na=False)


def _v4_reason(row) -> str:
    """Rule-based, never model-generated. A vendor must be able to check the
    sentence against the ledger themselves."""
    days, interval, n = row["days_since_last"], row["median_interval"], row["n_orders"]
    if pd.notna(interval) and interval > 0 and days > interval * 1.5:
        return (f"Achat en retard — habituellement tous les {interval:.0f} j, "
                f"dernier il y a {days:.0f} j")
    # Checked before the regular-buyer branch on purpose. A client with 27 orders on
    # a 12-day cadence who ordered yesterday used to read "Réachat régulier — 27
    # commandes sur 12 mois": true, and the reason they rank at all, but not a
    # reason to call them today. Recency is the operative fact and outranks the
    # label. It cannot collide with "en retard" above — the two are exclusive.
    if pd.notna(interval) and interval > 0 and days < interval:
        return (f"Servi récemment — il y a {days:.0f} j, "
                f"rythme habituel {interval:.0f} j")
    if n >= 3:
        return f"Réachat régulier — {int(n)} commandes sur 12 mois"
    if days <= RECENT_DAYS:
        return f"Achat récent — il y a {days:.0f} j"
    # No cadence to compare against and not recent: state the fact and claim
    # nothing. "Acheteur récent — il y a 357 j" reads as a contradiction of the
    # very number beside it.
    return f"Dernier achat il y a {days:.0f} j"



def client_damping(lapsed: pd.DataFrame) -> pd.Series:
    """Per-client recency multiplier for priority_score, indexed by client_id.

    priority_score is the sort key behind every queue surface, and every term in it
    could only ever raise a client: BOOST_INACTIVE and BOOST_DECLINING are both
    >= 1.0. Nothing could lower one for having just taken delivery, so a client on a
    12-day cadence who ordered yesterday sat near the top of the call list.

    Read across ALL of a client's lines, not just the one the queue pitches: the
    constraint is "do not ring someone you dealt with yesterday", which is true
    whatever article the pitch is about. Hence the client's most recent line
    (min days_since_last) against their fastest-cycling one (min median_interval) --
    a client whose quickest line turns every 12 days transacts at least that often,
    so it is the closest proxy for their own contact cadence in a pair-level frame.

    Degrades to all-ones on the v1 fallback universe, which carries no cadence
    columns -- the queue must still build there, the same way attach_model_scores
    degrades rather than raises.
    """
    clients = pd.Index(lapsed["client_id"].unique(), name="client_id")
    if not {"days_since_last", "median_interval"} <= set(lapsed.columns):
        return pd.Series(1.0, index=clients)

    grouped = lapsed.groupby("client_id").agg(
        days=("days_since_last", "min"), interval=("median_interval", "min"))
    damping = pd.Series(
        [recency_damping(d, m) for d, m in zip(grouped["days"], grouped["interval"])],
        index=grouped.index)
    # reindex, never a bare join: a client absent from the factor would multiply to
    # NaN and sort to the bottom -- a silent deletion dressed as a demotion.
    return damping.reindex(clients).fillna(1.0)


def build_v4_candidates(live: pd.DataFrame, articles: pd.DataFrame,
                        clients: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Turn v4 live scores into the pair-level frame the queue assembles from.

    Shaped like lapsed_reorders.csv so the downstream assembly is unchanged, but
    built on the MODEL'S OWN candidate universe — pairs with a purchase in the
    last 12 months — rather than the v1 "lapsed" construct.

    That distinction is not cosmetic. The reported precision@top-30 of 61.9%
    (reports/v4-embargoed/) was measured on this universe. Ranking a different
    population with the same scores would not be covered by that validation, and
    the lapsed universe only overlaps it by 37%.
    """
    out = live.rename(columns={"p": "p_model", "amount_sum": "pair_ca"}).copy()

    out = out.merge(articles[["article_id", "enseigne", "designation"]],
                    on="article_id", how="left")
    out["enseigne"] = out["enseigne"].fillna("")
    out["designation"] = out["designation"].fillna("(désignation inconnue)")

    cols = ["client_id", "client_name"] + (["blocked"] if "blocked" in clients.columns
                                            else [])
    out = out.merge(clients[cols], on="client_id", how="left")
    out["client_name"] = out["client_name"].fillna("(nom non disponible)")
    out["client_flagged"] = _flagged(out)

    out = out.merge(vendors[["vendor_code", "vendor_name"]],
                    on="vendor_code", how="left")
    out["vendor_name"] = out["vendor_name"].fillna(out["vendor_code"])

    out["reason"] = out.apply(_v4_reason, axis=1)
    out["score"] = out["p_model"]        # only the manual fallback reads this
    return out


def main():
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    # Full ledger, not ca_monthly.parquet: that file holds January-May only, so
    # "last activity" and the cross-sell "already owns this" set both had a
    # seven-month hole in them. See src/ledger.py.
    fact = monthly_ca(pd.read_parquet(DATA / "transactions.parquet",
                                      columns=ledger_columns()))
    react = pd.read_csv(DATA / "client_reactivation.csv", dtype={"vendor_code": str, "client_id": str}, encoding="utf-8-sig")
    pairs = pd.read_csv(DATA / "cross_sell_pairs.csv", dtype={"article_id": str, "partner_article_id": str}, encoding="utf-8-sig")
    vendors = pd.read_csv(DATA / "vendors.csv", dtype=str, encoding="utf-8-sig")
    clients = pd.read_csv(DATA / "clients.csv", dtype=str, encoding="utf-8-sig")

    # ---- Pair universe: the model's own candidates when they exist ----
    live_path = DATA / "pair_scores_live.csv"
    if live_path.exists():
        live = pd.read_csv(live_path, encoding="utf-8-sig",
                           dtype={"client_id": str, "article_id": str,
                                  "vendor_code": str})
        articles = pd.read_csv(DATA / "articles.csv", dtype={"article_id": str},
                               encoding="utf-8-sig")
        lapsed = build_v4_candidates(live, articles, clients, vendors)
        print(f"[model] v4 live universe: {len(lapsed):,} pairs, "
              f"{lapsed['client_id'].nunique():,} clients")
    else:
        lapsed = pd.read_csv(DATA / "lapsed_reorders.csv", encoding="utf-8-sig",
                             dtype={"vendor_code": str, "client_id": str,
                                    "article_id": str})
        lapsed = lapsed.rename(columns={"ca_2025": "pair_ca"})
        print("[model] no live scores -> v1 lapsed universe")
        lapsed = attach_model_scores(lapsed, DATA)

    lapsed = lapsed[~lapsed["client_flagged"]]
    react = react[~react["client_flagged"]]

    lapsed["valeur_attendue_pair"] = lapsed["p_model"] * lapsed["pair_ca"]
    # The confidence LABEL is no longer appended to the reason: the app and the Excel
    # sheets both show the probability itself now, and "51 %" next to "Confiance: Élevée"
    # is redundant and reads as a contradiction. Kept as a column for anyone who prefers
    # the banded form.
    lapsed["confiance"] = np.select(
        [lapsed["p_model"] >= 0.5, lapsed["p_model"] >= 0.25],
        ["Élevée", "Moyenne"], default="Faible",
    )

    # ---- 1. Reorder block: top-3 articles per client by expected value ----
    lp = lapsed.sort_values("valeur_attendue_pair", ascending=False)
    top = lp.groupby("client_id", group_keys=False).head(TOP_ARTICLES).copy()
    top["rank"] = top.groupby("client_id").cumcount() + 1
    top["article_txt"] = (
        top["article_id"].fillna("") + " " + top["enseigne"].fillna("")
        + " — " + top["designation"].fillna("(désignation inconnue)")
    ).str.strip()

    # Raw model probability, shown per article in the app next to the label.
    # NOT calibrated: an isotonic correction fitted on the validation split was measured
    # and made calibration WORSE on the test window (reports/model_catboost.md
    # "Calibration"). Correcting honestly would need a 2023 basis that does not exist, so
    # the number is shown raw and the UI states that it is optimistic.
    top["proba"] = (top["p_model"] * 100).round(1)

    wide = top.pivot(index="client_id", columns="rank",
                     values=["article_txt", "reason", "proba"])
    wide.columns = [f"{a}_{r}" for a, r in wide.columns]
    wide = wide.reset_index()

    agg = (
        top.groupby("client_id")
        .agg(vendor_code=("vendor_code", "first"),      # dominant vendor of best pair
             valeur_attendue=("valeur_attendue_pair", "sum"),
             ca_lapsed=("pair_ca", "sum"),              # plain figure shown to vendors
             n_lapsed=("article_id", "size"),
             top_article_id=("article_id", "first"))
        .reset_index()
    )
    n_all = lapsed.groupby("client_id")["article_id"].size().rename("n_lapsed_total")
    reorder = agg.merge(wide, on="client_id").merge(n_all, on="client_id")

    # ---- 2. Reactivation block ----
    r = react[["client_id", "vendor_code", "status", "ca_2025_ja", "ca_2026_ja", "drop_pct"]].copy()
    r["manque_a_gagner"] = (r["ca_2025_ja"] - r["ca_2026_ja"]).clip(lower=0).round(3)
    r = r.rename(columns={"vendor_code": "vendor_react"})

    # ---- Combine: one row per client ----
    q = reorder.merge(r.drop(columns=["ca_2025_ja", "ca_2026_ja"]), on="client_id", how="outer")
    q["vendor_code"] = q["vendor_code"].fillna(q["vendor_react"])
    q = q.drop(columns=["vendor_react"])
    q["status"] = q["status"].fillna("ACTIVE")
    for col in ["valeur_attendue", "manque_a_gagner", "n_lapsed", "n_lapsed_total"]:
        q[col] = q[col].fillna(0)

    boost = np.where(q["status"] == "INACTIVE_2026", BOOST_INACTIVE,
             np.where(q["status"] == "DECLINING", BOOST_DECLINING, 1.0))
    # Damping applies to priority_score alone. valeur_attendue and manque_a_gagner
    # stay untouched: those are the figures a vendor says out loud, and a client
    # served yesterday has not become worth less money -- only less worth ringing
    # today. Same split as article_prospects.existing_customers.
    damp = q["client_id"].map(client_damping(lapsed)).fillna(1.0).to_numpy()
    q["priority_score"] = (np.maximum(q["valeur_attendue"], q["manque_a_gagner"])
                           * boost * damp).round(1)

    # ---- Last activity per client ----
    pos = fact[fact["ca"] > 0]
    last = pos.groupby("client_id")["month"].max().dt.strftime("%Y-%m").rename("last_activity")
    q = q.merge(last, on="client_id", how="left")

    # ---- 3. Cross-sell: best partner of top article that the client never bought ----
    bought = pos[["client_id", "article_id"]].drop_duplicates()
    cand = (
        q[["client_id", "top_article_id"]].dropna()
        .merge(pairs, left_on="top_article_id", right_on="article_id")
        .merge(bought, left_on=["client_id", "partner_article_id"],
               right_on=["client_id", "article_id"], how="left", suffixes=("", "_owned"))
    )
    cand = cand[cand["article_id_owned"].isna()]
    cand = cand.sort_values("lift", ascending=False).drop_duplicates("client_id")
    cand["suggestion_complementaire"] = (
        cand["partner_article_id"] + " " + cand["partner_enseigne"].fillna("")
        + " — " + cand["partner_designation"].fillna("")
    ).str.strip()
    q = q.merge(cand[["client_id", "suggestion_complementaire"]], on="client_id", how="left")

    # ---- Final shape ----
    q = q.merge(vendors, on="vendor_code", how="left").merge(clients, on="client_id", how="left")
    q["client_name"] = q["client_name"].fillna("(nom non disponible)")
    q["resume"] = np.select(
        [q["status"] == "INACTIVE_2026", q["status"] == "DECLINING"],
        ["Client inactif en 2026 (CA 2025 Jan-Avr: " + q["manque_a_gagner"].round(0).astype(int).astype(str) + " TND)",
         "CA en baisse de " + (q["drop_pct"].fillna(0) * -100).round(0).astype(int).astype(str) + "% vs 2025"],
        default="Client actif — articles habituels non rachetés en 2026",
    )

    # A figure a vendor can say out loud, unlike priority_score which is a blended
    # ranking score (max of a model estimate and a revenue drop, times a boost) and
    # corresponds to nothing physical. priority_score still drives the ORDER and the
    # management summary; this pair is what the vendor screens display.
    q["ca_lapsed"] = q["ca_lapsed"].fillna(0.0)
    q["montant_cle"] = np.where(
        q["status"].isin(["INACTIVE_2026", "DECLINING"]),
        q["manque_a_gagner"].fillna(0.0),
        q["ca_lapsed"],
    ).round(0)
    q["montant_texte"] = np.select(
        [q["status"] == "INACTIVE_2026", q["status"] == "DECLINING"],
        ["achetés l'an dernier, rien cette année", "de baisse vs l'an dernier"],
        default="d'articles habituels non rachetés",
    )
    cols = ["vendor_code", "vendor_name", "client_id", "client_name", "status",
            "priority_score", "valeur_attendue", "manque_a_gagner",
            "montant_cle", "montant_texte",
            "n_lapsed_total", "last_activity", "resume",
            "article_txt_1", "reason_1", "proba_1",
            "article_txt_2", "reason_2", "proba_2",
            "article_txt_3", "reason_3", "proba_3",
            "suggestion_complementaire"]
    q = (
        q.reindex(columns=cols)
        .sort_values(["vendor_code", "priority_score"], ascending=[True, False])
    )
    q.to_csv(DATA / "clients_to_contact.csv", index=False, encoding="utf-8-sig")

    # ---- Report ----
    print("=== CLIENTS TO CONTACT ===")
    print(f"rows (clients): {len(q):,}  | vendors: {q.vendor_code.nunique()}")
    print(f"status mix: {q.status.value_counts().to_dict()}")
    print(f"total TND at stake (max-based): {q.priority_score.sum():,.0f}")
    per_v = q.groupby("vendor_code").size()
    print(f"clients per vendor: median={per_v.median():.0f}  min={per_v.min()}  max={per_v.max()}")
    print("\nTop 5 rows:")
    print(q.head(5)[["vendor_name", "client_name", "status", "priority_score", "resume"]].to_string(index=False))


if __name__ == "__main__":
    main()
