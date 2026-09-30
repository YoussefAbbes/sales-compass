# -*- coding: utf-8 -*-
"""
Baseline recommendation lists built from data/ca_monthly.parquet (no ML).

Outputs (data/):
  lapsed_reorders.csv          (client, article) bought in 2024 AND 2025, nothing in 2026
  lapsed_top30_per_vendor.csv  top 30 lapsed opportunities per vendor (for review)
  client_reactivation.csv      clients inactive or in strong decline in 2026
  cross_sell_pairs.csv         articles frequently bought in the same client-month basket

Method notes:
  - A "purchase" = a month with net CA > 0 (net-negative months = returns, not purchases).
  - Lapsed pairs are computed at (client, article) level across ALL vendors, then
    attributed to the vendor with the largest historical CA on that pair. This avoids
    flagging a pair as lapsed when the client simply switched vendors.
  - Reactivation drop % uses Jan-Apr only (May 2026 may be incomplete); May shown as context.
  - Cross-sell: basket = (client, month). Metric = lift, minimum 5 shared baskets.

Tunable business thresholds are the constants below.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---- Business thresholds (to validate with stakeholders) ----
MIN_CLIENT_CA_2025 = 500.0   # TND Jan-Apr 2025: ignore tiny clients in reactivation
DECLINE_THRESHOLD = -0.30    # flag clients whose Jan-Apr CA dropped >= 30% vs 2025
TOP_ARTICLES_XSELL = 5000    # co-occurrence computed on the N most-bought articles
MIN_BASKETS_TOGETHER = 5     # minimum shared baskets for a cross-sell pair
TOP_PARTNERS = 10            # partners kept per article
TOP_PER_VENDOR = 30          # rows in the per-vendor review file


def load():
    df = pd.read_parquet(DATA / "ca_monthly.parquet")
    df["year"] = df["month"].dt.year
    df["mois"] = df["month"].dt.month
    vendors = pd.read_csv(DATA / "vendors.csv", dtype=str)
    clients = pd.read_csv(DATA / "clients.csv", dtype=str)
    articles = pd.read_csv(DATA / "articles.csv", dtype=str)
    return df, vendors, clients, articles


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


def flagged(df: pd.DataFrame) -> pd.Series:
    """NAV's blocked column when present, else the legacy "***" in the name."""
    if "blocked" in df.columns:
        return _as_blocked(df["blocked"])
    return df["client_name"].str.contains(r"\*", na=False)


def months_observed(df: pd.DataFrame, year: int) -> int:
    """How many distinct months of that year the data actually contains.

    Derived, never assumed. This module was written when the extract only ever
    held January-May, so "5" and "10" were safe to hardcode. The NAV ledger has
    30 months: a fixed denominator would print "Commandé 8 mois sur 5" to a
    vendor and silently rescale every score.
    """
    return int(df.loc[df["year"] == year, "mois"].nunique())


def lapsed_reorders(df, vendors, clients, articles):
    """(client, article) purchased in 2024 AND 2025, no purchase in 2026."""
    pos = df[df["ca"] > 0]

    # per (client, article, year): months bought + CA
    g = (
        pos.groupby(["client_id", "article_id", "year"])
        .agg(months=("mois", "nunique"), ca=("ca", "sum"))
        .unstack("year", fill_value=0)
    )
    g.columns = [f"{a}_{y}" for a, y in g.columns]
    g = g.reset_index()

    lapsed = g[(g["months_2024"] > 0) & (g["months_2025"] > 0) & (g["months_2026"] == 0)].copy()

    # last observed purchase month per pair
    last = (
        pos.groupby(["client_id", "article_id"])["month"].max().dt.strftime("%Y-%m")
        .rename("last_purchase")
    )
    lapsed = lapsed.merge(last, on=["client_id", "article_id"], how="left")

    # dominant vendor on the pair = vendor with the largest historical positive CA
    vend_ca = (
        pos.groupby(["client_id", "article_id", "vendor_code"])["ca"].sum().reset_index()
        .sort_values("ca", ascending=False)
        .drop_duplicates(["client_id", "article_id"])
        [["client_id", "article_id", "vendor_code"]]
    )
    lapsed = lapsed.merge(vend_ca, on=["client_id", "article_id"], how="left")

    lapsed["ca_hist"] = (lapsed["ca_2024"] + lapsed["ca_2025"]).round(3)
    lapsed["months_hist"] = lapsed["months_2024"] + lapsed["months_2025"]
    lapsed["typical_monthly_ca"] = (lapsed["ca_hist"] / lapsed["months_hist"]).round(3)
    # score = historical value x regularity, normalised by the months actually
    # observed across the two history years (was a hardcoded 10 = 5 months x 2).
    obs_2024, obs_2025 = months_observed(df, 2024), months_observed(df, 2025)
    denom = (obs_2024 + obs_2025) or 1
    lapsed["score"] = (lapsed["ca_hist"] * lapsed["months_hist"] / denom).round(1)
    # Kept short and concrete: a vendor reads this while dialling.
    # "sur 5" matters: months_20XX counts DISTINCT MONTHS WITH A PURCHASE inside the only
    # window the extract contains (January-May). Without the denominator, "acheté 1 mois
    # en 2024" reads as "bought for one month", which is not what it means.
    lapsed["reason"] = (
        "Commandé " + lapsed["months_2024"].astype(str) + f" mois sur {obs_2024} en 2024 et "
        + lapsed["months_2025"].astype(str) + f" sur {obs_2025} en 2025 ("
        + lapsed["ca_hist"].round(0).astype(int).astype(str)
        + " TND au total). Aucune commande en 2026."
    )

    lapsed = (
        lapsed.merge(vendors, on="vendor_code", how="left")
        .merge(clients, on="client_id", how="left")
        .merge(articles, on="article_id", how="left")
        .sort_values("score", ascending=False)
    )
    # clients whose name carries a "***name*** MM-YY" marker look closed/blocked
    # -> kept in the full list with a flag, excluded from the vendor review file
    lapsed["client_flagged"] = flagged(lapsed)

    cols = ["vendor_code", "vendor_name", "client_id", "client_name", "client_flagged",
            "article_id", "enseigne", "designation",
            "months_2024", "months_2025", "ca_2024", "ca_2025",
            "last_purchase", "typical_monthly_ca", "score", "reason"]
    lapsed = lapsed[cols]
    lapsed.to_csv(DATA / "lapsed_reorders.csv", index=False, encoding="utf-8-sig")

    top = (
        lapsed[~lapsed["client_flagged"]]
        .groupby("vendor_code", group_keys=False).head(TOP_PER_VENDOR)
    )
    top.to_csv(DATA / "lapsed_top30_per_vendor.csv", index=False, encoding="utf-8-sig")
    return lapsed


def client_reactivation(df, vendors, clients):
    """Clients inactive in 2026 or in strong decline vs 2025 (Jan-Apr basis)."""
    net = df.copy()
    janavr = net[net["mois"] <= 4]

    c = (
        janavr.groupby(["client_id", "year"])["ca"].sum().unstack("year", fill_value=0.0)
        .rename(columns={2024: "ca_2024_ja", 2025: "ca_2025_ja", 2026: "ca_2026_ja"})
        .reset_index()
    )
    # active months in 2026 incl. May (any positive month)
    act26 = (
        net[(net["year"] == 2026) & (net["ca"] > 0)]
        .groupby("client_id")["mois"].nunique().rename("active_months_2026")
    )
    may = (
        net[net["mois"] == 5]
        .groupby(["client_id", "year"])["ca"].sum().unstack("year", fill_value=0.0)
        .rename(columns={2024: "ca_mai_2024", 2025: "ca_mai_2025", 2026: "ca_mai_2026"})
        .reset_index()
    )
    c = c.merge(act26, on="client_id", how="left").merge(may, on="client_id", how="left")
    c["active_months_2026"] = c["active_months_2026"].fillna(0).astype(int)
    c = c.fillna(0.0)

    c = c[c["ca_2025_ja"] >= MIN_CLIENT_CA_2025].copy()
    c["drop_pct"] = ((c["ca_2026_ja"] - c["ca_2025_ja"]) / c["ca_2025_ja"]).round(3)
    c["status"] = np.where(c["ca_2026_ja"] <= 0, "INACTIVE_2026",
                  np.where(c["drop_pct"] <= DECLINE_THRESHOLD, "DECLINING", "OK"))
    c = c[c["status"] != "OK"].copy()

    # main vendor 2025 = vendor with largest positive CA for the client in 2025
    v25 = (
        df[(df["year"] == 2025) & (df["ca"] > 0)]
        .groupby(["client_id", "vendor_code"])["ca"].sum().reset_index()
        .sort_values("ca", ascending=False)
        .drop_duplicates("client_id")[["client_id", "vendor_code"]]
    )
    c = (
        c.merge(v25, on="client_id", how="left")
        .merge(vendors, on="vendor_code", how="left")
        .merge(clients, on="client_id", how="left")
        .sort_values("ca_2025_ja", ascending=False)
    )
    c["client_flagged"] = flagged(c)
    cols = ["vendor_code", "vendor_name", "client_id", "client_name", "client_flagged",
            "status", "ca_2024_ja", "ca_2025_ja", "ca_2026_ja", "drop_pct",
            "active_months_2026", "ca_mai_2025", "ca_mai_2026"]
    c = c[cols].round(3)
    c.to_csv(DATA / "client_reactivation.csv", index=False, encoding="utf-8-sig")
    return c


def cross_sell(df, articles):
    """Article pairs frequently bought in the same client-month basket (lift)."""
    pos = df[df["ca"] > 0]

    # restrict to the most-bought articles (by distinct clients)
    top_arts = (
        pos.groupby("article_id")["client_id"].nunique()
        .nlargest(TOP_ARTICLES_XSELL).index
    )
    p = pos[pos["article_id"].isin(top_arts)].copy()

    basket_key = p["client_id"].astype(str) + "|" + p["month"].astype(str)
    b_codes, _ = pd.factorize(basket_key)
    a_codes, a_index = pd.factorize(p["article_id"])
    n_b = b_codes.max() + 1
    n_a = a_codes.max() + 1

    m = csr_matrix((np.ones(len(p), dtype=np.int32), (b_codes, a_codes)), shape=(n_b, n_a))
    m.data[:] = 1  # dedupe within basket
    co = (m.T @ m).tocoo()

    counts = np.asarray(m.sum(axis=0)).ravel()  # baskets per article
    mask = (co.row != co.col) & (co.data >= MIN_BASKETS_TOGETHER)
    r, cc, n_together = co.row[mask], co.col[mask], co.data[mask]
    lift = (n_together.astype(float) * n_b) / (counts[r] * counts[cc])

    pairs = pd.DataFrame({
        "article_id": a_index[r],
        "partner_article_id": a_index[cc],
        "baskets_together": n_together,
        "lift": lift.round(2),
    })
    pairs = pairs[pairs["lift"] > 1.0]
    pairs = (
        pairs.sort_values(["article_id", "lift"], ascending=[True, False])
        .groupby("article_id", group_keys=False).head(TOP_PARTNERS)
    )
    art_names = articles.set_index("article_id")[["enseigne", "designation"]]
    pairs = (
        pairs.join(art_names, on="article_id")
        .join(art_names.add_prefix("partner_"), on="partner_article_id")
    )
    pairs.to_csv(DATA / "cross_sell_pairs.csv", index=False, encoding="utf-8-sig")
    return pairs, n_b


def main():
    # Inside main(), never at module level: this module is imported by tests.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    df, vendors, clients, articles = load()

    lapsed = lapsed_reorders(df, vendors, clients, articles)
    print("=== LAPSED REORDERS ===")
    print(f"pairs: {len(lapsed):,}  | vendors covered: {lapsed.vendor_code.nunique()}")
    print(f"historical CA at stake (2024+2025): {lapsed.ca_2024.sum() + lapsed.ca_2025.sum():,.0f} TND")
    print(lapsed.head(5)[["vendor_code", "client_name", "designation", "score", "reason"]]
          .to_string(index=False))

    react = client_reactivation(df, vendors, clients)
    print("\n=== CLIENT REACTIVATION ===")
    print(react.status.value_counts().to_string())
    print(f"2025 Jan-Avr CA of flagged clients: {react.ca_2025_ja.sum():,.0f} TND")
    print(react.head(5)[["vendor_name", "client_name", "status", "ca_2025_ja", "ca_2026_ja", "drop_pct"]]
          .to_string(index=False))

    pairs, n_b = cross_sell(df, articles)
    print("\n=== CROSS-SELL PAIRS ===")
    print(f"baskets (client-month): {n_b:,}  | pairs kept: {len(pairs):,} "
          f"| articles with partners: {pairs.article_id.nunique():,}")
    print(pairs.nlargest(5, "baskets_together")
          [["designation", "partner_designation", "baskets_together", "lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
