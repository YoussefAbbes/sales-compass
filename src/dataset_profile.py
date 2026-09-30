# -*- coding: utf-8 -*-
"""Profiling of the monthly CA extract — what the dataset actually contains.

Pure pandas, no Streamlit and no FastAPI, so it is unit-testable on synthetic frames.

The numbers this produces are the honest limits of the project: a 7-month annual
blind spot, a large naming gap, extreme sparsity. They belong in the interface, not
only in a markdown file nobody opens during a defense.
"""
import pandas as pd


def profile(fact: pd.DataFrame, articles: pd.DataFrame,
            clients: pd.DataFrame, vendors: pd.DataFrame) -> dict:
    negatives = fact["ca"] < 0
    # Group on the month Period and format the handful of results, rather than
    # formatting every row before grouping.
    monthly = (fact.groupby(fact["month"].dt.to_period("M"))
                   .agg(rows=("ca", "size"), ca=("ca", "sum"))
                   .reset_index().sort_values("month"))

    pair_months = fact.groupby(["client_id", "article_id"])["month"].nunique()
    n_clients = int(fact["client_id"].nunique())
    n_articles = int(fact["article_id"].nunique())
    grid = n_clients * n_articles

    art_ca = fact.groupby("article_id")["ca"].sum().sort_values(ascending=False)
    cli_ca = fact.groupby("client_id")["ca"].sum().sort_values(ascending=False)

    def _named(tx_ids: set, ref_ids: set) -> dict:
        named = len(tx_ids & ref_ids)
        return {"total": len(tx_ids), "named": named, "unnamed": len(tx_ids) - named}

    def _top_share(s: pd.Series, frac: float = 0.1) -> float:
        if s.empty or s.sum() == 0:
            return 0.0
        return float(s.head(max(1, int(len(s) * frac))).sum() / s.sum())

    def _ids(df: pd.DataFrame, col: str) -> set:
        return set(df[col]) if col in df.columns else set()

    return {
        "coverage": [{"month": str(m), "rows": int(r), "ca": float(c)}
                     for m, r, c in zip(monthly["month"], monthly["rows"], monthly["ca"])],
        "entities": {
            "rows": int(len(fact)),
            "vendors": int(fact["vendor_code"].nunique()),
            "clients": n_clients,
            "articles": n_articles,
            "months": int(fact["month"].nunique()),
            "total_ca": float(fact["ca"].sum()),
        },
        "naming": {
            "articles": _named(set(fact["article_id"]), _ids(articles, "article_id")),
            "clients": _named(set(fact["client_id"]), _ids(clients, "client_id")),
            "vendors": _named(set(fact["vendor_code"]), _ids(vendors, "vendor_code")),
        },
        "sparsity": {
            "n_pairs": int(len(pair_months)),
            "dense_grid": int(grid),
            "fill_rate": float(len(pair_months) / grid) if grid else 0.0,
            "single_month_pairs": int((pair_months == 1).sum()),
        },
        "concentration": {
            "top10pct_articles_share": _top_share(art_ca),
            "top10pct_clients_share": _top_share(cli_ca),
        },
        "returns": {
            "n_rows": int(negatives.sum()),
            "share": float(negatives.mean()) if len(fact) else 0.0,
            "total_ca": float(fact.loc[negatives, "ca"].sum()),
        },
    }
