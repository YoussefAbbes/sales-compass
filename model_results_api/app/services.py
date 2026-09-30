"""Business-safe transformations from model outputs to public API responses."""
from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import NamedTuple

import pandas as pd
from fastapi import HTTPException

from . import repository as repo
from ..engine.article_prospects import recency_damping
from .feedback_store import JsonlFeedbackStore
from .schemas import (
    ArticleRecommendationsResponse,
    ArticleRepurchaseItem,
    ArticleRepurchaseSection,
    ArticleItem,
    ArticleSearchResponse,
    BrandDashboardResponse,
    BrandListResponse,
    BrandRankItem,
    BrandSummary,
    ClientDashboardResponse,
    ClientRepurchaseResult,
    DashboardClientSummary,
    DashboardModelSummary,
    DashboardOverviewResponse,
    EvaluationMode,
    FeatureImportanceItem,
    FeedbackEvent,
    FeedbackEventIn,
    FeedbackEventList,
    MonthlyCA,
    ProspectItem,
    ProspectResponse,
    ProspectSection,
    ProspectSignals,
    RepurchaseOpportunity,
    RepurchaseResponse,
    RepurchaseSignals,
    RepurchaseWindow,
    StatusBreakdown,
    VendorDashboardResponse,
    VendorItem,
    VendorListResponse,
)


def _ledger_coverage_text() -> str:
    """The ledger's real month span.

    The literal this replaces ("janvier a mai de chaque annee") described the old
    five-month extract and survived two data generations after it stopped being
    true. A caption read off the data cannot go stale on its own.
    """
    months = repo.fact()["month"]
    return f"{months.min():%m/%Y} - {months.max():%m/%Y}"


def _repurchase_window_text() -> str:
    """Describe the window the served scores actually answer for.

    This used to read `splits.test` out of rolling_metrics.json — the TRAINING
    span — and print it beside forward-looking recommendations. A training window
    and a serving window are different facts that happen to share a shape, which is
    why the wrong one went unnoticed. The serving record is the only source that
    describes the file being served.
    """
    record = repo.live_window()
    if not record:
        return "fenêtre de service inconnue"
    return (f"achats attendus entre le {record['snapshot_date']} et le "
            f"{record['window_end']} (horizon {record['horizon_months']} mois)")


def window_status(today: date | None = None) -> RepurchaseWindow | None:
    """Whether the served window is still open, and for how long.

    `today` is injectable so the boundary is testable without waiting for it.
    `window_end` is the last day INSIDE the window, so expiry is strictly after it.
    """
    record = repo.live_window()
    if not record:
        return None
    end = date.fromisoformat(record["window_end"])
    today = today or date.today()
    return RepurchaseWindow(
        snapshot_date=record["snapshot_date"],
        window_end=record["window_end"],
        expired=today > end,
        days_remaining=max((end - today).days, 0),
    )


def evaluation_mode(today: date | None = None) -> EvaluationMode:
    """What the served scores ARE, derived from the window they answer for.

    This was a pinned literal reading "retrospective_demo" — a claim the response
    made about itself that no data fed. When the source moved from the backtest to
    the live scores, `data_window` and the notices followed and this did not, so
    every forecast was published under the caption of a backtest.

    Deriving it closes the drift permanently, and says something a pinned value
    never could: the scores do not change when the window shuts, but what they
    mean does. An absent record is unknown, not expired.
    """
    window = window_status(today)
    if window is None:
        return "unknown"
    return "expired_forecast" if window.expired else "live_forecast"


def _scores_generated_at() -> str:
    """When the served score file was written.

    Prefers the record over the file's mtime: copying a CSV to a server rewrites
    its mtime, which would report the deploy time as the model's vintage.
    """
    recorded = repo.live_window().get("generated_at")
    if recorded:
        return recorded
    path = repo.settings.data_dir / "pair_scores_live.csv"
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _text(value, fallback: str = "") -> str:
    return fallback if value is None or pd.isna(value) else str(value)


def _plain_reason(value: str) -> str:
    return re.sub(r"\*\*", "", value).strip()


def list_vendors() -> VendorListResponse:
    q = repo.queue()
    grouped = (
        q.groupby(["vendor_code", "vendor_name"], dropna=False)
        .agg(queued_clients=("client_id", "size"),
             tnd_at_stake=("priority_score", "sum"))
        .reset_index()
        .sort_values("tnd_at_stake", ascending=False)
    )
    return VendorListResponse(items=[
        VendorItem(
            vendor_code=str(row.vendor_code),
            vendor_name=_text(row.vendor_name, str(row.vendor_code)),
            queued_clients=int(row.queued_clients),
            tnd_at_stake=float(row.tnd_at_stake),
        )
        for row in grouped.itertuples()
    ])


def search_articles(query: str, limit: int) -> ArticleSearchResponse:
    found = repo.article_engine.search_articles(repo.articles(), query, limit=limit)
    return ArticleSearchResponse(query=query, items=[
        ArticleItem(
            article_id=str(row.article_id),
            brand=_text(row.enseigne) or None,
            designation=_text(row.designation, repo.article_engine.UNKNOWN_DESIGNATION),
            label=str(row.label),
        )
        for row in found.itertuples()
    ])


def _known_vendor(vendor_code: str) -> None:
    positive = repo.fact()[repo.fact()["ca"] > 0]
    if vendor_code not in set(positive["vendor_code"].astype(str)):
        raise HTTPException(404, f"vendeur {vendor_code} inconnu")



# clients_to_contact.csv pivots each client's articles into article_txt_1..3. That
# number is the file's shape, not a business rule, and it was the reason `per_client`
# was capped at 3. Anything past it comes from the pair-level score table instead.
_QUEUE_ARTICLE_COLUMNS = 3


def extra_article_order(pool: pd.DataFrame, previous_ca: dict) -> pd.DataFrame:
    """Order a client's remaining scored articles the way block A orders clients.

    Ranked by expected value damped for recency, reported undamped -- an article the
    client took delivery of two days ago is a poor thing to pitch on today's call
    even when it is their biggest line, and `recency_damping` is per (client,
    article), so this is a real per-row signal rather than a restatement of the
    client's own priority_score.
    """
    if pool is None or not len(pool):
        return pool if pool is not None else pd.DataFrame()
    out = pool.copy()
    out["previous_ca"] = [float(previous_ca.get(str(a), 0.0)) for a in out["article_id"]]
    out["expected_value"] = out["p_model"] * out["previous_ca"]
    if {"days_since_last", "median_interval"} <= set(out.columns):
        damp = [recency_damping(d, m) for d, m
                in zip(out["days_since_last"], out["median_interval"])]
    else:
        damp = 1.0
    # No leading underscore: itertuples() renames such columns to positional
    # placeholders (_1, _2), which silently breaks attribute access downstream.
    return out.assign(ranked_by=out["expected_value"] * damp).sort_values(
        "ranked_by", ascending=False)


def _further_opportunities(client: str, pool, article_ref, ca_lookup,
                           seen: set, wanted: int, rank_offset: int) -> list:
    """The articles past the queue's three columns, as ranked opportunities."""
    if pool is None or wanted <= 0 or not len(pool):
        return []
    candidates = pool[~pool["article_id"].astype(str).isin(seen)]
    if not len(candidates):
        return []
    previous_ca = {str(a): float(ca_lookup.get((client, str(a)), 0.0))
                   for a in candidates["article_id"]}
    ordered = extra_article_order(candidates, previous_ca).head(wanted)

    out = []
    for offset, row in enumerate(ordered.itertuples(), start=1):
        article_id = str(row.article_id)
        ref = article_ref.loc[article_id] if article_id in article_ref.index else None
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[0]
        out.append(RepurchaseOpportunity(
            rank=rank_offset + offset,
            article_id=article_id,
            brand=(_text(ref.get("enseigne")) if ref is not None else "") or None,
            designation=(_text(ref.get("designation")) if ref is not None else "")
                        or article_id,
            repurchase_score_raw=float(row.p_model),
            previous_ca_tnd=float(row.previous_ca),
            expected_value_tnd=float(row.expected_value),
            reason=_repurchase_reason(_repurchase_signals(row)),
        ))
    return out


def repurchases(vendor_code: str, client_id: str | None,
                limit_clients: int, per_client: int) -> RepurchaseResponse:
    _known_vendor(vendor_code)
    q = repo.queue()
    qv = q[q["vendor_code"] == vendor_code].sort_values("priority_score", ascending=False)
    if client_id:
        qv = qv[qv["client_id"] == client_id]
        if qv.empty:
            raise HTTPException(404, "client absent de la file de ce vendeur")
    qv = qv.head(limit_clients).copy()
    article_ref = repo.articles().set_index("article_id")
    # The queue is the final, business-filtered delivery artifact available in both real
    # and anonymised modes. pair_scores supplies clean numeric values where available;
    # the queue's proba_N remains the authoritative fallback because contact_queue.py
    # deliberately fills model coverage gaps before producing the delivery file.
    score_lookup = repo.repurchase_scores().set_index(["client_id", "article_id"])
    positive_2025 = repo.fact()[
        (repo.fact()["ca"] > 0) & (repo.fact()["month"].dt.year == 2025)
    ]
    ca_lookup = positive_2025.groupby(["client_id", "article_id"])["ca"].sum()

    # Everything beyond the queue's three columns. Restricted to the clients actually
    # being returned, so asking for one client does not group 518k rows.
    extra_pool: dict[str, pd.DataFrame] = {}
    if per_client > _QUEUE_ARTICLE_COLUMNS:
        pool = repo.repurchase_scores()
        pool = pool[pool["client_id"].astype(str).isin(set(qv["client_id"].astype(str)))]
        if len(pool):
            extra_pool = dict(tuple(pool.groupby(pool["client_id"].astype(str))))

    results: list[ClientRepurchaseResult] = []
    for client_rank, qrow in enumerate(qv.itertuples(), start=1):
        client = str(qrow.client_id)
        opportunities: list[RepurchaseOpportunity] = []
        for rank in range(1, min(per_client, _QUEUE_ARTICLE_COLUMNS) + 1):
            label = getattr(qrow, f"article_txt_{rank}", None)
            if not isinstance(label, str) or not label.strip():
                continue
            article_id = label.strip().split(maxsplit=1)[0]
            ref = article_ref.loc[article_id] if article_id in article_ref.index else None
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            designation = (_text(ref.get("designation")) if ref is not None else "")
            brand = (_text(ref.get("enseigne")) if ref is not None else "")
            if not designation:
                designation = label.split("—", 1)[-1].strip() if "—" in label else label

            raw_percent = getattr(qrow, f"proba_{rank}", None)
            raw_score = (float(raw_percent) / 100.0
                         if raw_percent is not None and not pd.isna(raw_percent) else None)
            key = (client, article_id)
            if key in score_lookup.index:
                scored = score_lookup.loc[key]
                if isinstance(scored, pd.DataFrame):
                    scored = scored.iloc[0]
                raw_score = float(scored["p_model"])
            if raw_score is None:
                continue
            previous_ca = float(ca_lookup.get(key, 0.0))
            opportunities.append(RepurchaseOpportunity(
                # Contiguous, not the CSV column index: a blank article_txt_2 used to
                # leave a hole and the page would print #1 then #3.
                rank=len(opportunities) + 1,
                article_id=article_id,
                brand=brand or None,
                designation=designation,
                repurchase_score_raw=raw_score,
                previous_ca_tnd=previous_ca,
                expected_value_tnd=raw_score * previous_ca,
                reason=_text(getattr(qrow, f"reason_{rank}", None)),
            ))
        opportunities.extend(
            _further_opportunities(client, extra_pool.get(client), article_ref, ca_lookup,
                                   {o.article_id for o in opportunities},
                                   per_client - len(opportunities),
                                   len(opportunities)))
        if not opportunities:
            continue
        results.append(ClientRepurchaseResult(
            client_rank=client_rank,
            client_id=client,
            client_name=_text(qrow.client_name, client),
            status=_text(qrow.status, "ACTIVE"),
            priority_score=float(qrow.priority_score),
            opportunities=opportunities,
        ))

    return RepurchaseResponse(
        model_version=repo.repurchase_version(),
        generated_at=_scores_generated_at(),
        data_window=_repurchase_window_text(),
        evaluation_mode=evaluation_mode(),
        vendor_code=vendor_code,
        clients=results,
        notices=[
            "Prévision : la fenêtre servie n'est pas encore observée, aucun "
            "résultat n'y est mesuré.",
            "Le score brut de rachat n'est pas calibré et peut être optimiste.",
            f"Le registre couvre {_ledger_coverage_text()} en continu ; aucun mois n'est tronqué.",
        ],
    )


def _repurchase_signals(row) -> RepurchaseSignals:
    """Lift one client's observable facts off the scored row.

    median_interval is NaN for a client with a single purchase -- snapshots.py returns
    NaN rather than 0 so that "no cadence yet" stays distinct from "bought twice the
    same day". That distinction has to survive to the page, so it becomes None here
    instead of collapsing to a number the data never supported.
    """
    median = getattr(row, "median_interval", None)
    median = None if median is None or pd.isna(median) else float(median)
    days = int(getattr(row, "days_since_last", 0) or 0)
    return RepurchaseSignals(
        n_orders=int(getattr(row, "n_orders", 0) or 0),
        days_since_last=days,
        median_interval_days=median,
        overdue_ratio=(days / median) if median else None,
    )


def _repurchase_reason(signals: RepurchaseSignals) -> str:
    """Why THIS client ranks, in the terms the model ranked them by.

    The sentence this replaces was identical on every row and restated the formula
    printed directly above it. A reason that is true of every client explains none of
    them, and a vendor cannot check it against what they know about the account.

    The closing clause reads days_since_last against the client's OWN median interval
    rather than a fixed number of days: a garage that orders fortnightly and a garage
    that orders twice a year are not both late at 45 days.
    """
    parts = ["Un seul achat enregistré" if signals.n_orders <= 1
             else f"Achetée {signals.n_orders} fois"]
    parts.append(f"dernier achat il y a {signals.days_since_last} j")
    if signals.median_interval_days:
        parts.append(f"rythme habituel {signals.median_interval_days:.0f} j")
    text = " · ".join(parts)

    ratio = signals.overdue_ratio
    if ratio is None:
        return text
    if ratio >= 1.5:
        # One decimal is real precision at 2x and false precision at 21x, where the
        # elapsed days already carry the message. Comma, not point: the page is French.
        late = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}".replace(".", ",")
        return f"{text} — en retard : {late}× son rythme habituel"
    if ratio >= 1.0:
        return f"{text} — arrive à échéance"
    # Below their own cadence the client is not due, and article_prospects.
    # recency_damping has already moved them down the list for it. "encore dans son
    # cycle" stated that fact as though it supported the ranking: it was rendered
    # under a panel headed "Pourquoi il remonte", so the page offered the strongest
    # reason to demote a client as its justification for promoting them. The
    # sentence now names the demotion the vendor can see in the ranking.
    return f"{text} — servi récemment : priorité réduite"


def article_repurchases(vendor_code: str, article_id: str,
                        limit: int) -> ArticleRepurchaseSection:
    """Return block 1: existing buyers of one article, ranked by money at stake."""
    _known_vendor(vendor_code)
    facts = repo.fact()
    if article_id not in set(facts["article_id"].astype(str)):
        raise HTTPException(404, f"article {article_id} inconnu")

    frame = repo.article_engine.existing_customers(
        repo.repurchase_scores(),
        facts,
        article_id,
        vendor_code,
        top=limit,
    )
    names = dict(zip(repo.clients()["client_id"], repo.clients()["client_name"]))
    items = []
    for rank, row in enumerate(frame.itertuples(), start=1):
        signals = _repurchase_signals(row)
        items.append(ArticleRepurchaseItem(
            rank=rank,
            client_id=str(row.client_id),
            client_name=_text(names.get(row.client_id), str(row.client_id)),
            repurchase_score_raw=float(row.p_model),
            previous_ca_tnd=float(row.ca_sum),
            expected_value_tnd=float(row.expected_value),
            signals=signals,
            reason=_repurchase_reason(signals),
        ))
    return ArticleRepurchaseSection(
        model_version=repo.repurchase_version(),
        data_window=_repurchase_window_text(),
        evaluation_mode=evaluation_mode(),
        items=items,
        notices=[
            "Prévision : la fenêtre servie n'est pas encore observée, aucun "
            "résultat n'y est mesuré.",
            "Le score brut de rachat n'est pas calibré et peut être optimiste.",
            "Classement : valeur attendue = score brut de rachat × CA précédent.",
        ],
    )


def prospects(vendor_code: str, article_id: str, limit: int) -> ProspectResponse:
    _known_vendor(vendor_code)
    facts = repo.fact()
    if article_id not in set(facts["article_id"].astype(str)):
        raise HTTPException(404, f"article {article_id} inconnu")

    article_rows = repo.articles()[repo.articles()["article_id"] == article_id]
    if article_rows.empty:
        article = ArticleItem(
            article_id=article_id,
            brand=None,
            designation=repo.article_engine.UNKNOWN_DESIGNATION,
            label=f"{article_id} — {repo.article_engine.UNKNOWN_DESIGNATION}",
        )
    else:
        row = article_rows.iloc[0]
        brand = _text(row.get("enseigne")) or None
        designation = _text(row.get("designation"), repo.article_engine.UNKNOWN_DESIGNATION)
        article = ArticleItem(
            article_id=article_id,
            brand=brand,
            designation=designation,
            label=f"{article_id} {brand or ''} — {designation}".replace("  ", " ").strip(),
        )

    context = repo.prospect_context()
    all_time_buyers = repo.article_engine.buyers_of(facts, article_id)
    current_buyers = repo.article_engine.buyers_of(context.win, article_id)
    owned = repo.article_engine.vendor_clients(context.win, vendor_code)
    eligible_candidates = [
        client
        for client in repo.prospect_features.candidates_for(context, article_id)
        if client in owned and client not in all_time_buyers
    ]
    loaded_model = repo.loaded_prospect_model()
    frame, signature, source = repo.prospect_model.rank_prospects(
        context,
        article_id,
        vendor_code,
        model=loaded_model,
        top=limit,
        exclude=all_time_buyers,
    )
    # `rank_prospects` historically reports "heuristic" on every early empty return,
    # even when the model artifact is loaded and no ranking was attempted. For an API,
    # `source` describes the configured ranker; `empty_reason` describes why it could not
    # produce rows. Keeping those concepts separate avoids the contradiction
    # "model_version exists" + "artifact absent".
    source = "model" if loaded_model is not None else "heuristic"

    empty_reason = None
    if frame.empty:
        if len(current_buyers) < repo.article_engine.MIN_BUYERS:
            empty_reason = "insufficient_current_buyers"
        elif not signature:
            empty_reason = "no_signature_articles"
        elif not eligible_candidates:
            empty_reason = "no_eligible_clients"
        else:
            empty_reason = "no_basket_overlap"
    names = dict(zip(repo.clients()["client_id"], repo.clients()["client_name"]))
    target_brand = context.article_feats["enseigne"].get(article_id, "")
    target_family = context.fam.get(article_id, "")
    items = []
    for rank, row in enumerate(frame.itertuples(), start=1):
        reason = repo.prospect_model.prospect_reason(
            row, signature, target_brand, target_family)
        items.append(ProspectItem(
            rank=rank,
            client_id=str(row.client_id),
            client_name=_text(names.get(row.client_id), str(row.client_id)),
            client_ca_tnd=float(row.client_ca),
            signals=ProspectSignals(
                similar_buyers_out_of_20=int(row.knn_buyers),
                signature_articles_owned=int(row.n_signature),
                signature_articles_total=len(signature),
                same_brand_articles_owned=int(row.brand_n_held),
                same_family_articles_owned=int(row.family_n_held),
            ),
            reason=_plain_reason(reason),
        ))

    notices = [
        "La liste expose un ordre validé, pas une probabilité d'achat.",
        "Tout client ayant déjà acheté l'article dans l'extrait est exclu des prospects.",
        f"Les signaux utilisent les achats positifs de la fenêtre {repo.prospect_period()}.",
    ]
    if empty_reason == "insufficient_current_buyers":
        notices.append(
            f"Cet article a {len(current_buyers)} acheteur(s) sur la fenêtre "
            f"{repo.prospect_period()} ; il en faut au moins {repo.article_engine.MIN_BUYERS} "
            "pour construire un profil fiable."
        )
    elif empty_reason == "no_signature_articles":
        notices.append("Aucun article-repère ne satisfait les seuils de fréquence et de spécificité.")
    elif empty_reason == "no_eligible_clients":
        notices.append("Ce vendeur n'a aucun client actuel éligible pour cet article.")
    elif empty_reason == "no_basket_overlap":
        notices.append("Des clients sont éligibles, mais leurs paniers ne fournissent pas assez de preuves.")

    if loaded_model is None:
        notices.append("Le modèle de prospection est absent ; le classement par ressemblance est utilisé.")

    return ProspectResponse(
        model_version=repo.prospect_version(),
        source=source,
        data_window=f"fenêtre {repo.prospect_period()} → classement d'adoption sur la fenêtre suivante",
        vendor_code=vendor_code,
        article=article,
        current_window_buyers=len(current_buyers),
        minimum_buyers_required=repo.article_engine.MIN_BUYERS,
        eligible_candidates=len(eligible_candidates),
        empty_reason=empty_reason,
        items=items,
        notices=notices,
    )


def article_recommendations(vendor_code: str, article_id: str,
                            limit: int) -> ArticleRecommendationsResponse:
    """Return both article-centric UI blocks in one stable response."""
    repurchase_section = article_repurchases(vendor_code, article_id, limit)
    prospect_response = prospects(vendor_code, article_id, limit)
    prospect_section = ProspectSection(
        model_version=prospect_response.model_version,
        source=prospect_response.source,
        data_window=prospect_response.data_window,
        current_window_buyers=prospect_response.current_window_buyers,
        minimum_buyers_required=prospect_response.minimum_buyers_required,
        eligible_candidates=prospect_response.eligible_candidates,
        empty_reason=prospect_response.empty_reason,
        items=prospect_response.items,
        notices=prospect_response.notices,
    )
    return ArticleRecommendationsResponse(
        vendor_code=vendor_code,
        article=prospect_response.article,
        repurchases=repurchase_section,
        prospects=prospect_section,
    )


# ---------------------------------------------------------------------------
# Dashboard services
# ---------------------------------------------------------------------------


def _number(row, key: str) -> float:
    value = row.get(key, 0.0)
    return 0.0 if value is None or pd.isna(value) else float(value)


def _status_explanation(status: str) -> str:
    return {
        "INACTIVE_2026": "Ce client achetait auparavant mais aucun achat positif n'est observé en 2026.",
        "DECLINING": "Son chiffre d'affaires observé recule par rapport à la même période précédente.",
        "ACTIVE": "Le client reste actif, mais des articles habituels n'ont pas encore été repris.",
    }.get(status, "Le client apparaît dans la file calculée par les règles commerciales.")


def _why_for_queue_row(row) -> list[str]:
    facts = [_status_explanation(_text(row.get("status"), "ACTIVE"))]
    resume = _text(row.get("resume"))
    if resume:
        facts.append(resume)
    n_lapsed = _number(row, "n_lapsed_total")
    if n_lapsed:
        facts.append(
            f"{int(n_lapsed)} relation(s) client-article historique(s) sont en retard ou absentes."
        )
    last_activity = _text(row.get("last_activity"))
    if last_activity:
        facts.append(f"Dernier mois d'achat observé : {last_activity}.")
    return facts


def _dashboard_client(row, rank: int) -> DashboardClientSummary:
    top_article = _text(row.get("article_txt_1")) or None
    return DashboardClientSummary(
        rank=rank,
        client_id=str(row["client_id"]),
        client_name=_text(row.get("client_name"), str(row["client_id"])),
        status=_text(row.get("status"), "ACTIVE"),
        priority_score=_number(row, "priority_score"),
        expected_value_tnd=_number(row, "valeur_attendue"),
        lost_revenue_tnd=_number(row, "manque_a_gagner"),
        key_amount_tnd=_number(row, "montant_cle"),
        amount_explanation=_text(row.get("montant_texte"), "montant utilisé pour expliquer la priorité"),
        last_activity=_text(row.get("last_activity")) or None,
        summary=_text(row.get("resume"), _status_explanation(_text(row.get("status"), "ACTIVE"))),
        top_article=top_article,
        why_recommended=_why_for_queue_row(row),
    )


def _monthly(frame: pd.DataFrame) -> list[MonthlyCA]:
    """Total CA per month, oldest first.

    Reduce first, then format. Labelling before grouping — `.dt.strftime()` on the
    month column — costs a Python-level format call per row, which on the 1.75M-row
    ledger took ~15 s to produce 30 labels and dominated /dashboard/overview.

    The grouping key is a monthly Period rather than the timestamp itself: the
    ledger behind this frame is day-level, and grouping on the raw timestamp would
    emit one entry per day while still labelling each "YYYY-MM".
    """
    if frame.empty:
        return []
    grouped = frame.groupby(frame["month"].dt.to_period("M"))["ca"].sum().sort_index()
    return [MonthlyCA(month=str(period), ca_tnd=float(ca)) for period, ca in grouped.items()]


def _known_brand_share(frame: pd.DataFrame) -> float:
    total = float(frame["ca"].sum()) if not frame.empty else 0.0
    if total <= 0:
        return 0.0
    known = float(frame.loc[frame["brand"] != "INCONNU", "ca"].sum())
    return known / total


def _year_ca(by_year: pd.DataFrame, brand: str, year: int) -> float:
    if year not in by_year.columns or brand not in by_year.index:
        return 0.0
    return float(by_year.at[brand, year])


def _brand_summaries(frame: pd.DataFrame, limit: int | None = 12) -> list[BrandSummary]:
    """Rank brands in `frame` by CA. `limit=None` ranks all of them.

    Every brand is aggregated regardless of `limit`, which only truncates at the
    end; rank, share and evolution are therefore independent of it. That is what
    lets one full ranking be computed once and sliced by each caller.
    """
    known = frame[frame["brand"] != "INCONNU"].copy()
    if known.empty:
        return []
    known["year"] = known["month"].dt.year
    grouped = (
        known.groupby("brand", as_index=False)
        .agg(
            ca_tnd=("ca", "sum"),
            clients=("client_id", "nunique"),
            articles=("article_id", "nunique"),
            purchase_months=("month", "nunique"),
        )
        .sort_values("ca_tnd", ascending=False)
    )
    # Year-on-year has to compare equal spans. Under the old January-May extract
    # every year held the same five months, so two calendar sums were comparable.
    # The NAV ledger is continuous and its most recent year is partial, so the same
    # sums now weigh a full year against a half one and report a ~50% collapse for
    # every brand. Restrict both sides to the months the latest year actually covers.
    latest_year = int(known["year"].max())
    comparable = set(known.loc[known["year"] == latest_year, "month"].dt.month)
    by_year = (known[known["month"].dt.month.isin(comparable)]
               .pivot_table(index="brand", columns="year", values="ca",
                            aggfunc="sum", fill_value=0))
    denominator = float(grouped["ca_tnd"].sum())
    ranked = grouped if limit is None else grouped.head(limit)
    items: list[BrandSummary] = []
    for rank, row in enumerate(ranked.itertuples(), start=1):
        # A brand can drop out of `by_year` entirely once the comparable months are
        # applied, so the row is checked as well as the column.
        ca_2025 = _year_ca(by_year, row.brand, 2025)
        ca_2026 = _year_ca(by_year, row.brand, 2026)
        evolution = (ca_2026 - ca_2025) / ca_2025 if ca_2025 > 0 else None
        items.append(BrandSummary(
            brand=str(row.brand),
            rank=rank,
            ca_tnd=float(row.ca_tnd),
            share_of_scope_ca=float(row.ca_tnd) / denominator if denominator > 0 else 0.0,
            clients=int(row.clients),
            articles=int(row.articles),
            purchase_months=int(row.purchase_months),
            ca_2025_tnd=ca_2025,
            ca_2026_tnd=ca_2026,
            evolution_pct=evolution,
        ))
    return items


def _brand_concentration(items: list[BrandSummary], top: int) -> float:
    return float(sum(item.share_of_scope_ca for item in items[:top]))


class LedgerTotals(NamedTuple):
    """Ledger-wide scalars. Each `nunique` below is a full pass over ~1.75M rows."""
    ca: float
    rows: int
    months: int
    articles: int
    clients: int
    vendors: int


# Ledger-wide aggregates, derived once per process.
#
# These are process-lifetime caches, which is safe for the same reason
# `repository`'s own lru_cache is: the artifacts are opened once at first use and
# never reread, so a derived value cannot go stale relative to its source. Adding
# artifact reloading to a running process means clearing both layers together.
#
# They hold Pydantic models that are shared between requests. Serialising them is
# read-only, so callers slice and copy but must not mutate the items.

@lru_cache(maxsize=1)
def _ledger_totals() -> LedgerTotals:
    facts = repo.fact()
    return LedgerTotals(
        ca=float(facts["ca"].sum()),
        rows=int(len(facts)),
        months=int(facts["month"].nunique()),
        articles=int(facts["article_id"].nunique()),
        clients=int(facts["client_id"].nunique()),
        vendors=int(facts["vendor_code"].nunique()),
    )


@lru_cache(maxsize=1)
def _global_monthly() -> tuple[MonthlyCA, ...]:
    return tuple(_monthly(repo.fact()))


@lru_cache(maxsize=1)
def _all_brand_summaries() -> tuple[BrandSummary, ...]:
    return tuple(_brand_summaries(repo.branded_fact(), limit=None))


@lru_cache(maxsize=1)
def _global_known_brand_share() -> float:
    return _known_brand_share(repo.branded_fact())


class VendorBrands(NamedTuple):
    summaries: tuple[BrandSummary, ...]
    known_share: float


@lru_cache(maxsize=64)
def _vendor_brands(vendor_code: str) -> VendorBrands:
    """One vendor's brand ranking and known-brand share, from a single scan.

    38 vendors, so the whole set fits the cache. Both values previously rebuilt
    the vendor mask over the full branded ledger on every request — and
    `vendor_dashboard` evaluated `repo.branded_fact()` twice to do it.
    """
    frame = repo.branded_fact()
    scoped = frame[frame["vendor_code"].astype(str) == vendor_code]
    return VendorBrands(
        summaries=tuple(_brand_summaries(scoped, limit=None)),
        known_share=_known_brand_share(scoped),
    )


def _rank_brand_entities(frame: pd.DataFrame, entity: str, limit: int) -> list[BrandRankItem]:
    if entity == "vendor":
        code_col, name_col = "vendor_code", "vendor_name"
        names = dict(zip(repo.vendors()["vendor_code"], repo.vendors()["vendor_name"]))
        grouped = frame.groupby(code_col).agg(
            ca_tnd=("ca", "sum"), clients=("client_id", "nunique"), articles=("article_id", "nunique")
        )
    elif entity == "client":
        code_col, name_col = "client_id", "client_name"
        names = dict(zip(repo.clients()["client_id"], repo.clients()["client_name"]))
        grouped = frame.groupby(code_col).agg(
            ca_tnd=("ca", "sum"), clients=("client_id", "nunique"), articles=("article_id", "nunique")
        )
    else:
        code_col, name_col = "article_id", "designation"
        names = dict(zip(repo.articles()["article_id"], repo.articles()["designation"]))
        grouped = frame.groupby(code_col).agg(
            ca_tnd=("ca", "sum"), clients=("client_id", "nunique"), articles=("article_id", "nunique")
        )
    grouped = grouped.sort_values("ca_tnd", ascending=False).head(limit)
    denominator = float(frame["ca"].sum())
    result = []
    for rank, (code, row) in enumerate(grouped.iterrows(), start=1):
        result.append(BrandRankItem(
            rank=rank,
            code=str(code),
            name=_text(names.get(str(code)), str(code)),
            ca_tnd=float(row["ca_tnd"]),
            share_of_brand_ca=float(row["ca_tnd"]) / denominator if denominator > 0 else 0.0,
            clients=int(row["clients"]) if entity != "client" else None,
            articles=int(row["articles"]) if entity != "article" else None,
        ))
    return result


def _status_breakdown(frame: pd.DataFrame) -> list[StatusBreakdown]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby("status", dropna=False)
        .agg(count=("client_id", "size"), tnd_at_stake=("priority_score", "sum"))
        .reset_index()
        .sort_values("tnd_at_stake", ascending=False)
    )
    return [
        StatusBreakdown(
            status=_text(r.status, "UNKNOWN"),
            count=int(r.count),
            tnd_at_stake=float(r.tnd_at_stake),
        )
        for r in grouped.itertuples()
    ]


def _top_features(metrics: dict, limit: int = 8) -> list[FeatureImportanceItem]:
    values = metrics.get("feature_importance", {})
    ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
    return [FeatureImportanceItem(feature=str(k), importance=float(v)) for k, v in ordered]


def _not_comparable_notice() -> str:
    """Say out loud that the v4 numbers do not line up with the v1 ones.

    The metrics file carries the warning; the dashboard is where someone would
    otherwise put the two AUCs side by side and read a regression that is really a
    change of question.
    """
    note = (repo.model_metrics().get("not_comparable_to") or {}).get("reason")
    base = "Les mesures du modèle de rachat ne se comparent pas à celles du modèle annuel précédent"
    return f"{base} : {note}." if note else f"{base}."


def _model_summaries() -> list[DashboardModelSummary]:
    repurchase_metrics = repo.model_metrics()
    # v4 reports AUC per split as a mapping, and pairs the model against a
    # revenue-ranking baseline rather than the v1 manual rule.
    repurchase_auc = repurchase_metrics.get("auc")
    repurchase_auc = repurchase_auc if isinstance(repurchase_auc, dict) else {}
    prospect_metrics = repo.prospect_metrics()
    rankers = prospect_metrics.get("rankers", {})
    model_ranker = rankers.get("model", {})
    heuristic_ranker = rankers.get("heuristic", {})
    return [
        DashboardModelSummary(
            name="repurchase",
            headline_metric="Précision dans le top 30",
            headline_value=repurchase_metrics.get("precision_top30_model"),
            comparison_label="Classement par chiffre d'affaires",
            comparison_value=repurchase_metrics.get("precision_top30_revenue"),
            auc=repurchase_auc.get("test"),
            explanation=(
                "Classe les articles déjà achetés selon leur propension brute au rachat, "
                "sur une fenêtre glissante. La comparaison retenue est le classement par "
                "chiffre d'affaires, c'est-à-dire appeler d'abord les plus gros montants. "
                "Le score sert à ordonner les pistes et n'est pas une probabilité garantie."
            ),
            top_features=_top_features(repurchase_metrics),
        ),
        DashboardModelSummary(
            name="article_prospect",
            headline_metric="Précision moyenne dans le top 5",
            headline_value=model_ranker.get("precision_at_k"),
            comparison_label="Heuristique panier",
            comparison_value=heuristic_ranker.get("precision_at_k"),
            auc=prospect_metrics.get("auc"),
            explanation=(
                "Classe, pour une pièce, les clients qui ne l'ont jamais achetée à partir "
                "de leur panier, de leur taille et de leurs proximités de comportement."
            ),
            top_features=_top_features(prospect_metrics),
        ),
    ]


def dashboard_overview(limit_clients: int = 8) -> DashboardOverviewResponse:
    queue = repo.queue().sort_values("priority_score", ascending=False).reset_index(drop=True)
    vendors = list_vendors().items
    totals = _ledger_totals()
    # Computed once and reused by the notice below, rather than twice per request.
    data_window = _ledger_coverage_text()
    return DashboardOverviewResponse(
        data_window=data_window,
        generated_from="CA mensuel positif et négatif, agrégé par vendeur × client × article",
        total_ca_tnd=totals.ca,
        fact_rows=totals.rows,
        observed_months=totals.months,
        catalog_articles=totals.articles,
        known_clients=totals.clients,
        vendors=totals.vendors,
        queued_assignments=int(len(queue)),
        tnd_at_stake=float(queue["priority_score"].sum()),
        monthly_ca=list(_global_monthly()),
        status_breakdown=_status_breakdown(queue),
        vendor_comparison=vendors,
        known_brand_ca_share=_global_known_brand_share(),
        top_brands=list(_all_brand_summaries()[:10]),
        top_clients=[
            _dashboard_client(row, rank)
            for rank, (_, row) in enumerate(queue.head(limit_clients).iterrows(), start=1)
        ],
        models=_model_summaries(),
        notices=[
            "Le montant de priorité classe les appels; ce n'est pas une prévision de vente.",
            f"Le registre NAV couvre {data_window} en continu.",
            "Le dernier mois du registre peut être partiel et doit être confirmé par l'entreprise.",
            "Un même client peut appartenir à plusieurs portefeuilles vendeurs.",
            _not_comparable_notice(),
        ],
    )


def vendor_dashboard(vendor_code: str, limit_clients: int = 200) -> VendorDashboardResponse:
    _known_vendor(vendor_code)
    queue = repo.queue()
    scoped = (
        queue[queue["vendor_code"] == vendor_code]
        .sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )
    if scoped.empty:
        raise HTTPException(404, f"aucune file calculée pour le vendeur {vendor_code}")
    vendor_name = _text(scoped.iloc[0].get("vendor_name"), vendor_code)
    facts = repo.fact()
    vendor_facts = facts[facts["vendor_code"].astype(str) == vendor_code]
    vendor_brands = _vendor_brands(vendor_code)
    brand_items = list(vendor_brands.summaries[:15])
    clients = [
        _dashboard_client(row, rank)
        for rank, (_, row) in enumerate(scoped.head(limit_clients).iterrows(), start=1)
    ]
    return VendorDashboardResponse(
        vendor_code=vendor_code,
        vendor_name=vendor_name,
        queued_clients=int(len(scoped)),
        tnd_at_stake=float(scoped["priority_score"].sum()),
        expected_value_tnd=float(scoped["valeur_attendue"].fillna(0).sum()),
        lost_revenue_tnd=float(scoped["manque_a_gagner"].fillna(0).sum()),
        monthly_ca=_monthly(vendor_facts),
        status_breakdown=_status_breakdown(scoped),
        known_brand_ca_share=vendor_brands.known_share,
        brand_concentration_top5=_brand_concentration(brand_items, 5),
        brands=brand_items,
        clients=clients,
        notices=[
            "Les clients sont classés par priority_score, un indice composite réservé à l'ordre de la file.",
            "Les montants explicatifs sont issus soit de l'historique perdu, soit des articles non repris.",
        ],
    )


def client_dashboard(vendor_code: str, client_id: str,
                     per_client: int = _QUEUE_ARTICLE_COLUMNS) -> ClientDashboardResponse:
    _known_vendor(vendor_code)
    queue = repo.queue()
    scoped = queue[queue["vendor_code"] == vendor_code].sort_values(
        "priority_score", ascending=False
    ).reset_index(drop=True)
    matches = scoped[scoped["client_id"].astype(str) == client_id]
    if matches.empty:
        raise HTTPException(404, "client absent de la file de ce vendeur")
    row = matches.iloc[0]
    rank = int(matches.index[0]) + 1
    facts = repo.fact()
    history = facts[
        (facts["vendor_code"].astype(str) == vendor_code)
        & (facts["client_id"].astype(str) == client_id)
    ]
    brand_history = repo.branded_fact()[
        (repo.branded_fact()["vendor_code"].astype(str) == vendor_code)
        & (repo.branded_fact()["client_id"].astype(str) == client_id)
    ]
    brand_items = _brand_summaries(brand_history, limit=10)
    yearly = history.assign(year=history["month"].dt.year).groupby("year")["ca"].sum()
    response = repurchases(vendor_code, client_id, limit_clients=1,
                           per_client=per_client)
    opportunities = response.clients[0].opportunities if response.clients else []
    return ClientDashboardResponse(
        vendor_code=vendor_code,
        vendor_name=_text(row.get("vendor_name"), vendor_code),
        client_id=client_id,
        client_name=_text(row.get("client_name"), client_id),
        rank=rank,
        status=_text(row.get("status"), "ACTIVE"),
        priority_score=_number(row, "priority_score"),
        expected_value_tnd=_number(row, "valeur_attendue"),
        lost_revenue_tnd=_number(row, "manque_a_gagner"),
        key_amount_tnd=_number(row, "montant_cle"),
        amount_explanation=_text(row.get("montant_texte"), "montant de référence"),
        last_activity=_text(row.get("last_activity")) or None,
        summary=_text(row.get("resume"), _status_explanation(_text(row.get("status"), "ACTIVE"))),
        why_recommended=_why_for_queue_row(row),
        monthly_ca=_monthly(history),
        yearly_ca_tnd={str(int(year)): float(value) for year, value in yearly.items()},
        known_brand_ca_share=_known_brand_share(brand_history),
        brand_concentration_top3=_brand_concentration(brand_items, 3),
        brands=brand_items,
        opportunities=opportunities,
        cross_sell_suggestion=_text(row.get("suggestion_complementaire")) or None,
        notices=[
            "Les scores de rachat affichés sont bruts et peuvent être optimistes.",
            f"L'historique mensuel couvre {_ledger_coverage_text()}.",
            "La priorité est une aide à l'ordre d'appel, pas une promesse de revenu.",
        ],
    )


def brand_list(query: str = "", limit: int = 100) -> BrandListResponse:
    # The full ranking, not a 1000-brand page of it: `total_brands` reports the
    # catalogue size, which a page length silently caps once the catalogue grows
    # past it.
    all_items = _all_brand_summaries()
    normalized = query.strip().casefold()
    items = [item for item in all_items if normalized in item.brand.casefold()][:limit]
    return BrandListResponse(
        total_brands=len(all_items),
        known_brand_ca_share=_global_known_brand_share(),
        items=items,
        notices=[
            "Les classements utilisent uniquement les lignes de CA positif.",
            "La part d'une marque est calculée parmi les références dont la marque est connue.",
            "L'évolution compare les deux années sur les mois qu'elles ont en commun, sans extrapolation.",
        ],
    )


def brand_dashboard(brand: str, limit: int = 10) -> BrandDashboardResponse:
    frame = repo.branded_fact()
    available = {value.casefold(): value for value in frame["brand"].unique() if value != "INCONNU"}
    selected = available.get(brand.strip().casefold())
    if selected is None:
        raise HTTPException(404, f"marque {brand} inconnue")
    scoped = frame[frame["brand"] == selected].copy()
    known_total = float(frame.loc[frame["brand"] != "INCONNU", "ca"].sum())
    yearly = scoped.assign(year=scoped["month"].dt.year).groupby("year")["ca"].sum()
    return BrandDashboardResponse(
        brand=selected,
        ca_tnd=float(scoped["ca"].sum()),
        share_of_known_brand_ca=(float(scoped["ca"].sum()) / known_total if known_total > 0 else 0.0),
        clients=int(scoped["client_id"].nunique()),
        vendors=int(scoped["vendor_code"].nunique()),
        articles=int(scoped["article_id"].nunique()),
        purchase_months=int(scoped["month"].nunique()),
        monthly_ca=_monthly(scoped),
        yearly_ca_tnd={str(int(year)): float(value) for year, value in yearly.items()},
        top_vendors=_rank_brand_entities(scoped, "vendor", limit),
        top_clients=_rank_brand_entities(scoped, "client", limit),
        top_articles=_rank_brand_entities(scoped, "article", limit),
        notices=[
            "Le CA de marque additionne les achats positifs ; les retours ne créent pas une relation de marque.",
            "Les top vendeurs et top clients décrivent l'historique observé, pas une exclusivité commerciale.",
            f"Le registre couvre {_ledger_coverage_text()}.",
        ],
    )


# Module-level so tests can swap it for a scratch path, and so a database
# implementation later replaces one line rather than every call site.
feedback_store = JsonlFeedbackStore(repo.settings.data_dir / "feedback_events.jsonl")


def _known_client(client_id: str) -> None:
    if client_id not in set(repo.clients()["client_id"].astype(str)):
        raise HTTPException(404, f"client {client_id} inconnu")


def record_feedback(payload: FeedbackEventIn) -> FeedbackEvent:
    """Store one call outcome.

    `score_snapshot` is stamped here rather than accepted from the browser: the server
    knows the served window authoritatively, and it is the single field the measurement
    half depends on.
    """
    _known_vendor(payload.vendor_code)
    _known_client(payload.client_id)
    event = FeedbackEvent(
        **payload.model_dump(),
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        score_snapshot=repo.live_window().get("snapshot_date"),
    )
    stored = feedback_store.append(event.model_dump())
    return FeedbackEvent(**stored)


def vendor_feedback(vendor_code: str) -> FeedbackEventList:
    _known_vendor(vendor_code)
    return FeedbackEventList(
        vendor_code=vendor_code,
        items=[FeedbackEvent(**e) for e in feedback_store.read_for_vendor(vendor_code)],
    )
