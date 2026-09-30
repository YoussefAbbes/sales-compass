"""Pydantic response contracts exposed in OpenAPI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Derived from the serving record, never pinned. A forecast whose window has closed
# is a retrospective record, and an absent record is unknown rather than expired —
# the same three-valued reading /health applies to the window itself.
EvaluationMode = Literal["live_forecast", "expired_forecast", "unknown"]


class RepurchaseWindow(BaseModel):
    """The window the served repurchase scores answer for.

    Exposed rather than implied: the scores are a forecast for a fixed period, and
    once that period closes they are a historical record with no warning attached.
    `null` means the serving record is absent — unknown, not open.
    """
    snapshot_date: str
    window_end: str
    expired: bool
    days_remaining: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    data_ready: bool
    repurchase_ready: bool
    prospect_model_ready: bool
    mode: Literal["real", "fixtures"]
    repurchase_window: RepurchaseWindow | None = None


class ModelStatus(BaseModel):
    name: str
    ready: bool
    version: str | None
    behavior: str
    score_is_calibrated: bool


class MetaResponse(BaseModel):
    api_version: str
    mode: Literal["real", "fixtures"]
    models: list[ModelStatus]
    notices: list[str]
    repurchase_window: RepurchaseWindow | None = None


class VendorItem(BaseModel):
    vendor_code: str
    vendor_name: str
    queued_clients: int
    tnd_at_stake: float


class VendorListResponse(BaseModel):
    items: list[VendorItem]


class ArticleItem(BaseModel):
    article_id: str
    brand: str | None
    designation: str
    label: str


class ArticleSearchResponse(BaseModel):
    query: str
    items: list[ArticleItem]


class RepurchaseOpportunity(BaseModel):
    rank: int
    article_id: str
    brand: str | None
    designation: str
    repurchase_score_raw: float = Field(
        ge=0.0, le=1.0,
        description="Raw CatBoost score. It is intentionally not presented as calibrated.",
    )
    previous_ca_tnd: float
    expected_value_tnd: float
    reason: str


class ClientRepurchaseResult(BaseModel):
    client_rank: int
    client_id: str
    client_name: str
    status: str
    priority_score: float
    opportunities: list[RepurchaseOpportunity]


class RepurchaseResponse(BaseModel):
    model: Literal["repurchase"] = "repurchase"
    model_version: str | None
    generated_at: str | None
    data_window: str
    evaluation_mode: EvaluationMode
    score_is_calibrated: Literal[False] = False
    vendor_code: str
    clients: list[ClientRepurchaseResult]
    notices: list[str]


class RepurchaseSignals(BaseModel):
    """The observable facts behind one client's rank, in the model's own terms.

    Not ranking inputs -- the order is still expected value. These are the evidence
    a vendor can check against their own memory of the account, which is what makes
    a recommendation actionable rather than merely authoritative.
    """
    n_orders: int = Field(description="Times this client bought this article.")
    days_since_last: int = Field(description="Days since their last purchase of it.")
    median_interval_days: float | None = Field(
        default=None,
        description="Their usual gap between purchases. Null with a single purchase: "
                    "one order defines no cadence, and that is distinct from zero.",
    )
    overdue_ratio: float | None = Field(
        default=None,
        description="days_since_last / median_interval_days. Above 1 the client has "
                    "passed their own rhythm. Null when there is no rhythm to pass.",
    )


class ArticleRepurchaseItem(BaseModel):
    rank: int
    client_id: str
    client_name: str
    repurchase_score_raw: float = Field(
        ge=0.0, le=1.0,
        description="Raw CatBoost score. It is intentionally not presented as calibrated.",
    )
    previous_ca_tnd: float
    expected_value_tnd: float
    signals: RepurchaseSignals
    reason: str = Field(
        description="Why this client ranks, phrased from `signals`. Composed server-side "
                    "so every consumer reads the same explanation.",
    )


class ArticleRepurchaseSection(BaseModel):
    model: Literal["repurchase"] = "repurchase"
    model_version: str | None
    data_window: str
    evaluation_mode: EvaluationMode
    score_is_calibrated: Literal[False] = False
    items: list[ArticleRepurchaseItem]
    notices: list[str]


class ProspectSignals(BaseModel):
    similar_buyers_out_of_20: int
    signature_articles_owned: int
    signature_articles_total: int
    same_brand_articles_owned: int
    same_family_articles_owned: int


class ProspectItem(BaseModel):
    rank: int
    client_id: str
    client_name: str
    client_ca_tnd: float
    signals: ProspectSignals
    reason: str


class ProspectSection(BaseModel):
    model: Literal["article_prospect"] = "article_prospect"
    model_version: str | None
    source: Literal["model", "heuristic"]
    data_window: str
    probability_available: Literal[False] = False
    current_window_buyers: int
    minimum_buyers_required: int
    eligible_candidates: int
    empty_reason: Literal[
        "insufficient_current_buyers",
        "no_signature_articles",
        "no_eligible_clients",
        "no_basket_overlap",
    ] | None = None
    items: list[ProspectItem]
    notices: list[str]


class ProspectResponse(ProspectSection):
    vendor_code: str
    article: ArticleItem


class ArticleRecommendationsResponse(BaseModel):
    vendor_code: str
    article: ArticleItem
    repurchases: ArticleRepurchaseSection
    prospects: ProspectSection


# ---------------------------------------------------------------------------
# Dashboard contracts
# ---------------------------------------------------------------------------


class MonthlyCA(BaseModel):
    month: str
    ca_tnd: float


class StatusBreakdown(BaseModel):
    status: str
    count: int
    tnd_at_stake: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class BrandSummary(BaseModel):
    brand: str
    rank: int
    ca_tnd: float
    share_of_scope_ca: float
    clients: int
    articles: int
    purchase_months: int
    ca_2025_tnd: float
    ca_2026_tnd: float
    evolution_pct: float | None


class BrandRankItem(BaseModel):
    rank: int
    code: str
    name: str
    ca_tnd: float
    share_of_brand_ca: float
    clients: int | None = None
    articles: int | None = None


class DashboardModelSummary(BaseModel):
    name: Literal["repurchase", "article_prospect"]
    headline_metric: str
    headline_value: float | None
    comparison_label: str
    comparison_value: float | None
    auc: float | None
    explanation: str
    top_features: list[FeatureImportanceItem]


class DashboardClientSummary(BaseModel):
    rank: int
    client_id: str
    client_name: str
    status: str
    priority_score: float
    expected_value_tnd: float
    lost_revenue_tnd: float
    key_amount_tnd: float
    amount_explanation: str
    last_activity: str | None
    summary: str
    top_article: str | None
    why_recommended: list[str]


class DashboardOverviewResponse(BaseModel):
    data_window: str
    generated_from: str
    total_ca_tnd: float
    fact_rows: int
    observed_months: int
    catalog_articles: int
    known_clients: int
    vendors: int
    queued_assignments: int
    tnd_at_stake: float
    monthly_ca: list[MonthlyCA]
    status_breakdown: list[StatusBreakdown]
    vendor_comparison: list[VendorItem]
    known_brand_ca_share: float
    top_brands: list[BrandSummary]
    top_clients: list[DashboardClientSummary]
    models: list[DashboardModelSummary]
    notices: list[str]


class VendorDashboardResponse(BaseModel):
    vendor_code: str
    vendor_name: str
    queued_clients: int
    tnd_at_stake: float
    expected_value_tnd: float
    lost_revenue_tnd: float
    monthly_ca: list[MonthlyCA]
    status_breakdown: list[StatusBreakdown]
    known_brand_ca_share: float
    brand_concentration_top5: float
    brands: list[BrandSummary]
    clients: list[DashboardClientSummary]
    notices: list[str]


class ClientDashboardResponse(BaseModel):
    vendor_code: str
    vendor_name: str
    client_id: str
    client_name: str
    rank: int
    status: str
    priority_score: float
    expected_value_tnd: float
    lost_revenue_tnd: float
    key_amount_tnd: float
    amount_explanation: str
    last_activity: str | None
    summary: str
    why_recommended: list[str]
    monthly_ca: list[MonthlyCA]
    yearly_ca_tnd: dict[str, float]
    known_brand_ca_share: float
    brand_concentration_top3: float
    brands: list[BrandSummary]
    opportunities: list[RepurchaseOpportunity]
    cross_sell_suggestion: str | None
    notices: list[str]


class BrandListResponse(BaseModel):
    total_brands: int
    known_brand_ca_share: float
    items: list[BrandSummary]
    notices: list[str]


class BrandDashboardResponse(BaseModel):
    brand: str
    ca_tnd: float
    share_of_known_brand_ca: float
    clients: int
    vendors: int
    articles: int
    purchase_months: int
    monthly_ca: list[MonthlyCA]
    yearly_ca_tnd: dict[str, float]
    top_vendors: list[BrandRankItem]
    top_clients: list[BrandRankItem]
    top_articles: list[BrandRankItem]
    notices: list[str]


FeedbackAction = Literal[
    "CONTACTED", "ORDER", "QUOTE", "SNOOZE", "UNREACHABLE", "CLOSED", "DISMISS",
]


class FeedbackEventIn(BaseModel):
    """What the browser sends.

    `event_id` is minted by the client so a retry after a dropped response is
    recognised as the same event rather than becoming a duplicate.

    `vendor_code` is CLAIMED, not verified: a single shared API key cannot identify the
    caller. Stated rather than implied, so nobody reads this as authenticated.
    """
    event_id: str = Field(min_length=1, max_length=64)
    vendor_code: str = Field(min_length=1, max_length=32)
    client_id: str = Field(min_length=1, max_length=32)
    action: FeedbackAction
    note: str = Field(default="", max_length=2000)


class FeedbackEvent(FeedbackEventIn):
    """What was stored. Server-filled fields are added here, never accepted from
    the browser."""
    recorded_at: str
    score_snapshot: str | None = None


class FeedbackEventList(BaseModel):
    vendor_code: str
    items: list[FeedbackEvent]
