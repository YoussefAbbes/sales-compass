"""FastAPI application exposing both recommendation models through stable contracts."""
from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import repository as repo
from .config import docs_urls, guard_anonymous_access, settings
from .schemas import (
    ArticleRecommendationsResponse,
    ArticleSearchResponse,
    BrandDashboardResponse,
    BrandListResponse,
    ClientDashboardResponse,
    DashboardOverviewResponse,
    FeedbackEvent,
    FeedbackEventIn,
    FeedbackEventList,
    HealthResponse,
    MetaResponse,
    ModelStatus,
    ProspectResponse,
    RepurchaseResponse,
    VendorDashboardResponse,
    VendorListResponse,
)
from .services import (
    article_recommendations,
    brand_dashboard,
    brand_list,
    client_dashboard,
    dashboard_overview,
    list_vendors,
    prospects,
    record_feedback,
    repurchases,
    search_articles,
    vendor_dashboard,
    vendor_feedback,
    window_status,
)


# Before anything is mounted. A process that would serve real client data with no
# authentication must not reach the point of accepting a connection.
guard_anonymous_access(settings)

app = FastAPI(
    **docs_urls(settings),
    title="Vendor Recommendation Models API",
    version="1.0.0",
    description=(
        "Versioned REST contracts for the repurchase and article-prospect models. "
        "Real client data remains in the backend; use MODEL_RESULTS_SOURCE_ROOT=fixtures "
        "for an anonymised frontend-development dataset."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and (
        x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key)
    ):
        raise HTTPException(401, "clé API absente ou invalide")


router = APIRouter(prefix=settings.api_prefix, dependencies=[Depends(require_api_key)])


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"docs": "/docs", "openapi": "/openapi.json", "health": f"{settings.api_prefix}/health"}


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    data_ready = repo.required_data_ready()
    repurchase_ready = repo.repurchase_ready()
    window = window_status()
    # A closed window is a degraded service: the files load, the routes answer, and
    # every number describes a period that has already finished. An UNKNOWN window
    # is not treated as closed — absence of the record is not evidence of expiry.
    expired = bool(window and window.expired)
    return HealthResponse(
        status="ok" if data_ready and repurchase_ready and not expired else "degraded",
        data_ready=data_ready,
        repurchase_ready=repurchase_ready,
        prospect_model_ready=repo.prospect_model_ready(),
        mode="fixtures" if settings.fixture_mode else "real",
        repurchase_window=window,
    )


@router.get("/meta", response_model=MetaResponse, tags=["system"])
def meta() -> MetaResponse:
    prospect_ready = repo.prospect_model_ready()
    return MetaResponse(
        api_version=app.version,
        mode="fixtures" if settings.fixture_mode else "real",
        models=[
            ModelStatus(
                name="repurchase",
                ready=repo.repurchase_ready(),
                version=repo.repurchase_version(),
                behavior="Ranks previously purchased articles for likely repurchase.",
                score_is_calibrated=False,
            ),
            ModelStatus(
                name="article_prospect",
                ready=prospect_ready,
                version=repo.prospect_version(),
                behavior=("CatBoost adoption ranking" if prospect_ready
                          else "Basket-similarity fallback until the model artifact is available"),
                score_is_calibrated=False,
            ),
        ],
        notices=[
            "Model 1 exposes an explicitly uncalibrated raw score.",
            "Model 2 exposes rank and evidence only; it never exposes a probability.",
        ],
        repurchase_window=window_status(),
    )


@router.get("/vendors", response_model=VendorListResponse, tags=["discovery"])
def vendors() -> VendorListResponse:
    return list_vendors()


@router.get(
    "/dashboard/overview",
    response_model=DashboardOverviewResponse,
    tags=["dashboard"],
)
def dashboard_overview_route(
    limit_clients: int = Query(default=8, ge=1, le=25),
) -> DashboardOverviewResponse:
    return dashboard_overview(limit_clients)


@router.get(
    "/vendors/{vendor_code}/dashboard",
    response_model=VendorDashboardResponse,
    tags=["dashboard"],
)
def vendor_dashboard_route(
    vendor_code: str,
    limit_clients: int = Query(default=200, ge=1, le=500),
) -> VendorDashboardResponse:
    return vendor_dashboard(vendor_code, limit_clients)


@router.get(
    "/vendors/{vendor_code}/clients/{client_id}",
    response_model=ClientDashboardResponse,
    tags=["dashboard"],
)
def client_dashboard_route(
    vendor_code: str,
    client_id: str,
    per_client: int = Query(default=3, ge=1, le=50),
) -> ClientDashboardResponse:
    return client_dashboard(vendor_code, client_id, per_client)


@router.get("/brands", response_model=BrandListResponse, tags=["dashboard"])
def brands(
    query: str = Query(default="", max_length=80),
    limit: int = Query(default=100, ge=1, le=300),
) -> BrandListResponse:
    return brand_list(query, limit)


@router.get("/brands/detail", response_model=BrandDashboardResponse, tags=["dashboard"])
def brand_detail(
    brand: str = Query(min_length=1, max_length=80),
    limit: int = Query(default=10, ge=1, le=50),
) -> BrandDashboardResponse:
    return brand_dashboard(brand, limit)


@router.get("/articles", response_model=ArticleSearchResponse, tags=["discovery"])
def articles(
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=25, ge=1, le=100),
) -> ArticleSearchResponse:
    return search_articles(query, limit)


@router.get(
    "/vendors/{vendor_code}/repurchases",
    response_model=RepurchaseResponse,
    tags=["repurchase model"],
)
def vendor_repurchases(
    vendor_code: str,
    client_id: str | None = Query(default=None),
    limit_clients: int = Query(default=50, ge=1, le=200),
    per_client: int = Query(default=3, ge=1, le=50),
) -> RepurchaseResponse:
    return repurchases(vendor_code, client_id, limit_clients, per_client)


@router.get(
    "/vendors/{vendor_code}/articles/{article_id}/prospects",
    response_model=ProspectResponse,
    tags=["article prospect model"],
)
def article_prospects(
    vendor_code: str,
    article_id: str,
    limit: int = Query(default=5, ge=1, le=200),
) -> ProspectResponse:
    return prospects(vendor_code, article_id, limit)


@router.get(
    "/vendors/{vendor_code}/articles/{article_id}/recommendations",
    response_model=ArticleRecommendationsResponse,
    tags=["combined recommendations"],
)
def combined_article_recommendations(
    vendor_code: str,
    article_id: str,
    limit: int = Query(default=5, ge=1, le=200),
) -> ArticleRecommendationsResponse:
    return article_recommendations(vendor_code, article_id, limit)


@router.post("/feedback", response_model=FeedbackEvent, tags=["feedback"])
def post_feedback(payload: FeedbackEventIn) -> FeedbackEvent:
    """Record one call outcome.

    The first write path in this API. `vendor_code` is stored as claimed: a shared API
    key cannot identify the caller, and pretending otherwise would be worse than saying
    so. Verification lands here when per-vendor keys are decided.
    """
    return record_feedback(payload)


@router.get(
    "/vendors/{vendor_code}/feedback",
    response_model=FeedbackEventList,
    tags=["feedback"],
)
def get_vendor_feedback(vendor_code: str) -> FeedbackEventList:
    """This vendor's events only -- never another vendor's, even on a shared client."""
    return vendor_feedback(vendor_code)


app.include_router(router)
