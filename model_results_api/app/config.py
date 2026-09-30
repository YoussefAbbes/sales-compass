"""Runtime configuration for the model-results API.

Paths are configurable so the same API can serve the owner's real local data or the
committed anonymised fixtures without changing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = API_ROOT.parent


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _workspace_path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    source_root: Path
    api_key: str | None
    cors_origins: tuple[str, ...]
    api_prefix: str = "/api/v1"
    # Anonymous access to the REAL data is possible, but only when an operator asks
    # for it in so many words. See guard_anonymous_access below for why.
    allow_anonymous: bool = False

    @property
    def data_dir(self) -> Path:
        return self.source_root / "data"

    @property
    def reports_dir(self) -> Path:
        return self.source_root / "reports"

    @property
    def fixture_mode(self) -> bool:
        return self.source_root.name.lower() == "fixtures"


def load_settings() -> Settings:
    # MODEL_RESULTS_SOURCE_ROOT takes precedence. DATA_ROOT is accepted for compatibility
    # with the original API. Relative paths are resolved against the workspace root.
    source = os.environ.get("MODEL_RESULTS_SOURCE_ROOT") or os.environ.get("DATA_ROOT")
    source_root = _workspace_path(source, WORKSPACE_ROOT)
    origins = tuple(
        item.strip()
        for item in os.environ.get(
            "MODEL_RESULTS_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if item.strip()
    )
    return Settings(
        source_root=source_root,
        api_key=os.environ.get("MODEL_RESULTS_API_KEY") or None,
        cors_origins=origins,
        allow_anonymous=_flag(os.environ.get("MODEL_RESULTS_ALLOW_ANONYMOUS")),
    )


def guard_anonymous_access(settings: Settings) -> None:
    """Refuse to serve the real data with no key unless asked to, in so many words.

    The check this replaces was `if settings.api_key and ...`: the key was verified
    only when one had been configured, so an unset variable did not fail a request,
    it removed the question. Every environment starts with its variables unset, which
    made "open to anyone who can reach it" the state reached by doing nothing — over
    real client names and real revenue.

    Inverting it costs a developer one variable and buys a deployment that cannot be
    opened by forgetting. Fixtures are exempt because they are anonymised by
    construction and tests/test_no_data_leak.py enforces it: there is nothing there
    for a key to protect.

    Raised at import so a misconfiguration stops the process instead of waiting to be
    discovered on the first request that should have been rejected.
    """
    if settings.fixture_mode or settings.api_key or settings.allow_anonymous:
        return
    raise RuntimeError(
        """Refus de démarrer : l'API servirait data/ (noms de clients et chiffre
d'affaires réels) sans authentification.
  - pour exiger une clé   : MODEL_RESULTS_API_KEY=<secret long>
  - pour un accès anonyme : MODEL_RESULTS_ALLOW_ANONYMOUS=1
                            (à réserver à une machine locale liée à 127.0.0.1)
  - pour un jeu anonymisé : MODEL_RESULTS_SOURCE_ROOT=fixtures"""
    )


def docs_urls(settings: Settings) -> dict[str, str | None]:
    """Where /docs, /redoc and the schema live — or None once they should not.

    They carry no client data, so they are left open for local work. They do publish
    every route, field and parameter, and they answer without a key because they are
    mounted on the app rather than behind the router's dependency. A configured key
    is the signal that someone other than the developer can reach this process, so
    the surface closes with it.
    """
    if settings.api_key:
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


settings = load_settings()
