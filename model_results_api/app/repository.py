"""Cached access to model artifacts and generated result files."""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import settings

# The bundled engine modules each read DATA_ROOT once, at their own import time. The
# override is scoped to just this import (save, set, import, restore) rather than left
# on os.environ permanently — a lasting mutation here previously leaked into the
# unrelated src/ originals of the same modules when both were imported in one process
# (e.g. the full pytest suite), silently swapping their data source too.
_previous_data_root = os.environ.get("DATA_ROOT")
os.environ["DATA_ROOT"] = str(settings.source_root)
try:
    from ..engine import article_prospects as article_engine
    from ..engine import prospect_features, prospect_model
finally:
    if _previous_data_root is None:
        os.environ.pop("DATA_ROOT", None)
    else:
        os.environ["DATA_ROOT"] = _previous_data_root


def _csv(name: str, **kwargs) -> pd.DataFrame:
    path = settings.data_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


@lru_cache(maxsize=1)
def fact() -> pd.DataFrame:
    path = settings.data_dir / "ca_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


@lru_cache(maxsize=1)
def clients() -> pd.DataFrame:
    return _csv("clients.csv", dtype=str)


@lru_cache(maxsize=1)
def articles() -> pd.DataFrame:
    return _csv("articles.csv", dtype=str)


@lru_cache(maxsize=1)
def branded_fact() -> pd.DataFrame:
    """Positive purchase facts enriched with the best available brand/designation.

    Brand analytics describe what clients actually work with, so returns/credits do
    not create brand relationships. Unknown references stay measurable through the
    coverage ratio but are not presented as a commercial brand.
    """
    refs = (
        articles()[["article_id", "enseigne", "designation"]]
        .drop_duplicates("article_id", keep="first")
        .rename(columns={"enseigne": "brand"})
    )
    frame = fact()[fact()["ca"] > 0].copy()
    frame["article_id"] = frame["article_id"].astype(str)
    frame = frame.merge(refs, on="article_id", how="left")
    frame["brand"] = frame["brand"].fillna("").astype(str).str.strip()
    frame.loc[frame["brand"] == "", "brand"] = "INCONNU"
    frame["designation"] = frame["designation"].fillna("Désignation non disponible")
    return frame


@lru_cache(maxsize=1)
def vendors() -> pd.DataFrame:
    return _csv("vendors.csv", dtype=str)


@lru_cache(maxsize=1)
def queue() -> pd.DataFrame:
    return _csv("clients_to_contact.csv",
                dtype={"vendor_code": str, "client_id": str})


@lru_cache(maxsize=1)
def repurchase_scores() -> pd.DataFrame:
    """The v4 LIVE scores — the window that has not happened yet.

    Not pair_scores_rolling.csv. That file is the backtest: its newest snapshot is
    2026-04-01 and its horizon closed 2026-06-30, inside the ledger this API ships
    with, so serving it answered a question the data already answers. Worse, the
    delivery queue is built on the live file, so the rolling scores were overriding
    a fresher number with a staler one — see contact_queue.attach_model_scores.

    One row per pair already, because a live snapshot is scored at a single T. The
    rolling file needed a sort-and-deduplicate to collapse four snapshots per pair;
    doing that here would be dead code hiding the fact that the shapes differ.

    `label` is the backtest's ground truth and the live file has none by
    construction (see tests/test_score_live.py). The drop stays regardless: it is
    the one guard a later change of source cannot silently defeat.
    """
    frame = _csv("pair_scores_live.csv",
                 dtype={"vendor_code": str, "client_id": str, "article_id": str})
    frame = frame.drop(columns="label", errors="ignore").reset_index(drop=True)
    return frame.rename(columns={"p": "p_model", "amount_sum": "ca_sum"})


@lru_cache(maxsize=1)
def live_window() -> dict:
    """The window the live scores answer for, as recorded by score_live.py.

    Empty when the record is absent, and deliberately NOT reconstructed from the
    ledger: the ledger's last month describes the DATA, while this must describe
    the SCORES. They agree only while both come from the same run, and the moment
    a stale score file is shipped against a fresh ledger a derived answer would be
    confidently wrong. Unknown is the honest result.
    """
    path = settings.reports_dir / "live_window.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def model_metrics() -> dict:
    path = settings.reports_dir / "rolling_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def prospect_metrics() -> dict:
    path = settings.reports_dir / "prospect_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prospect_period() -> str:
    """The window the adoption model is served on, as "YYYY-H1" / "YYYY-H2".

    A complete half-year, because that is how prospect_model.cbm was trained. The
    calendar year used previously happens to agree while the ledger stops in June,
    and stops agreeing the moment it does not — at which point every client-level
    count the model reads (basket size, articles held, similarity) would be built
    over twice the span it learned on.

    Deliberately uncached: it is a max over an already-cached frame, and a cached
    period would outlive the ledger it was derived from.
    """
    return prospect_features.latest_complete_half(fact())


@lru_cache(maxsize=1)
def prospect_context():
    return prospect_features.build_context(fact(), articles(), prospect_period())


@lru_cache(maxsize=1)
def loaded_prospect_model():
    return prospect_model.load_model(settings.data_dir / "prospect_model.cbm")


def artifact_version(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"sha256:{digest}"


def repurchase_version() -> str | None:
    metrics = model_metrics()
    return metrics.get("generated_at") or artifact_version(settings.data_dir / "model_rolling.cbm")


def prospect_version() -> str | None:
    return artifact_version(settings.data_dir / "prospect_model.cbm")


def required_data_ready() -> bool:
    required = [
        "ca_monthly.parquet", "clients.csv", "articles.csv", "vendors.csv",
        "clients_to_contact.csv",
    ]
    return all((settings.data_dir / name).exists() for name in required)


def repurchase_ready() -> bool:
    return all((settings.data_dir / name).exists()
               for name in ("clients_to_contact.csv", "pair_scores_live.csv"))


def prospect_model_ready() -> bool:
    return (settings.data_dir / "prospect_model.cbm").exists()
