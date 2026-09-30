"""Read-only owner-side smoke test for both real model endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from model_results_api.app.main import app
from model_results_api.app import repository as repo


def main() -> None:
    client = TestClient(app)
    health_response = client.get("/api/v1/health")
    health_response.raise_for_status()
    health = health_response.json()
    assert health["mode"] == "real", health
    assert health["repurchase_ready"], health
    assert health["prospect_model_ready"], health

    vendors = client.get("/api/v1/vendors").json()["items"]
    assert vendors
    repurchase_vendor = vendors[0]["vendor_code"]
    repurchase_response = client.get(
        f"/api/v1/vendors/{repurchase_vendor}/repurchases",
        params={"limit_clients": 3, "per_client": 2},
    )
    repurchase_response.raise_for_status()
    repurchases = repurchase_response.json()
    assert repurchases["clients"], repurchases

    # Select a high-coverage article from the owner's data at runtime. No real identifier
    # is hardcoded into the handoff folder.
    positive = repo.fact()[repo.fact()["ca"] > 0]
    article_id = str(positive.groupby("article_id")["client_id"].nunique().idxmax())
    article_vendor = None
    combined_body = None
    for vendor in vendors:
        response = client.get(
            f"/api/v1/vendors/{vendor['vendor_code']}/articles/{article_id}/recommendations",
            params={"limit": 5},
        )
        body = response.json()
        if (response.status_code == 200
                and body["repurchases"]["items"]
                and body["prospects"]["items"]):
            article_vendor = vendor["vendor_code"]
            combined_body = body
            break

    assert combined_body is not None, "No real combined response with both sections found"
    assert combined_body["repurchases"]["score_is_calibrated"] is False, combined_body
    assert combined_body["prospects"]["source"] == "model", combined_body
    assert combined_body["prospects"]["probability_available"] is False, combined_body

    print({
        "health": health,
        "repurchase_vendor": repurchase_vendor,
        "repurchase_clients": len(repurchases["clients"]),
        "article_vendor": article_vendor,
        "article_repurchase_items": len(combined_body["repurchases"]["items"]),
        "article_prospect_items": len(combined_body["prospects"]["items"]),
        "prospect_source": combined_body["prospects"]["source"],
    })


if __name__ == "__main__":
    main()
