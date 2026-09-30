from fastapi.testclient import TestClient

from model_results_api.app.main import app


client = TestClient(app)


def test_health_reports_fixture_mode():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "fixtures"
    assert body["repurchase_ready"] is True


def test_openapi_has_concrete_response_schemas():
    schema = client.get("/openapi.json").json()
    schemas = schema["components"]["schemas"]
    assert "RepurchaseResponse" in schemas
    assert "ProspectResponse" in schemas
    path = schema["paths"]["/api/v1/vendors/{vendor_code}/repurchases"]["get"]
    ref = path["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/RepurchaseResponse")
    combined_path = schema["paths"][
        "/api/v1/vendors/{vendor_code}/articles/{article_id}/recommendations"
    ]["get"]
    combined_ref = combined_path["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert combined_ref.endswith("/ArticleRecommendationsResponse")
    dashboard_path = schema["paths"]["/api/v1/dashboard/overview"]["get"]
    dashboard_ref = dashboard_path["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert dashboard_ref.endswith("/DashboardOverviewResponse")


def test_vendor_discovery_and_repurchase_contract():
    vendors = client.get("/api/v1/vendors").json()["items"]
    assert vendors
    code = vendors[0]["vendor_code"]
    response = client.get(
        f"/api/v1/vendors/{code}/repurchases",
        params={"limit_clients": 5, "per_client": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "repurchase"
    assert body["score_is_calibrated"] is False
    # Shape only. Which mode is correct depends on the serving window, and is
    # pinned against that record in test_live_window.py rather than frozen here.
    assert body["evaluation_mode"] in {
        "live_forecast", "expired_forecast", "unknown"}
    assert body["vendor_code"] == code
    if body["clients"]:
        assert len(body["clients"][0]["opportunities"]) <= 2
        opportunity = body["clients"][0]["opportunities"][0]
        assert 0 <= opportunity["repurchase_score_raw"] <= 1


def test_article_search_contract():
    response = client.get("/api/v1/articles", params={"query": "PIECE", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "PIECE"
    assert len(body["items"]) <= 5


def test_prospect_contract_never_exposes_a_probability():
    vendors = client.get("/api/v1/vendors").json()["items"]
    code = vendors[0]["vendor_code"]
    articles = client.get("/api/v1/articles", params={"limit": 1}).json()["items"]
    article_id = articles[0]["article_id"]
    response = client.get(
        f"/api/v1/vendors/{code}/articles/{article_id}/prospects",
        params={"limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "article_prospect"
    assert body["probability_available"] is False
    assert "probability" not in str(body["items"]).lower()
    assert len(body["items"]) <= 5
    assert body["current_window_buyers"] >= 0
    assert body["minimum_buyers_required"] == 3
    if not body["items"]:
        assert body["empty_reason"] is not None


def test_combined_article_contract_returns_both_sections():
    vendors = client.get("/api/v1/vendors").json()["items"]
    code = vendors[0]["vendor_code"]
    articles = client.get("/api/v1/articles", params={"limit": 1}).json()["items"]
    article_id = articles[0]["article_id"]
    response = client.get(
        f"/api/v1/vendors/{code}/articles/{article_id}/recommendations",
        params={"limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["vendor_code"] == code
    assert body["article"]["article_id"] == article_id
    assert body["repurchases"]["model"] == "repurchase"
    assert body["repurchases"]["score_is_calibrated"] is False
    assert len(body["repurchases"]["items"]) <= 5
    assert body["prospects"]["model"] == "article_prospect"
    assert body["prospects"]["probability_available"] is False
    assert len(body["prospects"]["items"]) <= 5
    assert "probability" not in str(body["prospects"]["items"]).lower()


def test_unknown_vendor_and_article_are_404():
    assert client.get("/api/v1/vendors/UNKNOWN/repurchases").status_code == 404
    vendors = client.get("/api/v1/vendors").json()["items"]
    code = vendors[0]["vendor_code"]
    response = client.get(f"/api/v1/vendors/{code}/articles/UNKNOWN/prospects")
    assert response.status_code == 404


def test_dashboard_overview_exposes_bounded_explainable_statistics():
    response = client.get("/api/v1/dashboard/overview", params={"limit_clients": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["fact_rows"] > 0
    assert body["total_ca_tnd"] > 0
    assert len(body["top_clients"]) <= 3
    assert len(body["models"]) == 2
    assert body["models"][0]["top_features"]
    assert body["top_brands"]
    assert 0 < body["known_brand_ca_share"] <= 1
    assert body["notices"]


def test_vendor_dashboard_and_client_dossier_explain_the_ranking():
    vendor = client.get("/api/v1/vendors").json()["items"][0]
    code = vendor["vendor_code"]
    response = client.get(
        f"/api/v1/vendors/{code}/dashboard", params={"limit_clients": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["vendor_code"] == code
    assert len(body["clients"]) <= 3
    assert body["status_breakdown"]
    assert body["brands"]
    assert 0 <= body["brand_concentration_top5"] <= 1
    selected = body["clients"][0]
    assert selected["why_recommended"]

    detail = client.get(
        f"/api/v1/vendors/{code}/clients/{selected['client_id']}"
    )
    assert detail.status_code == 200
    dossier = detail.json()
    assert dossier["rank"] >= 1
    assert dossier["why_recommended"]
    assert dossier["monthly_ca"]
    assert dossier["yearly_ca_tnd"]
    assert dossier["brands"]
    assert 0 <= dossier["brand_concentration_top3"] <= 1
    assert dossier["notices"]


def test_brand_list_and_detail_connect_brand_to_vendors_clients_and_articles():
    response = client.get("/api/v1/brands", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) <= 5
    assert body["items"]
    assert body["total_brands"] >= len(body["items"])
    top = body["items"][0]
    assert top["clients"] > 0
    assert top["articles"] > 0
    assert 0 < top["share_of_scope_ca"] <= 1

    detail = client.get("/api/v1/brands/detail", params={"brand": top["brand"], "limit": 4})
    assert detail.status_code == 200
    brand = detail.json()
    assert brand["brand"] == top["brand"]
    assert brand["monthly_ca"]
    assert brand["top_vendors"]
    assert brand["top_clients"]
    assert brand["top_articles"]
    assert len(brand["top_vendors"]) <= 4
    assert client.get("/api/v1/brands/detail", params={"brand": "UNKNOWN"}).status_code == 404
