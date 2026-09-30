"""Ledger-wide aggregates are derived once per process, not once per request.

`/dashboard/overview` reports totals over the whole ledger: distinct clients,
articles and months, the CA sum, the monthly series, and the brand ranking. None
of them depend on the request, and none can change while the process runs — the
artifacts behind them are themselves read once and cached in `repository`.

They were nevertheless recomputed on every call, which is most of the ~2.7 s the
endpoint still cost after the per-row formatting defect was fixed. These tests pin
the cache boundary: the cached wrappers must agree with a direct computation, and
a second request must not repeat the work.
"""
import pandas as pd

from model_results_api.app import repository as repo, services


def test_cached_brand_summaries_match_a_direct_computation():
    """The cache must not change the answer, only how often it is computed."""
    services._all_brand_summaries.cache_clear()

    cached = services._all_brand_summaries()
    direct = services._brand_summaries(repo.branded_fact(), limit=None)

    assert [item.brand for item in cached] == [item.brand for item in direct]
    assert [item.ca_tnd for item in cached] == [item.ca_tnd for item in direct]
    assert [item.evolution_pct for item in cached] == [item.evolution_pct for item in direct]


def test_cached_ledger_totals_match_a_direct_computation():
    services._ledger_totals.cache_clear()
    facts = repo.fact()

    totals = services._ledger_totals()

    assert totals.rows == len(facts)
    assert totals.months == facts["month"].nunique()
    assert totals.articles == facts["article_id"].nunique()
    assert totals.clients == facts["client_id"].nunique()
    assert totals.vendors == facts["vendor_code"].nunique()
    assert totals.ca == pd.Series(facts["ca"]).sum()


def test_a_second_overview_reuses_the_cached_aggregates():
    services._ledger_totals.cache_clear()
    services._all_brand_summaries.cache_clear()
    services._global_monthly.cache_clear()

    services.dashboard_overview()
    services.dashboard_overview()

    for cached in (services._ledger_totals,
                   services._all_brand_summaries,
                   services._global_monthly):
        info = cached.cache_info()
        assert info.misses == 1, f"{cached.__name__} was computed {info.misses} times"
        assert info.hits >= 1, (
            f"{cached.__name__} did not serve the second request from cache; the "
            "ledger-wide aggregate is being rebuilt per request"
        )


def test_the_overview_brand_table_is_the_head_of_the_full_ranking():
    """Slicing a cached full ranking must equal asking for a short one directly.

    `_brand_summaries` ranks every brand and only then truncates, so rank, share
    and evolution do not depend on the limit. That is what makes one cached
    ranking reusable by both the overview and the brand list.
    """
    full = services._all_brand_summaries()
    short = services._brand_summaries(repo.branded_fact(), limit=3)

    assert [item.brand for item in full[:3]] == [item.brand for item in short]
    assert [item.rank for item in short] == [1, 2, 3]
    assert [item.share_of_scope_ca for item in full[:3]] == [
        item.share_of_scope_ca for item in short
    ]


def test_the_brand_total_counts_every_brand_not_one_page():
    """`total_brands` used to be `len(_brand_summaries(frame, limit=1000))`.

    That is a page length, not a total: it silently stops counting at 1000. The
    fixture catalogue is far smaller, so this guards the intent rather than a
    current miscount.
    """
    response = services.brand_list(limit=2)

    assert len(response.items) <= 2
    assert response.total_brands == len(services._all_brand_summaries())
    assert response.total_brands >= len(response.items)
