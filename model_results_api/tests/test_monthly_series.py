"""The monthly CA series must be built by reducing first, then formatting.

`_monthly` used to label every row before grouping — `frame["month"].dt.strftime()`
runs a Python-level strftime per row, so the 1.75M-row ledger paid 1.75M format
calls to produce 30 distinct labels. It cost ~15 s of the ~17 s that
`/dashboard/overview` spent on every single request, cache warm.

The obvious repair — grouping on the raw datetime — is faster still and passes on
`ca_monthly.parquet`, whose timestamps are already month-start. It is wrong anyway:
the NAV ledger is day-level, and any day-level frame would then emit one entry per
*day* while still labelling each of them "YYYY-MM". The first test below is the
guard for that, and it is the reason the implementation groups on a monthly Period.
"""
import time

import pandas as pd
import pytest

from model_results_api.app import services


def test_a_day_level_frame_collapses_to_one_point_per_month():
    """Three March days and one April day are two months, not four."""
    frame = pd.DataFrame({
        "month": pd.to_datetime(["2025-03-04", "2025-03-19", "2025-03-28", "2025-04-02"]),
        "ca": [10.0, 5.0, 2.5, 7.0],
    })

    series = services._monthly(frame)

    assert [point.month for point in series] == ["2025-03", "2025-04"], (
        "day-level timestamps must be reduced to one point per month; repeated "
        "labels mean the grouping key is the timestamp rather than the month"
    )
    assert series[0].ca_tnd == pytest.approx(17.5)
    assert series[1].ca_tnd == pytest.approx(7.0)


def test_the_series_is_chronological_across_a_year_boundary():
    frame = pd.DataFrame({
        "month": pd.to_datetime(["2025-01-01", "2024-12-01", "2024-02-01"]),
        "ca": [3.0, 2.0, 1.0],
    })

    assert [point.month for point in services._monthly(frame)] == [
        "2024-02", "2024-12", "2025-01",
    ]


def test_an_empty_frame_yields_no_points():
    empty = pd.DataFrame({"month": pd.to_datetime([]), "ca": []})

    assert services._monthly(empty) == []


def test_the_series_scales_to_a_ledger_sized_frame():
    """Cost must follow the number of months, not the number of rows.

    400K rows over 12 months: formatting per row took ~3 s here, reducing first
    takes ~60 ms. The 1 s budget is a ~15x margin over the fixed implementation
    and a ~3x margin under the defect, so it flags a regression without turning
    into a flaky timing assertion on a loaded machine.
    """
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    frame = pd.DataFrame({
        "month": months.repeat(34_000),
        "ca": 1.0,
    })

    started = time.perf_counter()
    series = services._monthly(frame)
    elapsed = time.perf_counter() - started

    assert len(series) == 12
    assert elapsed < 1.0, (
        f"building 12 monthly points from {len(frame):,} rows took {elapsed:.2f}s; "
        "this is the per-row formatting defect returning"
    )
