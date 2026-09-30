"""The vendored engine must never drift from the src/ originals.

This is a regression test for a bug that had already happened twice. `src/` gained
half-year windows and a dormant-buyer block; the vendored copies kept the old
calendar-year code, and nothing anywhere noticed — the API kept returning numbers,
they were just built by a feature pipeline the served model had never seen.

Drift in vendored *model* code is silent by construction, so it needs a loud test.
"""
import pytest

from model_results_api.tools.sync_engine import ENGINE_DIR, MODULES, expected


@pytest.mark.parametrize("module", MODULES)
def test_vendored_engine_matches_src(module):
    committed = (ENGINE_DIR / f"{module}.py").read_text(encoding="utf-8")
    assert committed == expected()[module], (
        f"model_results_api/engine/{module}.py has drifted from src/{module}.py. "
        "Run: py model_results_api/tools/sync_engine.py"
    )
