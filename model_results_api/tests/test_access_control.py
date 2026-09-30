# -*- coding: utf-8 -*-
"""Serving real data must never be the thing that happens by default.

The original check read `if settings.api_key and ...`: the key was enforced only
when one had been configured. An unset environment variable is the default state
of every environment, so the open configuration was the one reached by doing
nothing at all, and nothing said so — no error, no log, no failed request.

Two guards, both derived from configuration rather than remembered by a person:

  startup  serving the real data/ with no key is refused outright, unless an
           operator explicitly asks for anonymous access. Fixtures are exempt:
           they are anonymised by construction, so there is nothing to protect.

  schema   /docs, /redoc and /openapi.json answer without a key. They leak no
           client data, but they publish the whole surface, so they close once a
           key means the API is reachable by someone other than the developer.
"""
import pytest

from model_results_api.app.config import (
    Settings,
    docs_urls,
    guard_anonymous_access,
)


def _settings(root_name: str, api_key=None, allow_anonymous=False) -> Settings:
    return Settings(
        source_root=__import__("pathlib").Path(root_name).resolve(),
        api_key=api_key,
        cors_origins=(),
        allow_anonymous=allow_anonymous,
    )


# --- the startup guard ----------------------------------------------------------

def test_fixtures_serve_anonymously_because_there_is_nothing_to_protect():
    """The frontend-developer workflow must not need a secret to run."""
    guard_anonymous_access(_settings("fixtures"))


def test_real_data_with_a_key_configured_is_allowed():
    guard_anonymous_access(_settings("project_root", api_key="a-long-secret"))


def test_real_data_without_a_key_is_refused():
    """The regression this whole module exists for: no key used to mean no check."""
    with pytest.raises(RuntimeError):
        guard_anonymous_access(_settings("project_root"))


def test_refusal_names_both_ways_out_so_an_operator_is_not_left_guessing():
    """A guard that blocks without saying how to proceed gets disabled, not fixed."""
    with pytest.raises(RuntimeError) as excinfo:
        guard_anonymous_access(_settings("project_root"))

    message = str(excinfo.value)
    assert "MODEL_RESULTS_API_KEY" in message
    assert "MODEL_RESULTS_ALLOW_ANONYMOUS" in message


def test_anonymous_access_is_possible_but_only_when_asked_for_explicitly():
    """Local work against real data stays possible. It just stops being the
    default: someone has to type it."""
    guard_anonymous_access(_settings("project_root", allow_anonymous=True))


# --- the schema endpoints -------------------------------------------------------

def test_schema_endpoints_stay_open_when_no_key_is_configured():
    """Local development keeps Swagger."""
    urls = docs_urls(_settings("fixtures"))

    assert urls["docs_url"] == "/docs"
    assert urls["openapi_url"] == "/openapi.json"


def test_schema_endpoints_close_once_a_key_marks_the_api_as_shared():
    """A key means someone other than the developer can reach this."""
    urls = docs_urls(_settings("project_root", api_key="a-long-secret"))

    assert urls["docs_url"] is None
    assert urls["redoc_url"] is None
    assert urls["openapi_url"] is None
