# -*- coding: utf-8 -*-
"""Committed directories must never contain confidential data.

data/ holds real individuals' names and their revenue, which is why it is gitignored.
Two directories that ARE committed can leak it:

  web/       the frontend. The API is the only permitted path between data/ and the
             browser, so any client name or article designation appearing here is a
             leak — whether through a static export, a hardcoded fixture, or (as
             happened once) a docstring example written with a real name.

  fixtures/  the anonymised dataset handed to a frontend developer
             (tools/make_fixtures.py). Its entire purpose is to carry the *shape* of
             data/ and none of its contents. A single unmapped identifier or name
             defeats it, and unlike web/ it is built from the real files by a script —
             so a regression in that script is exactly what this guards against.

  model_results_api/  the standalone REST handoff. It may contain contracts and engine
             code, but never copied client names or article designations.

Build output is scanned too. node_modules is excluded: third-party, gitignored, and
scanning it would be slow and meaningless.
"""
import re

import pandas as pd
import pytest

# Short strings produce false positives (a two-word company name can appear inside
# ordinary prose). Only test values long enough to be unambiguous.
MIN_NAME = 9
MIN_DESIGNATION = 13
# Tunisian landline/mobile numbers are 8 digits — long enough to be unambiguous.
MIN_PHONE = 8
# Tax identifiers look like "0569199MNC000".
MIN_TAX_ID = 10
# Street names are shared between clients; require a longer match to avoid noise.
MIN_ADDRESS = 12

# Directories that are committed and must stay clean.
SCANNED_TREES = ["web", "fixtures", "model_results_api"]


def _tree_text(project_root, tree: str) -> dict:
    root = project_root / tree
    if not root.exists():
        pytest.skip(f"{tree}/ not present")
    out = {}
    for path in root.rglob("*"):
        if not path.is_file() or "node_modules" in path.parts:
            continue
        try:
            # errors="ignore" keeps binary members (parquet) readable as text: a name
            # stored inside a column still appears as a plain substring.
            out[path] = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return out


def _first_hit(values, texts, min_len):
    for value in values:
        if not isinstance(value, str) or len(value) < min_len:
            continue
        for path, text in texts.items():
            if value in text:
                return value, path
    return None, None


def _first_token_hit(values, texts, min_len):
    """Like _first_hit, but the value must appear as a standalone token.

    Phone numbers and tax ids are short alphanumeric runs, and the committed
    fixtures contain high-precision floats. An 8-digit needle has 1e8 possible
    values while pair_scores_2026.csv alone offers ~300k 8-digit windows across
    its p_model and manual_score columns, so plain substring matching produces
    a predictable ~14 false positives against clean data — and did: '12345678'
    occurs inside the probability 0.6712345678857319.

    Requiring the match not to be flanked by alphanumerics keeps the guarantee
    intact, because a leaked identifier is a field value delimited by a comma,
    quote, colon or newline. The accepted trade-off is that a value welded into
    a longer alphanumeric run (tel12345678) would be missed; that is not a
    shape any export or fixture-builder produces.
    """
    for value in values:
        if not isinstance(value, str) or len(value) < min_len:
            continue
        pattern = None
        for path, text in texts.items():
            # `in` is CPython's C substring search and runs at GB/s; the
            # lookaround regex runs in the far slower regex engine. A boundary
            # match is necessarily a substring match, so this prefilter is
            # exactly equivalent and keeps the regex off the hot path. Without
            # it the scan is ~12k values x 9.4 MB of regex and takes >10 min.
            if value not in text:
                continue
            if pattern is None:
                pattern = re.compile(
                    rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])")
            if pattern.search(text):
                return value, path
    return None, None


def test_first_token_hit_ignores_float_digits_but_catches_delimited_values():
    """Proves the trade-off _first_token_hit makes, on synthetic text only.

    A phone number's digits buried inside an unrelated float's decimal
    expansion (exactly the '12345678' inside '0.6712345678857319' shape that
    caused test_no_master_pii_appears to false-positive) must NOT be flagged.
    The same digits appearing as an actual delimited field value — the shape a
    real leak takes in a CSV, a quoted JSON string, free text, or a bare line —
    MUST still be flagged. If this collapsed back to plain substring matching,
    the first assertion below would fail.
    """
    phone = "12345678"

    hit, _ = _first_token_hit([phone], {"scores.csv": "0.6712345678857319"}, 8)
    assert hit is None, "a needle inside a float's digits must not be a hit"

    leak_shapes = {
        "csv row": "a,12345678,b",
        "quoted json": '"12345678"',
        "labelled text": "tel: 12345678",
        "bare line": "12345678",
    }
    for shape, text in leak_shapes.items():
        hit, path = _first_token_hit([phone], {"f": text}, 8)
        assert hit == phone, f"expected a hit for the {shape} shape ({text!r})"
        assert path == "f"


@pytest.mark.parametrize("tree", SCANNED_TREES)
def test_no_client_name_appears(project_root, data_dir, tree):
    names = pd.read_csv(data_dir / "clients.csv", dtype=str,
                        encoding="utf-8-sig")["client_name"].dropna().tolist()
    hit, path = _first_hit(names, _tree_text(project_root, tree), MIN_NAME)
    assert hit is None, (
        f"client name {hit!r} found in {path} — confidential data must never live "
        f"under {tree}/. Serve it from the API, or re-run tools/make_fixtures.py."
    )


@pytest.mark.parametrize("tree", SCANNED_TREES)
def test_no_article_designation_appears(project_root, data_dir, tree):
    designations = pd.read_csv(data_dir / "articles.csv", dtype=str,
                               encoding="utf-8-sig")["designation"].dropna().tolist()
    hit, path = _first_hit(designations, _tree_text(project_root, tree), MIN_DESIGNATION)
    assert hit is None, (
        f"article designation {hit!r} found in {path} — serve it from the API, "
        f"or re-run tools/make_fixtures.py."
    )


def test_fixture_identifiers_are_all_remapped(project_root, data_dir):
    """Names are not the only identity: raw client ids would re-link the two sets.

    Every fixture id follows the synthetic scheme (C001 / V01 / A0001), so any real id
    surviving the mapping shows up as a value that does not match it.
    """
    fixtures = project_root / "fixtures" / "data"
    if not (fixtures / "clients_to_contact.csv").exists():
        pytest.skip("fixtures/ not built — run py tools/make_fixtures.py")

    queue = pd.read_csv(fixtures / "clients_to_contact.csv",
                        dtype={"vendor_code": str, "client_id": str},
                        encoding="utf-8-sig")
    bad_clients = [c for c in queue["client_id"].dropna().unique()
                   if not (c.startswith("C") and c[1:].isdigit())]
    bad_vendors = [v for v in queue["vendor_code"].dropna().unique()
                   if not (v.startswith("V") and v[1:].isdigit())]
    assert not bad_clients, f"real client ids leaked into fixtures: {bad_clients[:5]}"
    assert not bad_vendors, f"real vendor codes leaked into fixtures: {bad_vendors[:5]}"


@pytest.mark.parametrize("tree", SCANNED_TREES)
def test_no_master_pii_appears(project_root, data_dir, tree):
    """The NAV client master adds phone numbers, tax ids and addresses.

    The original extract carried only names and revenue. These columns are a new and
    larger identity surface, and fixtures/ is built from them by a script — so the
    guarantee has to grow with the data, not after a leak.
    """
    master = data_dir / "clients_master.csv"
    if not master.exists():
        pytest.skip("data/clients_master.csv missing — run py src/build_masters.py")

    df = pd.read_csv(master, dtype=str, encoding="utf-8-sig")
    texts = _tree_text(project_root, tree)

    # Identifier columns match on token boundaries, free-text columns on
    # substring. See _first_token_hit for why.
    for column, min_len, finder in (
        ("phone_1", MIN_PHONE, _first_token_hit),
        ("phone_2", MIN_PHONE, _first_token_hit),
        ("phone_3", MIN_PHONE, _first_token_hit),
        ("tax_id", MIN_TAX_ID, _first_token_hit),
        ("address", MIN_ADDRESS, _first_hit),
        ("contact", MIN_NAME, _first_hit),
    ):
        if column not in df.columns:
            continue
        values = df[column].dropna().unique().tolist()
        hit, path = finder(values, texts, min_len)
        assert hit is None, (
            f"{column} value {hit!r} found in {path} — the NAV master carries personal "
            f"contact data. Serve it from the API, or re-run tools/make_fixtures.py."
        )
