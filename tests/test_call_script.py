# -*- coding: utf-8 -*-
"""Unit tests for src/call_script.py.

The sentence produced here is read out loud to a real client, so it must never assert
something the data does not contain, and must stay grammatical with the uppercase trade
designations used in this catalogue.
"""
import pytest

from call_script import initiales, mois_fr, phrase_appel


def test_mois_fr_formats_a_month():
    assert mois_fr("2026-03") == "mars 2026"
    assert mois_fr("2025-12") == "décembre 2025"


@pytest.mark.parametrize("value", [None, "", "pas une date", "2026", 42])
def test_mois_fr_returns_none_on_anything_unparseable(value):
    assert mois_fr(value) is None


def test_initiales_skips_punctuation_and_legal_forms():
    assert initiales('LES PIECES DU NORD "SARL"') == "PN"
    assert initiales("Mr EXEMPLE NOMFICTIF") == "EN"
    assert initiales("ETS UNTEL & FILS") == "UF"


def test_initiales_falls_back_on_the_id_when_the_name_is_unusable():
    assert initiales(None, "0508") == "05"
    assert initiales("***", "0049") == "00"


def test_inactive_client_mentions_the_part_and_the_month():
    s = phrase_appel("Mr EXEMPLE", "INACTIVE_2026", "2025-05",
                     "01381 FEBI — ANTIGEL 1.5L ROUGE")
    assert "Mr EXEMPLE" in s
    assert "ANTIGEL 1.5L ROUGE" in s
    assert "mai 2025" in s


def test_missing_information_drops_the_clause_instead_of_guessing():
    s = phrase_appel("Mr EXEMPLE", "INACTIVE_2026", None, None)
    assert "None" not in s and "nan" not in s
    # No month and no part -> no invented specifics, just an opening question.
    assert "depuis" not in s
    assert s.endswith("?")


def test_designations_are_quoted_never_governed_by_an_article():
    """'du ANTIGEL' is broken French: trade references must be quoted instead."""
    s = phrase_appel("Client", "DECLINING", None, "X — ANTIGEL 1.5L ROUGE")
    assert "« ANTIGEL 1.5L ROUGE »" in s
    assert "du ANTIGEL" not in s
    assert "de ANTIGEL" not in s


def test_each_status_produces_its_own_sentence():
    args = ("Client", )
    inactive = phrase_appel(*args, status="INACTIVE_2026", top_article="X — PIECE")
    declining = phrase_appel(*args, status="DECLINING", top_article="X — PIECE")
    active = phrase_appel(*args, status="ACTIVE", top_article="X — PIECE")
    assert len({inactive, declining, active}) == 3
    assert all(s.startswith("Bonjour Client.") for s in (inactive, declining, active))


def test_no_name_still_produces_a_usable_greeting():
    s = phrase_appel(None, "ACTIVE", None, None)
    assert s.startswith("Bonjour.")


def test_placeholder_name_is_treated_as_no_name():
    """contact_queue.py fills missing names with a literal placeholder.

    Saying "Bonjour (nom non disponible)." on a call would be absurd.
    """
    from call_script import nom_utilisable
    assert nom_utilisable("(nom non disponible)") is None
    assert nom_utilisable("  ") is None
    assert nom_utilisable("nan") is None
    assert nom_utilisable("Mr EXEMPLE") == "Mr EXEMPLE"

    s = phrase_appel("(nom non disponible)", "INACTIVE_2026", "2025-02", None)
    assert s.startswith("Bonjour.")
    assert "non disponible" not in s
