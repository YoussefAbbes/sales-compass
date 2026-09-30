# -*- coding: utf-8 -*-
"""Drafts the opening line of the phone call.

Every clause is built from a field the queue actually contains. When a piece of
information is missing the clause is dropped rather than guessed: a vendor repeating
this on the phone must never say something the data does not support.

Mirrors web/src/callScript.ts so both interfaces phrase things identically.
"""
import re

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

# Words carrying no identity: honorifics, legal forms, articles.
_NOISE = {"MR", "MME", "M", "STE", "STÉ", "SOCIETE", "SOCIÉTÉ", "ETS", "ETABLISSEMENT",
          "SARL", "SUARL", "SA", "SNC", "SPA", "EURL",
          "LA", "LE", "LES", "DE", "DU", "DES", "ET", "D"}


def mois_fr(value) -> str | None:
    """'2026-03' -> 'mars 2026'. None when unparseable."""
    if not value or not isinstance(value, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})", value.strip())
    if not m:
        return None
    idx = int(m.group(2)) - 1
    return f"{MOIS[idx]} {m.group(1)}" if 0 <= idx < 12 else None


def initiales(name, fallback: str = "") -> str:
    """Initials for the avatar disc, tolerant of quotes and legal forms."""
    words = [re.sub(r"[^^\w]", "", w, flags=re.UNICODE)
             for w in re.split(r"[\s,.'’\"«»()\-/]+", str(name or ""))]
    words = [w for w in words if w and not w.isdigit()]
    meaningful = [w for w in words if w.upper() not in _NOISE] or words
    if not meaningful:
        return (fallback or "?")[:2].upper()
    if len(meaningful) == 1:
        return meaningful[0][:2].upper()
    return (meaningful[0][0] + meaningful[-1][0]).upper()


# contact_queue.py fills missing names with this literal. Saying "Bonjour (nom non
# disponible)" on a call would be absurd, so it is treated as no name at all.
_PLACEHOLDERS = {"(nom non disponible)", "nan", "none", ""}


def nom_utilisable(name) -> str | None:
    """The client's name, or None when there isn't a real one to say."""
    if name is None or not isinstance(name, str):
        return None
    cleaned = name.strip()
    return None if cleaned.lower() in _PLACEHOLDERS else cleaned


def _piece(label) -> str | None:
    """Keep the designation half of 'CODE ENSEIGNE — DESIGNATION'."""
    if not label or not isinstance(label, str):
        return None
    parts = label.split(" — ")
    text = (parts[1] if len(parts) > 1 else parts[0]).strip()
    if not text or text.startswith("("):
        text = parts[0].strip()
    return (text[:46].rstrip() + "…") if len(text) > 46 else text


def phrase_appel(client_name, status: str, last_activity=None, top_article=None) -> str:
    """The sentence a vendor can read out loud when the client picks up."""
    nom = nom_utilisable(client_name)
    who = f"Bonjour {nom}." if nom else "Bonjour."
    depuis = mois_fr(last_activity)
    piece = _piece(top_article)
    # Designations are uppercase trade references, not common nouns: quoting them
    # avoids broken French like "du ANTIGEL 1.5L ROUGE".
    ref = f"« {piece} »" if piece else None

    if status == "INACTIVE_2026":
        if ref and depuis:
            return f"{who} Vous preniez régulièrement {ref} — et plus rien depuis {depuis}. Je vous en remets de côté ?"
        if ref:
            return f"{who} Vous preniez régulièrement {ref}, et on ne vous a plus vu cette année. Je vous en remets de côté ?"
        if depuis:
            return f"{who} On n'a plus rien reçu de votre part depuis {depuis}. Qu'est-ce qui vous manque en ce moment ?"
        return f"{who} On ne vous a plus vu cette année. Qu'est-ce qui vous manque en ce moment ?"

    if status == "DECLINING":
        if ref:
            return f"{who} Vos commandes ont ralenti cette année. Il vous faut {ref} ?"
        return f"{who} Vos commandes ont ralenti cette année. Qu'est-ce qui vous manque en ce moment ?"

    if ref:
        return f"{who} Vous reprenez régulièrement {ref} — je vous prépare la prochaine commande ?"
    return f"{who} Je fais le point sur vos besoins du moment — qu'est-ce qu'il vous faut ?"
