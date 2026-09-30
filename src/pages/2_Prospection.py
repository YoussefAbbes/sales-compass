# -*- coding: utf-8 -*-
"""Prospection par article — l'inverse de la file d'appels.

Run:  streamlit run src/app.py   (cette page apparaît dans la navigation)

Le vendeur part de la pièce, pas du client : il a une promotion à pousser, un objectif
fournisseur, ou du stock à écouler. Deux blocs, deux signaux, jamais mélangés — le
modèle ne sait noter que les clients ayant déjà acheté l'article.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import article_prospects as ap
import prospect_features as pfeat
import prospect_model as pmod
from ledger import COLUMNS as LEDGER_COLS, monthly_ca
from ui_theme import fmt_tnd, inject_css

st.set_page_config(page_title="Prospection par article", page_icon="🔎", layout="wide")
inject_css()


@st.cache_resource
def load_prospect_model():
    """The fitted block-2 model, or None when training has not been run.

    cache_resource rather than cache_data: a CatBoost model is an unhashable object that must
    not be copied per session. None is a normal outcome — data/ is gitignored, so a fresh
    clone has no artifact and the page falls back to the heuristic.
    """
    return pmod.load_model()


@st.cache_data
def load_context(period: str):
    """The window context for block 3, built once per period and reused.

    Half-year window from the FULL ledger, matching how the adoption model is
    now trained (src/train_prospect_model.py: 2024-H1 -> 2024-H2 and
    2024-H2 -> 2025-H1, embargoed, tested on 2025-H2 -> 2026-H1).

    Both halves of that sentence matter. Serving twelve-month windows to a model
    trained on six would roughly double every count it learned thresholds on;
    serving a PARTIAL half-year would halve them. latest_complete_half() picks
    the newest window the data fully covers, so neither happens.
    """
    return pfeat.build_context(fact, articles, period)


@st.cache_data
def load_all():
    # Full ledger, not ca_monthly.parquet: that file holds January-May only, so
    # buyers_of() missed 400,353 buyer pairs and offered real buyers as prospects.
    fact = monthly_ca(pd.read_parquet(ap.DATA / "transactions.parquet",
                                      columns=LEDGER_COLS))
    articles = pd.read_csv(ap.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")
    clients = pd.read_csv(ap.DATA / "clients.csv", dtype=str, encoding="utf-8-sig")
    vendors = pd.read_csv(ap.DATA / "vendors.csv", dtype=str, encoding="utf-8-sig")

    # v4 live scores first. Renamed HERE rather than inside article_prospects so
    # that module stays independent of which model produced the numbers.
    dt = {"client_id": str, "article_id": str, "vendor_code": str}
    live = ap.DATA / "pair_scores_live.csv"
    v1 = ap.DATA / "pair_scores_2026.csv"
    if live.exists():
        scores = (pd.read_csv(live, dtype=dt, encoding="utf-8-sig")
                    .rename(columns={"p": "p_model", "amount_sum": "ca_sum"}))
    elif v1.exists():
        scores = pd.read_csv(v1, dtype=dt, encoding="utf-8-sig")
    else:
        scores = pd.DataFrame(columns=["client_id", "article_id", "vendor_code",
                                       "ca_sum", "p_model"])
    return fact, articles, clients, vendors, scores


try:
    fact, articles, clients, vendors, scores = load_all()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nLancez d'abord le pipeline (`python src/build_dataset.py`).")
    st.stop()

name_of = dict(zip(clients["client_id"], clients["client_name"]))

st.markdown(
    '<div class="masthead"><div class="eyebrow">Atelier — prospection</div>'
    '<h1>Qui appeler pour cette pièce ?</h1>'
    '<div class="sub">Choisissez un article : voici les clients de votre secteur qui '
    'l\'achetaient et ceux qui ne l\'ont jamais pris.</div></div>',
    unsafe_allow_html=True,
)

# ---------------- Sélection ----------------
# No format_func: st.testing.v1.AppTest exposes Selectbox.options as the *formatted*
# strings, so set_value(options[i]) — the standard way this suite drives a selectbox
# (see tests/test_validation_page.py) — only round-trips when options are raw values.
vendor_codes = sorted(set(fact["vendor_code"]))
labels = dict(zip(vendors["vendor_code"], vendors["vendor_name"]))
vendor_code = st.sidebar.selectbox("Vendeur", vendor_codes)
st.sidebar.caption(labels.get(vendor_code, vendor_code))

query = st.text_input("Chercher une pièce", placeholder="huile atf, plaquette, 21345131…")
found = ap.search_articles(articles, query, limit=25)

if found.empty:
    st.info("Aucune pièce ne correspond. Essayez une référence, une marque, "
            "ou un mot de la désignation.")
    st.stop()

choice = st.selectbox("Article", found["label"].tolist())
article_id = found.loc[found["label"] == choice, "article_id"].iloc[0]

owned = ap.vendor_clients(fact, vendor_code)
buyers = ap.buyers_of(fact, article_id)
c1, c2 = st.columns(2)
c1.metric("Acheteurs, tous vendeurs", f"{len(buyers):,}".replace(",", " "))
c2.metric("Dont vos clients", f"{len(buyers & owned):,}".replace(",", " "))

# ---------------- Bloc 1 ----------------
st.markdown('<div class="section-label">1 · Ils l\'achètent</div>',
            unsafe_allow_html=True)

block1 = ap.existing_customers(scores, fact, article_id, vendor_code)
if block1.empty:
    st.info("Aucun de vos clients n'a acheté cette pièce dans la fenêtre de 12 mois "
            "du modèle. Regardez les blocs 2 et 3.")
else:
    st.caption("Classés par **argent en jeu** — propension au rachat × CA sur 12 mois. "
               "Les pourcentages ne sont **pas calibrés** : lisez-les comme un "
               "classement, pas comme une probabilité réelle.")
    for r in block1.itertuples():
        st.markdown(
            f"**{name_of.get(r.client_id, r.client_id)}** · `{r.client_id}` — "
            f"**{r.p_model * 100:.0f} %** de propension au rachat · "
            f"{fmt_tnd(r.ca_sum)} l'an dernier")

# ---------------- Bloc 2 : anciens acheteurs ----------------
# Half of all buyer pairs sit outside the v4 window. Without this block they leave
# block 1 (no score) and are excluded from block 3 (they ARE buyers) — the vendor
# would silently lose sight of them.
dormant = ap.dormant_buyers(fact, scores, article_id, vendor_code)
if not dormant.empty:
    st.markdown('<div class="section-label">2 · Ils l\'achetaient</div>',
                unsafe_allow_html=True)
    st.caption("Ils ont acheté cette pièce puis se sont arrêtés. **Aucun pourcentage** : "
               "la pièce est sortie de la fenêtre de 12 mois du modèle, qui n'a donc "
               "pas d'avis sur eux. Classés par ce qu'ils y dépensaient.")
    for r in dormant.itertuples():
        st.markdown(
            f"**{name_of.get(r.client_id, r.client_id)}** · `{r.client_id}` — "
            f"{fmt_tnd(r.article_ca)} sur cette pièce · "
            f"dernier achat {r.last_month:%m/%Y}")

# ---------------- Bloc 3 ----------------
st.markdown('<div class="section-label">3 · À prospecter</div>', unsafe_allow_html=True)

# Features come from the most recent complete window; the model was trained to read one
# window and predict the next. `exclude=buyers` keeps the "jamais pris" promise true, since a
# single window cannot see a purchase made in an earlier year.
ctx = load_context(pfeat.latest_complete_half(fact))
block2, signature, source = pmod.rank_prospects(
    ctx, article_id, vendor_code, model=load_prospect_model(), exclude=buyers)
if not signature:
    st.info(f"Trop peu d'acheteurs ({len(buyers)}) pour établir un profil fiable : "
            f"il en faut au moins {ap.MIN_BUYERS}.")
elif block2.empty:
    if not (owned - buyers):
        st.info("Aucun prospect : tous vos clients achètent déjà cette pièce.")
    else:
        st.info("Aucun prospect : vos clients qui n'ont pas encore pris cette pièce "
                "n'ont acheté aucun des articles-repères de ses acheteurs. Absence de "
                "ressemblance, pas de marché saturé.")
else:
    if source == "model":
        st.caption("Clients qui n'ont **jamais** pris cette pièce, classés par un modèle "
                   "entraîné sur les adoptions réellement constatées l'année suivante.")
    else:
        st.caption(f"Clients qui n'ont **jamais** pris cette pièce, classés par ressemblance "
                   f"avec le panier de ses acheteurs ({len(signature)} articles-repères).")
    target_brand = ctx.article_feats["enseigne"].get(article_id, "")
    target_family = ctx.fam.get(article_id, "")
    for r in block2.itertuples():
        st.markdown(
            f"**{name_of.get(r.client_id, r.client_id)}** · `{r.client_id}` — "
            f"{pmod.prospect_reason(r, signature, target_brand, target_family)} "
            f"· {fmt_tnd(r.client_ca)} de CA total")
    if source == "model":
        # The old wording ("le modèle ne note pas ces clients") became false the moment a
        # model ranked them. What stays true is that no probability is shown: the ordering
        # was measured on held-out data, the calibration was not.
        st.warning("Aucun chiffre de propension n'est affiché ici : seul l'**ordre** de "
                   "cette liste a été mesuré sur des données non vues, pas la probabilité "
                   "elle-même.")
    else:
        st.warning("Le modèle ne note pas ces clients : il n'a aucun historique sur ce "
                   "couple client-article. Ce classement vient de la ressemblance des "
                   "paniers, pas d'une probabilité.")
