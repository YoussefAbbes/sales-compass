# -*- coding: utf-8 -*-
"""Streamlit demo — vendor recommendation interface ("Clients à contacter").

Run:  streamlit run src/app.py   (or: py -m streamlit run src/app.py)
Local only: reads data/ (confidential), writes vendor feedback to data/feedback.csv.

Layout: master / detail. The call list stays on the left while a client's file opens
on the right, because a vendor alternates between "who is next" and "what do I say" —
a single scrolling page made them travel between the two on every call.

The right pane is ordered by the moment of the call, not by the shape of the data:
what to say → what to offer → what to add → their history → what happened.

Design note: this app relies only on Streamlit mechanisms that are provably stable —
native theming (.streamlit/config.toml), documented data-testid selectors, and plain
HTML/CSS this file fully controls.
"""
from datetime import datetime
from itertools import count
from pathlib import Path

import pandas as pd
import streamlit as st

from call_script import initiales, mois_fr, nom_utilisable, phrase_appel
from ledger import COLUMNS as LEDGER_COLS, monthly_ca
from ui_theme import STATUS_META, badge_html, fmt_tnd, inject_css

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEEDBACK = DATA / "feedback.csv"

ACTIONS = ["Contacté", "Devis envoyé", "Commande", "À rappeler", "Refus",
           "Client fermé", "Injoignable"]
LIST_HEIGHT = 620   # px; the queue scrolls inside this, the fiche stays put

st.set_page_config(page_title="Clients à contacter", page_icon="📞", layout="wide")


@st.cache_data
def load():
    q = pd.read_csv(DATA / "clients_to_contact.csv",
                    dtype={"vendor_code": str, "client_id": str}, encoding="utf-8-sig")
    # The full ledger (2024-01 .. 2026-06), not ca_monthly.parquet, which holds
    # January-May only. The chart used to apologise for that hole; now there
    # isn't one. Cached, so the 1.9M-row read happens once per session.
    fact = monthly_ca(pd.read_parquet(DATA / "transactions.parquet",
                                      columns=LEDGER_COLS))
    return q, fact


def save_feedback(vendor, client_id, client_name, action, comment):
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "vendor_code": vendor, "client_id": client_id, "client_name": client_name,
        "action": action, "comment": comment,
    }])
    header = not FEEDBACK.exists()
    row.to_csv(FEEDBACK, mode="a", header=header, index=False, encoding="utf-8-sig")


inject_css()
q, fact = load()

# ---------------- Sidebar ----------------
st.sidebar.markdown('<div class="eyebrow">Atelier — file d\'appels</div>', unsafe_allow_html=True)
st.sidebar.title("📞 Clients à contacter")

vendor_options = q[["vendor_code", "vendor_name"]].drop_duplicates().sort_values("vendor_code")
vendor_label = st.sidebar.selectbox(
    "Vendeur",
    vendor_options.apply(lambda r: f"{r.vendor_code} — {r.vendor_name}", axis=1),
)
vendor_code, vendor_name = vendor_label.split(" — ")[0], vendor_label.split(" — ")[1]

vq = (q[q["vendor_code"] == vendor_code]
      .sort_values("priority_score", ascending=False).reset_index(drop=True))

st.sidebar.divider()
c1, c2 = st.sidebar.columns(2)
c1.metric("Clients à appeler", len(vq))
c2.metric("TND concernés", f"{vq.montant_cle.sum():,.0f}",
          help="Somme des montants concrets : baisse de CA pour les clients en recul, "
               "valeur des références suivies pour les autres.")
st.sidebar.markdown(
    f'<span class="badge badge-urgent">● {int((vq.status == "INACTIVE_2026").sum())} inactifs</span> '
    f'<span class="badge badge-watch">● {int((vq.status == "DECLINING").sum())} en baisse</span> '
    f'<span class="badge badge-steady">● {int((vq.status == "ACTIVE").sum())} actifs</span>',
    unsafe_allow_html=True,
)
st.sidebar.divider()
search = st.sidebar.text_input("🔍 Rechercher un client")
st.sidebar.caption("Les clients sont triés du plus important au moins important.")

# ---------------- Masthead ----------------
today = datetime.now().strftime("%d/%m/%Y")
st.markdown(f"""
<div class="masthead">
  <div class="eyebrow">File d'appels · Vendeur {vendor_code} · {today}</div>
  <h1>{vendor_name}</h1>
  <div class="sub">{len(vq)} clients à appeler aujourd'hui. Choisissez-en un à gauche.</div>
</div>
""", unsafe_allow_html=True)

view = vq.copy()
if search:
    view = view[view["client_name"].str.contains(search, case=False, na=False)].reset_index(drop=True)

if not len(view):
    st.info(f"Aucun client ne correspond à « {search} ». Essayez un autre nom, "
            f"ou effacez la recherche.")
    st.stop()

# The open client must always exist in the current view (vendor switch, search, …).
if st.session_state.get("open_client_id") not in view["client_id"].tolist():
    st.session_state["open_client_id"] = view.iloc[0]["client_id"]

left, right = st.columns([1, 1.9], gap="large")

# ---------------- Left: the queue ----------------
with left:
    st.markdown(
        f'<div class="pane-title"><span>Ma file</span>'
        f'<span class="n">{len(view)} client(s)</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(height=LIST_HEIGHT, border=False):
        for rank, (_, r) in enumerate(view.iterrows(), start=1):
            opened = r["client_id"] == st.session_state["open_client_id"]
            mark = "▸ " if opened else ""
            nom_row = nom_utilisable(r["client_name"]) or f"Client {r['client_id']}"
            label = (f"{mark}**{rank}. {nom_row}**  \n"
                     f"{r['montant_cle']:,.0f} TND · {r['montant_texte']}")
            if st.button(label, key=f"open_top_{r['client_id']}", width="stretch"):
                st.session_state["open_client_id"] = r["client_id"]
                st.rerun()

client_id = st.session_state["open_client_id"]
row = view[view["client_id"] == client_id].iloc[0]

# ---------------- Right: the open file ----------------
with right:
    st.markdown('<div class="pane-title"><span>Le dossier</span>'
                f'<span class="n">{row["client_id"]}</span></div>',
                unsafe_allow_html=True)

    nom = nom_utilisable(row["client_name"])
    titre = nom or f"Client {row['client_id']}"
    sous_titre = "" if nom else " — nom absent de l'extrait"
    st.markdown(f"""
    <div class="dossier-head">
      {badge_html(row['status'])}
      <div class="name">{initiales(nom, row['client_id'])} · {titre}</div>
      <div class="meta">Dernier achat : {mois_fr(row['last_activity']) or '—'}
        · {int(row['n_lapsed_total'])} références suivies{sous_titre}</div>
      <div class="key-figure">{row['montant_cle']:,.0f} TND
        <span class="lbl">{row['montant_texte']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # 1 — the sentence to say
    st.markdown(
        f'<div class="script"><span class="lead">Ce qu\'il faut dire</span>'
        f'{phrase_appel(row["client_name"], row["status"], row["last_activity"], row.get("article_txt_1"))}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Steps follow the order of the call. The number is issued at render time rather
    # than hardcoded: a client with no cross-sell suggestion would otherwise show
    # steps 1, 3, 4 and look broken.
    etapes = count(1)

    def step(titre: str) -> None:
        st.markdown(f'<div class="step"><span class="num">{next(etapes)}</span>'
                    f'{titre}</div>', unsafe_allow_html=True)

    step("À lui proposer")
    shown = 0
    for i in (1, 2, 3):
        art, reason, proba = row.get(f"article_txt_{i}"), row.get(f"reason_{i}"), row.get(f"proba_{i}")
        if pd.notna(art):
            shown += 1
            with st.container(border=True):
                st.markdown(f'<span class="article-idx">{i}</span>'
                            f'<span class="art-name">{art}</span>', unsafe_allow_html=True)
                if pd.notna(proba):
                    st.markdown(
                        f'<div class="proba-line"><b>{proba:.0f} %</b> '
                        f'de chance qu\'il le reprenne '
                        f'<span class="proba-hint">(estimation optimiste)</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(f'<div class="confidence-note">{reason}</div>',
                            unsafe_allow_html=True)
    if not shown:
        st.markdown('<div class="empty-detail">Aucun article précis à proposer.<br>'
                    'Ouvrez la conversation sur ses besoins du moment.</div>',
                    unsafe_allow_html=True)
    else:
        st.caption("Articles classés par **montant en jeu** (valeur du poste × chance "
                   "de rachat), pas seulement par chance.")
        st.caption("Les pourcentages ne sont **pas calibrés** : lisez-les comme un "
                   "classement, pas comme une probabilité réelle. Le modèle est mesuré "
                   "sur son pouvoir de tri — 62 % des 30 premières suggestions par "
                   "vendeur donnent lieu à un achat, contre 32 % en classant par "
                   "chiffre d'affaires passé.")

    # 3 — the extra sale
    if pd.notna(row.get("suggestion_complementaire")):
        step("Si ça se passe bien")
        with st.container(border=True):
            st.markdown(f'<span class="art-name">{row["suggestion_complementaire"]}</span>',
                        unsafe_allow_html=True)
            st.markdown('<div class="confidence-note">Souvent commandé en même temps '
                        'par d\'autres clients.</div>', unsafe_allow_html=True)

    # 4 — context
    step("Ses achats, mois par mois")
    hist = (fact[fact["client_id"] == client_id]
            .groupby("month")["ca"].sum().reset_index()
            .assign(mois=lambda d: d["month"].dt.strftime("%Y-%m")))
    if len(hist):
        st.bar_chart(hist.set_index("mois")["ca"], height=200, color="#D64B1F")
        st.caption("Historique complet du grand livre (janvier 2024 → juin 2026). "
                   "Les barres négatives sont des avoirs.")
    else:
        st.markdown('<div class="empty-detail">Aucun historique disponible.</div>',
                    unsafe_allow_html=True)

    # 5 — close the loop
    step("Après l'appel")
    with st.form("feedback", clear_on_submit=True):
        action = st.selectbox("Action", ACTIONS)
        comment = st.text_input("Commentaire (facultatif)")
        if st.form_submit_button("Enregistrer"):
            save_feedback(vendor_code, client_id, row["client_name"], action, comment)
            st.success("Enregistré ✔")

    if FEEDBACK.exists():
        fb = pd.read_csv(FEEDBACK, dtype={"client_id": str}, encoding="utf-8-sig")
        fb_client = fb[fb["client_id"] == client_id]
        if len(fb_client):
            st.caption("Appels déjà notés pour ce client :")
            st.dataframe(fb_client[["timestamp", "action", "comment"]],
                         hide_index=True, width="stretch", height=140)
