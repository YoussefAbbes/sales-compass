# -*- coding: utf-8 -*-
"""Validation page — is the model's prediction actually true?

Run:  streamlit run src/app.py   (this page appears in the sidebar nav)

Grades the v4 embargoed rolling model on its RETROSPECTIVE test window
(2026-01 … 2026-04), four snapshots the model never saw and whose outcome is
fully observed in the NAV ledger. It does not track live recommendations: that
needs vendor feedback that has not been collected yet.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validation as v
from ledger import COLUMNS as LEDGER_COLS, monthly_ca
from ui_theme import fmt_tnd, inject_css

st.set_page_config(page_title="Validation du modèle", page_icon="🎯", layout="wide")
inject_css()


@st.cache_data
def load_all():
    metrics = v.load_metrics()
    test = v.load_scores("test")
    # Full ledger. ca_monthly.parquet is now generated from it (src/ledger.py),
    # but reading the ledger directly keeps this page independent of that step.
    fact = monthly_ca(pd.read_parquet(v.DATA / "transactions.parquet",
                                      columns=LEDGER_COLS))
    articles = pd.read_csv(v.DATA / "articles.csv", dtype=str, encoding="utf-8-sig")
    return metrics, test, fact, articles


try:
    metrics, test, fact, articles = load_all()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

base = float(test["label"].mean())

st.markdown(
    '<div class="masthead"><div class="eyebrow">Atelier — contrôle qualité</div>'
    '<h1>Validation du modèle</h1>'
    '<div class="sub">Les prédictions étaient-elles vraies ? Quatre mois de test '
    '(janvier → avril 2026) que le modèle n\'a jamais vus.</div></div>',
    unsafe_allow_html=True,
)

# ---------------- §1 Le protocole ----------------
st.markdown('<div class="section-label">1 · Le protocole</div>', unsafe_allow_html=True)

st.markdown(
    "Le modèle est entraîné sur des instantanés mensuels et jugé sur des mois "
    "**postérieurs**, jamais sur les mêmes. Entre chaque bloc, des mois sont "
    "**jetés** : c'est l'embargo, et c'est ce qui rend le score honnête."
)

proto = pd.DataFrame([
    {"Rôle": "Entraînement", "Fenêtre": metrics["train"]["window"],
     "Instantanés": metrics["train"]["n_snapshots"],
     "Paires": metrics["train"]["n_pairs"], "AUC": metrics["train"]["auc"]},
    {"Rôle": "Embargo (jeté)", "Fenêtre": metrics["embargo_train_val"]["window"],
     "Instantanés": metrics["embargo_train_val"]["n_snapshots"],
     "Paires": None, "AUC": None},
    {"Rôle": "Validation", "Fenêtre": metrics["validation"]["window"],
     "Instantanés": metrics["validation"]["n_snapshots"],
     "Paires": metrics["validation"]["n_pairs"], "AUC": metrics["validation"]["auc"]},
    {"Rôle": "Embargo (jeté)", "Fenêtre": metrics["embargo_val_test"]["window"],
     "Instantanés": metrics["embargo_val_test"]["n_snapshots"],
     "Paires": None, "AUC": None},
    {"Rôle": "Test (intouché)", "Fenêtre": metrics["test"]["window"],
     "Instantanés": metrics["test"]["n_snapshots"],
     "Paires": metrics["test"]["n_pairs"], "AUC": metrics["test"]["auc"]},
])
st.dataframe(
    proto.style.format({"Paires": "{:,.0f}", "AUC": "{:.3f}"}, na_rep="—"),
    hide_index=True, width="stretch",
)

c1, c2, c3 = st.columns(3)
c1.metric("AUC test", f"{metrics['test']['auc']:.3f}")
c2.metric("Écart train − test", f"{metrics['gap_train_test']:+.3f}",
          help="Un écart important signale un sur-apprentissage : le modèle mémorise "
               "ses mois d'entraînement au lieu d'apprendre une règle générale.")
c3.metric("Écart validation − test", f"{metrics['gap_validation_test']:+.3f}",
          help="Négatif = la validation SOUS-estime le test. C'est la bonne direction : "
               "on ne peut pas se flatter en optimisant dessus.")

st.markdown(
    f'<div class="caveat"><strong>Pourquoi deux embargos</strong><br>'
    f'L\'étiquette d\'un instantané regarde {metrics["horizon_months"]} mois en avant. '
    f'Sans embargo, les derniers mois d\'entraînement étiquettent les mêmes achats que '
    f'la validation : le modèle est alors <b>payé pour mémoriser</b>. '
    f'Une version à 6000 itérations a fait exactement cela — validation 0.781 → 0.788 '
    f'pendant que le test réel <b>baissait</b> 0.775 → 0.771. '
    f'Avec l\'embargo train → validation, l\'écart validation − test est passé de '
    f'<b>+0.016 à {metrics["gap_validation_test"]:+.3f}</b> : la validation est devenue '
    f'légèrement <i>pessimiste</i>, ce qui est la seule direction sûre. '
    f'L\'arrêt anticipé s\'est déclenché à l\'itération {metrics["best_iteration"]} '
    f'(plafond {metrics["iterations_cap"]}) — '
    f'{"résultat convergé" if metrics["converged"] else "plafond atteint, pas convergé"}.'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------- §2 Performance agrégée ----------------
st.markdown('<div class="section-label">2 · Performance agrégée</div>', unsafe_allow_html=True)

st.markdown("**Stabilité mois par mois** — un modèle qui ne marche qu'un mois n'est pas déployable.")
snap = pd.DataFrame(
    [{"Instantané": k, "AUC": val} for k, val in metrics["auc_per_snapshot"].items()])
s1, s2 = st.columns([2, 1])
s1.bar_chart(snap.set_index("Instantané")["AUC"], height=220)
s2.metric("Écart max entre mois",
          f"{snap['AUC'].max() - snap['AUC'].min():.4f}",
          help="Plus c'est petit, plus le modèle est régulier d'un mois sur l'autre.")

st.markdown("**Lift par décile** — les paires les mieux notées rachètent-elles davantage ?")
dec = v.decile_table(test["p_model"], test["label"])
st.bar_chart(dec.set_index("decile")["rate"], height=240)
st.markdown(
    f'<div class="metric-note">Taux de base : {base:.1%}. '
    f'Décile 10 : {dec.iloc[-1]["rate"]:.1%} (×{dec.iloc[-1]["rate"] / base:.1f}). '
    f'Décile 1 : {dec.iloc[0]["rate"]:.1%}.</div>',
    unsafe_allow_html=True,
)

st.markdown("**Calibration** — quand le modèle annonce 60 %, est-ce que 60 % achètent vraiment ?")
cal = v.calibration_table(test["p_model"], test["label"], n_bins=10)
cal_display = cal.assign(
    Tranche=cal.apply(lambda r: f"{r['bin_low']:.0%}–{r['bin_high']:.0%}", axis=1)
)[["Tranche", "n", "mean_pred", "observed", "gap"]].rename(
    columns={"n": "Paires", "mean_pred": "Probabilité annoncée",
             "observed": "Rachat observé", "gap": "Écart"})
st.dataframe(
    cal_display.style.format({"Paires": "{:,.0f}", "Probabilité annoncée": "{:.1%}",
                              "Rachat observé": "{:.1%}", "Écart": "{:+.1%}"}),
    hide_index=True, width="stretch",
)
st.line_chart(cal.set_index("mean_pred")[["observed"]], height=240)
st.markdown(
    f'<div class="metric-note">Un modèle parfaitement calibré suit la diagonale : '
    f'probabilité annoncée = rachat observé. Écart maximal observé ici : '
    f'<b>{cal["gap"].max():+.1%}</b>. '
    f'Contrairement à la version précédente, les étiquettes sont ici <b>complètes</b> : '
    f'le grand livre observe les {metrics["horizon_months"]} mois de chaque fenêtre, '
    f'donc cet écart mesure le modèle et non un angle mort des données. '
    f'La calibration n\'a pas été corrigée : c\'est pourquoi la file d\'appels présente '
    f'ces nombres comme un <i>classement</i> et non comme des probabilités réelles.</div>',
    unsafe_allow_html=True,
)

st.markdown("**Matrice de confusion** — à partir de quelle probabilité décide-t-on d'appeler ?")
thr = st.slider("Seuil de décision", 0.0, 1.0, float(round(base, 3)), 0.01,
                help="Par défaut : le taux de base. Au-dessus de ce seuil, on considère "
                     "que le client va racheter.")
cm = v.confusion_at(test["p_model"], test["label"], thr)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vrais positifs", f"{cm['tp']:,}")
m2.metric("Faux positifs", f"{cm['fp']:,}")
m3.metric("Précision", f"{cm['precision']:.1%}")
m4.metric("Rappel", f"{cm['recall']:.1%}")
st.markdown(
    f'<div class="metric-note">Au seuil {thr:.2f}, on contacterait {cm["n_flagged"]:,} paires '
    f'sur {len(test):,}. Précision et rappel se déplacent en sens inverse : '
    f'monter le seuil augmente la précision (moins d\'appels inutiles) mais rate '
    f'plus d\'opportunités.</div>',
    unsafe_allow_html=True,
)

# ---------------- §3 Les recommandations ----------------
st.markdown('<div class="section-label">3 · Les recommandations</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="caveat"><strong>Portée de ce rejeu</strong><br>'
    'Ce qui est rejoué : le classement par <b>valeur attendue</b> (p × CA de la fenêtre) '
    'et le top-3 articles par client. Ce qui ne l\'est pas : le boost de statut client '
    'et la suggestion complémentaire. Les chiffres ci-dessous jugent donc le '
    '<i>classement du modèle</i>, pas la fiche complète telle qu\'un vendeur la reçoit.</div>',
    unsafe_allow_html=True,
)

st.markdown("**Le modèle contre la règle actuelle** — trier par CA passé, ce que fait le métier aujourd'hui :")
p1, p2, p3 = st.columns(3)
p1.metric("Top 30 / vendeur — modèle", f"{metrics['precision_top30_model']:.1%}")
p2.metric("Top 30 / vendeur — CA passé", f"{metrics['precision_top30_revenue']:.1%}")
p3.metric("Gain", f"×{metrics['precision_top30_model'] / metrics['precision_top30_revenue']:.2f}",
          help="Sur les 30 suggestions qu'un vendeur traite réellement.")

st.markdown("**Précision @ N** — si un vendeur ne traite que ses N meilleures paires :")
rows = []
for n in (5, 10, 30):
    r_model = v.precision_at_n(test, "p_model", n)
    r_ca = v.precision_at_n(test, "ca_sum", n)
    rows.append({"N": n, "Paires listées": r_model["n_pairs"],
                 "Précision — modèle": r_model["precision"],
                 "Précision — CA passé": r_ca["precision"],
                 "Taux de base": base})
st.dataframe(
    pd.DataFrame(rows).style.format({
        "Paires listées": "{:,.0f}", "Précision — modèle": "{:.1%}",
        "Précision — CA passé": "{:.1%}", "Taux de base": "{:.1%}"}),
    hide_index=True, width="stretch",
)


@st.cache_data
def build_replay():
    el = v.eligible_pairs(fact, (2024, 2025))
    q = v.queue_replay(test, el, top_k=3)
    real = v.realized_ca(fact, 2026)
    q = q.merge(real, on=["client_id", "article_id"], how="left")
    q["ca_realized"] = q["ca_realized"].fillna(0.0)
    return q


replay = build_replay()

st.markdown("**Ce que ça représente en argent** (top-3 articles par client, pool éligible) :")
hit = replay[replay["label"] == 1]
g1, g2, g3 = st.columns(3)
g1.metric("Paires recommandées", f"{len(replay):,}")
g2.metric("Rachetées sur la fenêtre", f"{len(hit):,}",
          delta=f"{len(hit) / len(replay):.1%} de réussite" if len(replay) else None)
g3.metric("CA réellement réalisé", fmt_tnd(hit["ca_realized"].sum()),
          help="CA 2026 effectivement observé sur les paires recommandées et rachetées.")
st.markdown(
    f'<div class="metric-note">Valeur attendue annoncée sur ces paires : '
    f'{fmt_tnd(replay["valeur_attendue"].sum())}. CA réalisé : '
    f'{fmt_tnd(replay["ca_realized"].sum())}. L\'écart entre les deux mesure '
    f'l\'optimisme du classement. Le grand livre s\'arrête au 30 juin 2026, donc le '
    f'réalisé reste un minorant sur la fin de période.</div>',
    unsafe_allow_html=True,
)

# ---------------- §4 Inspection individuelle ----------------
st.markdown('<div class="section-label">4 · Inspection individuelle</div>', unsafe_allow_html=True)
st.markdown("Descendre jusqu'à une prédiction précise et vérifier ce qui s'est réellement passé.")

art_lookup = articles.set_index("article_id")["designation"].to_dict()

i1, i2 = st.columns([1, 2])
vendors = sorted(test["vendor_code"].dropna().unique())
vendor = i1.selectbox("Vendeur", vendors)
scope = test[test["vendor_code"] == vendor]

view = i2.radio(
    "Afficher",
    ["Tout", "Réussites uniquement", "Échecs uniquement", "Erreurs les plus confiantes"],
    horizontal=True,
)

clients = (scope.groupby("client_id")
                .agg(paires=("article_id", "size"), rachats=("label", "sum"))
                .sort_values("paires", ascending=False)
                .reset_index())
client = st.selectbox(
    "Client",
    clients["client_id"],
    format_func=lambda c: (
        f"{c} — {int(clients.loc[clients.client_id == c, 'paires'].iloc[0])} paires, "
        f"{int(clients.loc[clients.client_id == c, 'rachats'].iloc[0])} rachetées"
    ),
)

detail = scope[scope["client_id"] == client].copy()
detail["Désignation"] = detail["article_id"].map(art_lookup).fillna("(désignation inconnue)")
detail["Réalité"] = detail["label"].map({1: "✓ acheté", 0: "✗ pas acheté"})
detail["Valeur attendue"] = detail["p_model"] * detail["ca_sum"]

if view == "Réussites uniquement":
    detail = detail[detail["label"] == 1]
elif view == "Échecs uniquement":
    detail = detail[detail["label"] == 0]
elif view == "Erreurs les plus confiantes":
    detail = detail[detail["label"] == 0].sort_values("p_model", ascending=False)

if detail.empty:
    st.info("Aucune paire ne correspond à ce filtre pour ce client.")
else:
    show = (detail.sort_values("p_model", ascending=False)
                  .rename(columns={"article_id": "Article", "p_model": "Probabilité",
                                   "ca_sum": "CA fenêtre"})
            [["Article", "Désignation", "Probabilité", "CA fenêtre",
              "Valeur attendue", "Réalité"]])
    st.dataframe(
        show.style.format({"Probabilité": "{:.1%}", "CA fenêtre": "{:,.0f}",
                           "Valeur attendue": "{:,.0f}"}),
        hide_index=True, width="stretch", height=420,
    )
    st.markdown(
        f'<div class="metric-note">{len(detail):,} paire(s) affichée(s). '
        f'Sur l\'ensemble du client : {int(scope[scope.client_id == client]["label"].sum())} '
        f'rachat(s) sur {len(scope[scope.client_id == client])} paires.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Modèle v4 embargé — test sur quatre instantanés intouchés (2026-01 → 2026-04), "
    "étiquettes complètes sur le grand livre NAV. Cette page ne suit pas les "
    "recommandations en cours : cela demandera les retours vendeurs, qui ne sont pas "
    "encore collectés."
)
