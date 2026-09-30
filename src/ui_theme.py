# -*- coding: utf-8 -*-
"""Shared visual language for the Streamlit pages (atelier/dispatch-board direction).

Kept in its own module because src/app.py executes at import time — importing it to
reuse a helper would run the whole vendor app as a side effect.

Direction: warm paper background, dark graphite chrome, industrial condensed type
(Archivo Expanded), monospace for figures/codes (JetBrains Mono), safety-orange
accent — grounded in the auto-parts/workshop domain rather than a generic dashboard.
"""
import streamlit as st

STATUS_META = {
    "INACTIVE_2026": {"label": "Inactif 2026", "css": "urgent"},
    "DECLINING": {"label": "En baisse", "css": "watch"},
    "ACTIVE": {"label": "Actif", "css": "steady"},
}

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Archivo+Expanded:wght@700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --ink: #1B1A17; --paper: #F6F3EC; --line: #E2DBCB; --muted: #6B655A;
  --urgent: #D64B1F; --watch: #8A6A16; --steady: #2F6B4C;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: 'Archivo', -apple-system, sans-serif;
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

/* Dark sidebar chrome. Only a plain (non-!important) color is set on the
   container: Streamlit's own widgets (selectbox/text_input) compute their
   own internal text/background from the theme at render time and do not
   simply inherit this, so they keep their normal light box + dark text and
   stay readable; only plain, unstyled text (labels, captions, our badge
   spans) picks up the light color via ordinary CSS inheritance. */
[data-testid="stSidebar"] { background-color: #232220; color: #EDE8DC; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #B9B2A2;
}
[data-testid="stSidebar"] hr { border-color: #3A3833; }

.eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted);
}
.masthead { padding-bottom: .9rem; margin-bottom: 1.3rem; border-bottom: 3px solid var(--ink); }
.masthead h1 {
  font-family: 'Archivo Expanded', sans-serif; font-weight: 800; font-size: 2rem;
  margin: .2rem 0 .3rem; color: var(--ink); letter-spacing: -.01em;
}
.masthead .sub { color: var(--muted); font-size: .95rem; }

.badge {
  display: inline-flex; align-items: center; gap: .35rem; font-family: 'JetBrains Mono', monospace;
  font-size: .72rem; letter-spacing: .03em; padding: .22rem .6rem; border-radius: 999px; font-weight: 600;
}
.badge-urgent { background: rgba(214,75,31,.12); color: var(--urgent); }
.badge-watch  { background: rgba(138,106,22,.14); color: var(--watch); }
.badge-steady { background: rgba(47,107,76,.14); color: var(--steady); }

.ticket-tag {
  display: inline-block; background: var(--ink); color: var(--paper); font-family: 'JetBrains Mono', monospace;
  font-size: .66rem; letter-spacing: .14em; padding: .28rem .6rem; border-radius: 5px; margin-bottom: .6rem;
}
.hero-name { font-family: 'Archivo Expanded', sans-serif; font-weight: 700; font-size: 1.35rem; color: var(--ink); margin: .2rem 0 .1rem; }
.hero-amount { font-family: 'Archivo Expanded', sans-serif; font-weight: 800; font-size: 2.4rem; color: var(--ink); line-height: 1; }
.hero-amount .unit { font-size: .95rem; font-weight: 600; color: var(--muted); margin-left: .3rem; }
.hero-resume { color: var(--muted); font-size: .92rem; margin-top: .35rem; }

.rank-chip {
  display: inline-flex; align-items: center; justify-content: center; width: 1.5rem; height: 1.5rem;
  border-radius: 50%; background: var(--ink); color: var(--paper); font-family: 'JetBrains Mono', monospace;
  font-size: .72rem; font-weight: 700; margin-right: .4rem; vertical-align: middle;
}
.top-name { font-family: 'Archivo Expanded', sans-serif; font-weight: 700; font-size: 1rem; color: var(--ink); margin: .45rem 0 .1rem; }
.top-amount { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 1.05rem; color: var(--ink); }
.top-resume { color: var(--muted); font-size: .8rem; margin-top: .25rem; line-height: 1.35; }

.section-label {
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin: 1.7rem 0 .6rem; padding-bottom: .35rem; border-bottom: 1px solid var(--line);
}
.article-idx {
  display: inline-flex; align-items: center; justify-content: center; width: 1.3rem; height: 1.3rem;
  border-radius: 4px; background: var(--ink); color: var(--paper); font-family: 'JetBrains Mono', monospace;
  font-size: .7rem; font-weight: 700; margin-right: .45rem;
}
.confidence-note { color: var(--muted); font-size: .82rem; line-height: 1.5; }
.proba-line { font-size: .92rem; color: var(--ink); margin: .35rem 0 .1rem; }
.proba-line b { font-family: 'JetBrains Mono', monospace; font-size: 1.02rem; color: var(--urgent); }
.proba-hint { color: var(--muted); font-size: .78rem; }

/* ---------- master / detail layout ---------- */

.pane-title {
  font-family: 'JetBrains Mono', monospace; font-size: .7rem; letter-spacing: .13em;
  text-transform: uppercase; color: var(--muted);
  padding-bottom: .4rem; margin-bottom: .7rem; border-bottom: 2px solid var(--ink);
  display: flex; justify-content: space-between; align-items: baseline;
}
.pane-title .n { font-size: .78rem; letter-spacing: 0; text-transform: none; }

/* Compact list rows are Streamlit buttons; style them as tickets, not buttons. */
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
  background: #FFFFFF; color: var(--ink); border: 1px solid var(--line);
  border-left: 4px solid var(--line); border-radius: 8px;
  text-align: left; padding: .55rem .8rem; font-weight: 500; line-height: 1.35;
}
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover {
  background: #FBF8F1; border-left-color: var(--urgent); color: var(--ink);
}
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button p { font-size: .84rem; margin: 0; }

/* ---------- the open dossier ---------- */

.dossier-head {
  border-bottom: 2px solid var(--ink); padding-bottom: .8rem; margin-bottom: 1rem;
}
.dossier-head .name {
  font-family: 'Archivo Expanded', sans-serif; font-weight: 700;
  font-size: 1.5rem; line-height: 1.15; margin: .35rem 0 .3rem; color: var(--ink);
}
.dossier-head .meta { color: var(--muted); font-size: .84rem; }
.key-figure {
  font-family: 'Archivo Expanded', sans-serif; font-weight: 800;
  font-size: 2.1rem; line-height: 1; color: var(--ink); margin-top: .6rem;
}
.key-figure .lbl {
  display: block; font-family: 'Archivo', sans-serif; font-weight: 600;
  font-size: .84rem; color: var(--muted); margin-top: .25rem; letter-spacing: 0;
}

/* The line the vendor says out loud — the one thing this screen is for. */
.script {
  background: #F0EDE4; border-left: 4px solid var(--ink); border-radius: 0 10px 10px 0;
  padding: .9rem 1.1rem; margin: 1rem 0 1.2rem; font-size: 1.02rem; line-height: 1.55;
}
.script .lead {
  display: block; font-family: 'JetBrains Mono', monospace; font-size: .66rem;
  letter-spacing: .13em; text-transform: uppercase; color: var(--muted); margin-bottom: .35rem;
}

.step {
  font-family: 'Archivo Expanded', sans-serif; font-weight: 700; font-size: .95rem;
  color: var(--ink); margin: 1.5rem 0 .5rem;
  display: flex; align-items: center; gap: .5rem;
}
.step .num {
  width: 1.35rem; height: 1.35rem; border-radius: 50%; background: var(--ink);
  color: var(--paper); font-family: 'JetBrains Mono', monospace; font-size: .68rem;
  display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto;
}
.art-name { font-weight: 600; font-size: .95rem; line-height: 1.35; }
.empty-detail {
  border: 1px dashed var(--line); border-radius: 10px; padding: 2rem 1.5rem;
  text-align: center; color: var(--muted);
}

.stButton > button {
  background: var(--ink); color: var(--paper); border: none; border-radius: 8px; font-weight: 600;
  font-family: 'Archivo', sans-serif; letter-spacing: .01em;
}
.stButton > button:hover { background: var(--urgent); color: #fff; }
[data-testid="stForm"] { border: 1px dashed var(--line); border-radius: 10px; }
</style>
"""

# Additional rules used only by the validation page.
_VALIDATION_CSS = """
<style>
.verdict-hit  { color: var(--steady); font-weight: 700; }
.verdict-miss { color: var(--urgent); font-weight: 700; }
.caveat {
  border-left: 3px solid var(--watch); background: rgba(138,106,22,.07);
  padding: .8rem 1rem; margin: .8rem 0; font-size: .88rem; color: var(--ink);
}
.caveat strong { font-family: 'JetBrains Mono', monospace; font-size: .78rem;
  letter-spacing: .08em; text-transform: uppercase; color: var(--watch); }
.metric-note { color: var(--muted); font-size: .82rem; margin-top: -.6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(_VALIDATION_CSS, unsafe_allow_html=True)


def badge_html(status: str) -> str:
    meta = STATUS_META.get(status, {"label": status, "css": "steady"})
    return f'<span class="badge badge-{meta["css"]}">● {meta["label"]}</span>'


def fmt_tnd(x) -> str:
    return f"{x:,.0f} TND"
