# Progress Log

> Chronological working log kept during the internship. It is historical: some scripts and reports it links to (`backtest.py`, `train_model.py`, `api/`) were later replaced by the v4 pipeline described in the [README](../../README.md).

## 2026-07-31 — Repository published

Git repository created (README, requirements.txt added).
**Confidential data excluded** via `.gitignore`: the raw Excel and all `data/` outputs contain real client names and revenue — only code and documentation are versioned.

## 2026-07-31 — Dataset built from the CA Excel extract

### Input

`CA client vendeur article Janv-mai  24-25-26.xlsx` (51 MB, 2 sheets):

- **Janv-fev-mars-avril**: 581K rows — vendor / client / article codes + 12 monthly CA columns (Jan–Apr × 2024/2025/2026).
- **Mai**: 173K rows — same keys + vendor name, client name, article brand (enseigne) & designation + 3 May columns.
- CA = revenue in TND (3 decimals). `-` = no sale.

### What was done

Script: [`src/build_dataset.py`](../../src/build_dataset.py) — Python 3.13, `openpyxl` (read Excel), `pandas` + `pyarrow` (transform & save).

1. **Unpivoted** both sheets into one long "fact table": one row = `(vendor_code, client_id, article_id, month, ca)`.
2. **Extracted lookup tables** from the Mai sheet (the only sheet with names).
3. **Cleaning rules applied:**
   - IDs kept as text, whitespace stripped (preserves leading zeros like `007`).
   - Empty cells (`-`) not stored → sparse table.
   - CA rounded to 3 decimals (millime precision, removes float noise); exact-zero cells dropped (6).
   - Duplicate keys **summed** — 355 found, all in the Mai sheet (same key split across rows: partial amounts + small correction rows).
   - **Negative CA kept** (returns/credit notes are netted into CA by the source system). Each analysis decides how to treat them.
4. **Validation:** per-month CA totals of the output match the source sheets exactly; no duplicate keys remain.

### Outputs (in `data/`)

| File | Content | Size |
| --- | --- | --- |
| `ca_monthly.parquet` | 896,312 rows: vendor, client, article, month, CA | 5.5 MB |
| `vendors.csv` | 44 vendor codes → names | 1 KB |
| `clients.csv` | 1,778 client ids → names | 52 KB |
| `articles.csv` | 24,683 article ids → brand + designation | 1.4 MB |

### Key numbers & findings

- 54 vendors, 2,060 clients, 40,321 articles, 15 observed months (Jan–May of 2024/2025/2026).
- Monthly CA is stable across years. May 2026 is noticeably lower (−29 % vs May 2025) — to confirm with the company (real drop or partial extract?).
- ~3.9% of observations are negative (returns netted in) → source system nets returns into CA.
- 1,562 clients appear under several vendor codes → vendor–client is many-to-many; an ownership rule is needed.
- Lookups only cover what was sold in May: article names exist for 24.7K of 40.3K articles; client names for 1,778 of 2,060.

### Limits of this data (vs. project.md needs)

Monthly aggregates only — no order dates, quantities, statuses, favorites, stock, contacts.
Only Jan–May of each year (Jun–Dec invisible). → Reorder logic works at month granularity; CatBoost training deferred until transactional data exists.

## 2026-07-31 — Baseline recommendation lists (no ML)

Script: [`src/baseline_lists.py`](../../src/baseline_lists.py) — pandas + scipy (sparse matrix for co-occurrence).
Rule used everywhere: a "purchase" = a month with net CA > 0 (negative months = returns).

### 1. Lapsed reorders → `data/lapsed_reorders.csv` (+ `lapsed_top30_per_vendor.csv`)

- Definition (strict): (client, article) bought in **2024 AND 2025** (Jan–May windows), **nothing in 2026**.
- Computed at client level across all vendors (avoids false alarms when a client switched vendors), then attributed to the vendor with the largest historical CA on the pair.
- Score = historical CA × regularity (months bought / 10). Reason text in French for vendors.
- **Result: 28,119 pairs of historical CA at stake.** Top items are motor oils (recurring consumables) — good face validity.

### 2. Client reactivation → `data/client_reactivation.csv`

- Basis: Jan–Apr only (May 2026 might be incomplete). Filter: ≥ 500 TND CA in Jan–Apr 2025.
- `INACTIVE_2026` = no CA in 2026 (141 clients). `DECLINING` = drop ≥ 30% vs 2025 (348 clients).
- **Flagged clients represent a material share of Jan–Apr 2025 revenue.**

### 3. Cross-sell candidates → `data/cross_sell_pairs.csv`

- Basket = (client, month). Top 5,000 articles, pairs with ≥ 5 shared baskets and lift > 1, top 10 partners per article.
- 18,678 baskets → 49,998 pairs. Caveat: co-occurrence finds substitutes as well as complements (e.g. coolant color variants).

### Data-quality discovery

Client names like `***STE X*** 09-25` (56 clients) look like **closed/blocked accounts** (date = closure?). Added a `client_flagged` column; they are excluded from the per-vendor top-30 file. **To confirm with the company.**

### Tunable thresholds (constants at top of the script, to validate with stakeholders)

Min client CA (500 TND), decline threshold (−30%), min shared baskets (5), top partners (10), top per vendor (30).

## 2026-07-31 — "Clients to contact today" queue

Script: [`src/contact_queue.py`](../../src/contact_queue.py) → `data/clients_to_contact.csv`.
One row per client, ranked per vendor, combining the 3 signals:

- Top-3 lapsed articles (with French reason texts) + one cross-sell suggestion the client never bought.
- Client status: `INACTIVE_2026` / `DECLINING` / `ACTIVE` (from the reactivation list).
- **Priority = max(valeur_reachat, manque_a_gagner) × status boost** — both measured in TND.
  Max, not sum, because the revenue gap already partly contains the lapsed articles (no double counting).
- One row per client (dominant-vendor ownership rule); closed/blocked clients (`***`) excluded.

**Result: 1,204 clients across 38 vendors (median 28 per vendor).**
Face-validity check passed (e.g. top article "bobine allumage Clio" → suggestion "jeu fils bougies Clio").

## 2026-07-31 — Backtest: the signals predict (no leakage)

Script: [`src/backtest.py`](../../src/backtest.py) → report [`reports/backtest.md`](../../reports/backtest.md)
(aggregates only — no client data, safe in the repo). All tests train on past, measure on future.

| Test | Result |
| --- | --- |
| T1 Regularity → repurchase | 1 month bought → ~15%; 5 months → ~80%. Stable across 2024→25 and 2025→26. |
| T2 Both-years rule | Bought 2024 AND 2025 → 39% rebuy in 2026, vs 10–14% for one-year pairs (×3–4). |
| T3 Score ranking | Monotonic deciles 13.5% → 40.2%. **Top-30/vendor list built on 2024 only: 52% repurchased in 2025 (lift ×2.6).** |
| T4 Decline → churn | Declining clients: 21% inactive next year vs 6% for stable (×3.4). Inactive clients: 77% stay lost. |

Caveat: Jan–May windows only → all rates are lower bounds (Jun–Dec repurchases invisible).

## 2026-07-31 — Vendor call sheets (Excel)

Script: [`src/vendor_sheets.py`](../../src/vendor_sheets.py) — openpyxl.

- `data/fiches_vendeurs/` — **38 workbooks, one per vendor** (a vendor only receives their own
  clients — permission principle of project.md §12). Per client: statut coloré, priorité (TND),
  résumé, 3 articles recommandés avec raisons, suggestion complémentaire.
- Feedback columns built in: **"Action vendeur" dropdown** (Contacté / Devis envoyé / Commande /
  À rappeler / Refus / Client fermé / Injoignable) + free comment → manual version of the
  Phase 2 feedback loop, usable before any application exists.
- `data/synthese_direction.xlsx` — per-vendor overview for management (clients, statuts, TND en jeu),
  sorted by TND at stake.

## 2026-07-31 — CatBoost purchase-propensity model

Script: [`src/train_model.py`](../../src/train_model.py) → report [`reports/model_catboost.md`](../../reports/model_catboost.md).
Question: pair bought in year Y (Jan–May) → bought again in Jan–May Y+1?
**Train 2024→2025, test untouched 2025→2026** (chronological, features from snapshot year only — no leakage).

| Model (test 2025→2026) | AUC | Top decile | Precision top-30/vendor |
| --- | --- | --- | --- |
| Manual score (current queue) | 0.616 | 37.7% | 42.3% |
| Logistic regression (benchmark) | 0.738 | 49.8% | 59.0% |
| **CatBoost** | **0.762** | **53.0%** | **60.4%** |

- Decile gradient monotonic: 3.0% → 53.0% (base rate 18.4%).
- Top features: client health (`client_n_articles`, `client_ca`, `client_late_share`), brand (`enseigne`),
  article popularity — the model learned that *who the client is* matters more than pair regularity alone.
- Tools: `catboost`, `scikit-learn`. No tuning yet (plan.md: baseline first). Model + scored pairs saved in `data/`.

## 2026-07-31 — Queue re-ranked by the model

[`src/contact_queue.py`](../../src/contact_queue.py) updated + fiches regenerated:

- Article ranking now uses **expected value = p_model × CA_2025** (model probability × last year's revenue)
  instead of the manual score. Priority = max(valeur_attendue, manque_a_gagner) × status boost.
- Each recommendation now shows a **confidence label** (Élevée ≥50% / Moyenne ≥25% / Faible) — labels,
  not raw percentages, per project.md §9 (calibration not yet validated). Explanations stay rule-based.
- Graceful fallback to the manual score if the model scores file is absent.
- Effect vs the manual ranking: **42.9% of clients got a different #1 article**; top-10 clients per vendor
  overlap 83.8% (client order is stable, article choice is where the model helps).
  TND at stake now probability-weighted: 7.1 M (was 7.8 M raw historical).

**Pipeline order is now:** build_dataset → baseline_lists → train_model → contact_queue → vendor_sheets.

## 2026-07-31 — Streamlit demo app + model iteration bump

### Demo app — [`src/app.py`](../../src/app.py) (`streamlit run src/app.py`, local only)

Interactive version of the fiches (plan.md Phase 2 in miniature):

- Sidebar: vendor selector + queue stats + client search.
- Main: ranked queue table → client detail (statut, priorité, articles recommandés avec raisons
  et confiance, suggestion complémentaire, **historique CA mensuel en graphique**).
- **Feedback form** (action dropdown + commentaire) → appends to `data/feedback.csv` with timestamp —
  the digital feedback loop, same 7 actions as the Excel fiches.
- Tested with Streamlit's `AppTest` (load, vendor switch, feedback write). One real bug found &
  fixed by the tests: empty client list (search without match) crashed the detail view.

### Model iteration experiment → reverted (a finding, not a failure)

The 800-iteration run never triggered early stopping, so a 4000-iteration run was tried:
**validation AUC kept improving (0.798 → 0.810) but the chronological test degraded (0.762 → 0.748).**
Cause: the validation split is a random slice of the *same 2024 snapshot* as training — improving on
it rewards memorizing 2024-specific patterns that don't transfer to the next year.
→ Reverted to 800 iterations (test AUC 0.762) and **stopped tuning**: with a single honest test
window, further configuration search would fit the test set. Documented in `src/train_model.py`.
Lesson for the report: a random validation split is not a reliable proxy for temporal generalization.

## 2026-07-31 — App redesign: structured, prioritized, on-brand

[`src/app.py`](../../src/app.py) restyled + restructured (custom theme in `.streamlit/config.toml`).
Direction: "atelier/dispatch board" — warm paper background, dark graphite chrome, industrial
condensed type (Archivo Expanded), monospace for figures/codes (JetBrains Mono), safety-orange
accent — grounded in the auto-parts/workshop domain rather than a generic dashboard look.

**Structural change (the "use Top" ask):** the queue is no longer one flat table. Each vendor now sees:

1. **Priorité du jour** — the #1 client as a large featured "ticket" card (name, TND, reason, one click to open).
2. **Prochaines opportunités** — ranks #2–4 as a 3-card row.
3. **File complète** — everyone else (rank 5+) in a compact table with a priority *progress bar*, not just a number.
4. A reliable "Ouvrir un dossier client" selector always available to jump to any client directly.

Search collapses this to a single filtered table (no "Top" framing on an ad-hoc filter — would be misleading).

**Engineering note on verification:** no browser/screenshot tool was available this session, so the design
was built only on mechanisms provable without one — Streamlit's native `config.toml` theming, documented
stable `data-testid` selectors, and hand-written HTML/CSS. The tempting `st.dataframe(on_select=...)`
row-click pattern was avoided because it can't be verified end-to-end here; opening a client's file uses
tested buttons + a selectbox instead. Verified via: `AppTest` (7 scenarios — load, clicking a "Top" card
correctly switches the open dossier, feedback saves against the *right* client, vendor switch, empty-search
state, smallest-vendor edge case with 1 client) and a live headless server boot (HTTP 200, clean logs).
Two bugs were caught and fixed by these tests: the original test itself mis-indexed a widget (fixed to
label-based lookup) and an unguarded empty-view crash from an earlier pass.

**Bug found by the user after trying it live: invisible text.** Root cause: `secondaryBackgroundColor`
in `.streamlit/config.toml` was set to a dark graphite for sidebar chrome, but Streamlit reuses that same
value for the *internal* background of every selectbox/text input app-wide — while their text stays the
theme's dark `textColor` → dark-on-dark, unreadable. The sidebar CSS made it worse with a wildcard
`color: ... !important` on `*`, which also forced widget-internal text light against those same
(still light-boxed) widgets → light-on-light there too. Fixed: `secondaryBackgroundColor` set back to a
light tone (safe everywhere), and the sidebar override narrowed to a plain, non-`!important`, non-wildcard
rule on the container only — native widgets compute their own text color at render time and don't blindly
inherit it, so they stay readable; only plain text (labels, captions, badges) picks up the light color.
Lesson: this is exactly the class of bug the "no browser this session" caveat had flagged as a risk.

### Next

- Distribute the fiches / demo the app; collect feedback (first real labels).
- Confirm with the company: May 2026 completeness, `***` = closed accounts, ownership rule, priority weights.
- When transactional data (dates, quantities, statuses) arrives: 30-day horizon model (plan.md Phase 3–4).

## 2026-08-02 — Page de validation : les prédictions étaient-elles vraies ?

Nouvelle page Streamlit ([`src/pages/1_Validation.py`](../../src/pages/1_Validation.py)) qui rend
vérifiable, et non plus seulement affirmable, le résultat du modèle. L'app devient multipage :
page 1 = file d'appels (vendeurs), page 2 = validation (technique). Design :
[`docs/design/specs/2026-08-02-model-validation-interface-design.md`](../design/specs/2026-08-02-model-validation-interface-design.md).

**Architecture.** Tout le calcul de métriques vit dans [`src/validation.py`](../../src/validation.py),
sans aucun import Streamlit — donc testable en pytest sans navigateur. La page n'est qu'une couche
d'affichage. [`src/ui_theme.py`](../../src/ui_theme.py) porte le CSS partagé (`app.py` s'exécute au niveau
module : l'importer lancerait toute l'app vendeur en effet de bord).

**Prérequis livré.** [`train_model.py`](../../src/train_model.py) écrit désormais
`data/pair_scores_train.csv` et `reports/model_metrics.json` — sans eux aucune comparaison
train/test n'était possible. Le ré-entraînement reproduit `pair_scores_2026.csv` **au md5 près**
(`37597531d78fac9ad4b79e16e9e836ee`), vérifié avant/après : les chiffres de la page sont stables.

### Ce que la page montre

| § | Contenu | Chiffre clé |
| --- | --- | --- |
| 1 · Protocole | AUC train / validation interne / test | 0.815 / 0.798 / **0.762** — écart train−test **+0.053** |
| 2 · Agrégat | AUC comparés, lift par décile, **calibration**, matrice de confusion à seuil réglable | décile 10 : 53.0 % vs base 18.4 % |
| 3 · Recommandations | Précision@N vs score manuel, argent réalisé | P@5 : **60.6 %** (modèle) vs 40.9 % (manuel) |
| 4 · Inspection | vendeur → client → article : probabilité, prédiction, **réalité ✓/✗** | filtre « erreurs les plus confiantes » |

### Calibration — mesurée pour la première fois

Le modèle est **systématiquement trop confiant, et l'écart croît avec la confiance** : −0.2 % dans
la tranche 0–10 %, **+10.7 %** dans la tranche 70–80 %. Mais c'est un *majorant* de l'erreur : les
labels n'observent que janvier–mai, donc une paire rachetée en septembre est comptée comme un échec.
Une partie de l'écart vient de l'angle mort juin–décembre, pas du modèle.

Conséquence directe : cela **valide rétroactivement** le choix d'afficher « Élevée / Moyenne /
Faible » dans la file d'appels plutôt que des pourcentages bruts (project.md §9). Annoncer « 84 % »
là où le taux mesuré est 74 % aurait été faux ; « Élevée » reste honnête dans les deux lectures.

### Rejeu des recommandations — portée déclarée dans l'UI

Un rejeu littéral de la file est impossible ([`backtest.py`](../../src/backtest.py) le documentait déjà :
la règle « lapsed » demande deux années antérieures, 2023 n'existe pas). Ce qui est rejoué : le
classement par valeur attendue (p × CA), la restriction aux paires achetées en 2024 **et** 2025, le
top-3 par client → 3 056 paires, 1 099 clients, **48.8 % rachetées** (base 18.4 %, ×2.65). Valeur
attendue annoncée, environ 79 % a effectivement été réalisée. Ce qui n'est **pas** rejoué (filtre
« lapsed », boost de statut) est écrit dans un encadré visible sur la page, pas seulement ici.

### Tests

Suite pytest créée : **24 tests** (`py -m pytest tests/`). 12 unitaires sur `validation.py` avec des
frames synthétiques (aucune dépendance à l'extrait de 51 Mo), 5 AppTest sur la page de validation,
7 AppTest sur `app.py`. Ces 7 derniers étaient cités dans ce journal comme preuve de vérification
mais n'existaient que dans un scratchpad de session : ils sont maintenant dans `tests/`, donc
reproductibles depuis un clone. `use_container_width` (déprécié depuis fin 2025) remplacé par
`width="stretch"` sur la nouvelle page.

### Hors périmètre, volontairement

- **Suivi prospectif** des recommandations en cours : demande les retours vendeurs
  (`data/feedback.csv` n'existe pas encore). La page est structurée pour l'accueillir plus tard.
- **Tuning du modèle.** Avec une seule fenêtre de test honnête, une page qui affiche l'AUC de test
  en direct est une invitation à optimiser dessus. L'interdiction est écrite dans le spec.

## 2026-08-02 — Tableau de bord React + API FastAPI

Les trois livrables du projet (fiches Excel, file d'appels Streamlit, page de validation) étaient
dispersés, et rien ne présentait le jeu de données lui-même — là où une grande partie du travail
d'analyse est passée. Nouveau tableau de bord React à quatre vues, servi par une API FastAPI.
Design : [`docs/design/specs/2026-08-02-react-dashboard-design.md`](../design/specs/2026-08-02-react-dashboard-design.md).

### Pourquoi une API plutôt qu'un export statique

L'extrait contient des **personnes nommées** (« Mr … ») et leur chiffre d'affaires ; `data/` est
gitignoré exactement pour ça. Un export JSON statique aurait déposé ces noms dans `web/`, puis dans
tout `dist/` construit — trivialement copiable et facile à committer par accident. L'API lit `data/`
à l'exécution, écoute sur `127.0.0.1` uniquement, et ne renvoie que des agrégats ou des tranches
bornées (`pair_scores_2026.csv` fait 15 Mo et ne traverse jamais le réseau en entier).

**Vérifié, pas supposé** : un scan des 744 000 caractères de `web/` (hors `node_modules`) contre les
noms de `clients.csv` et les désignations de `articles.csv` remonte **0 correspondance**.

### Architecture

```text
api/main.py + deps.py + routes/{overview,queue,validation,dataset}.py
src/dataset_profile.py     nouveau — profilage, pandas pur, testable
web/                       React 18 + Vite 5 + TypeScript 5 + Recharts
```

`src/validation.py` est **importé tel quel** par l'API : calibration, lift, précision@N et rejeu de
file ne sont pas réimplémentés. Conséquence utile : la page React et la page Streamlit ne peuvent
pas afficher des chiffres différents, elles lisent le même module testé.

Vite proxie `/api` vers `127.0.0.1:8000`, donc le frontend n'utilise que des chemins relatifs : pas
de CORS, pas d'hôte à configurer, rien à fuiter dans un build.

### Un bug que les tests masquaient

Les modules de routes importent `validation` / `dataset_profile` au niveau module, avant que
`deps.py` ne modifie `sys.path`. La suite passait quand même parce que `tests/conftest.py` avait
déjà ajouté `src/` au path — **mais le vrai serveur ne démarrait pas**. Corrigé en déplaçant la
manipulation de `sys.path` dans `api/__init__.py`, et couvert par un test qui importe l'app dans un
sous-processus, sans l'aide de conftest. Leçon : une suite verte ne prouve pas que le programme
démarre ; il faut aussi lancer le binaire.

### Tests

**45 tests** (`py -m pytest tests/`) : 16 API (chaque endpoint affirme une vraie valeur, pas
seulement un 200), 5 profil de données, 12 `validation.py`, 7 AppTest `app.py`, 5 AppTest page de
validation Streamlit. Typecheck TypeScript strict et build de production propres.
Bout en bout vérifié : les deux serveurs démarrent, les quatre routes React répondent 200 et le
proxy `/api` transmet correctement.

### Non vérifié : l'apparence

Les tests prouvent que les données sont justes ; ils ne prouvent pas que les pages sont lisibles.
Le CSS est écrit à la main (là où Streamlit fournissait des composants thémés), donc la surface de
bugs visuels est plus grande — c'est exactement la classe de bug qui a déjà frappé deux fois ici
(texte invisible, régression de contraste). **Une relecture visuelle manuelle reste à faire.**

## 2026-08-03 — Refonte visuelle : « Le carnet d'appels »

Le tableau de bord ressemblait à une console d'administration. Cause identifiable : il
traitait le vendeur comme quelqu'un qui **consulte des enregistrements** (tableaux denses,
micro-libellés en capitales, bandeau de KPI, chrome sombre). Or un vendeur ne consulte pas :
il s'apprête à **téléphoner à quelqu'un**. Toute la refonte part de ce verbe.

**Direction.** Palette chaude — pin `#0F5C4A`, terre `#E4572E`, beurre `#F2C14E` sur crème
`#FDFBF7`. Typo : Bricolage Grotesque (titres, du caractère), Plus Jakarta Sans (texte, très
lisible en français), DM Mono réservé aux codes pièces, là où l'alignement sert vraiment.
Barre de navigation claire à la place de la barre latérale sombre.

**Élément signature — « Ce qu'il faut dire ».** Chaque fiche rédige la phrase d'ouverture de
l'appel à partir de l'historique réel du client. Pas une statistique à interpréter : une
phrase à dire à voix haute. Chaque membre de phrase provient d'un champ réellement renvoyé
par l'API ; si l'information manque, la proposition est supprimée plutôt que devinée
(`web/src/callScript.ts`).

**Périmètre.** Refonte structurelle sur les écrans vendeurs (Aujourd'hui, Mes clients).
Les pages Fiabilité et Les données gardent leur densité — leur public est le jury, et un
tableau y est la bonne forme. Les rendre « conviviales » les aurait dégradées. Elles
reçoivent la nouvelle palette et la nouvelle typo, ainsi qu'une réécriture des textes en
langage courant.

### Une vraie fuite de données, trouvée et corrigée

Le scan de `web/` a détecté **un nom de client réel** — utilisé par erreur comme exemple dans
une docstring de `callScript.ts`. `data/` est gitignoré, `web/` ne l'est pas : ce nom serait
parti sur GitHub. Corrigé, et surtout **transformé en test permanent**
(`tests/test_no_data_leak.py`) : deux tests balaient tout `web/` (source et build) contre les
noms de `clients.csv` et les désignations de `articles.csv`. La vérification manuelle est
devenue une garantie.

### Un bug de langue révélé par les vraies données

Le script d'appel produisait « Il vous manque **du** ANTIGEL 1.5L ROUGE » — les désignations
sont des références commerciales en capitales, pas des noms communs, et les faire suivre un
article partitif donne du français cassé. Corrigé : les références sont désormais toujours
citées « … » et jamais gouvernées par un article. Ce bug n'était visible qu'en interrogeant
l'API en conditions réelles, pas dans les tests.

**47 tests** au vert. Typecheck TypeScript strict et build de production propres. Les deux
serveurs démarrent, `/docs` et `/openapi.json` répondent, le proxy `/api` fonctionne.
**L'apparence reste non vérifiée visuellement** — aucun rendu n'a été regardé.

## 2026-08-03 — Probabilités affichées, et une calibration écartée pour cause de preuve

Question posée : « le montant affiché est-il un nombre prédit ? » Non — et la formulation
« TND attendus » le laissait croire.

### Ce que vaut vraiment le montant affiché

`priority_score = max(valeur_attendue, manque_a_gagner) × boost`, et selon le client le
gagnant du `max` n'est pas le même :

| Type de client | Nombre | Le montant vient de |
| --- | --- | --- |
| ACTIVE | 735 | **le modèle** (`valeur_attendue` = Σ p × CA) |
| DECLINING | 343 | **l'historique** (baisse de CA × 1.1) |
| INACTIVE_2026 | 126 | **l'historique** (baisse de CA × 1.2) |

`manque_a_gagner` n'est calculé que pour les clients signalés, donc il vaut 0 ailleurs et
le modèle gagne par défaut. Concrètement : **pour 39 % de la file, le montant affiché ne
contient aucune sortie de modèle.** Exemple du client n°1 (0508) : le modèle estimait
2 863 TND, l'historique 42 174 TND — c'est ce dernier × 1.2 = 50 608 TND qui s'affichait
sous le mot « attendus ». Libellé corrigé en **« TND en jeu »** (Streamlit et React), avec
une note explicite : ce n'est pas une prévision de vente.

### Calibration : tentative, mesure, abandon

Objectif : afficher de vrais pourcentages au lieu des libellés Élevée/Moyenne/Faible.
Méthode correcte appliquée — courbe isotone ajustée sur le **split de validation**
(jamais sur la fenêtre de test, qui reste intacte).

| Écart de calibration (test 2025→2026) | Avant | Après |
| --- | --- | --- |
| Maximum sur une tranche | 10.7 % | **13.9 %** |
| Moyen pondéré | 2.2 % | **2.5 %** |

**La correction dégrade le résultat.** Mécanisme identifié : sur le split de validation le
modèle est bien calibré, voire sous-confiant — ce split partage la *même année* que
l'entraînement, donc ses prédictions confiantes se réalisent vraiment. La courbe apprend
donc à **remonter** les probabilités. D'une année sur l'autre elles ne se réalisent plus à
ce rythme, et remonter creuse l'écart : la tranche haute passe de 375 à 1 525 paires dont
seulement 78.9 % rachètent.

**Cause racine : il n'existe aucun jeu de données représentant une année neuve.** Calibrer
honnêtement demanderait des snapshots 2023→2024, et 2023 n'est pas dans l'extrait. C'est
la **même année manquante** qui rend impossible le rejeu littéral de la file
([`backtest.py`](../../src/backtest.py)) — un seul trou dans les données, deux limites distinctes.

C'est le deuxième résultat, obtenu par une voie indépendante, qui confirme la leçon de
l'expérience à 4000 itérations : *un découpage aléatoire à l'intérieur d'une année n'est
pas un substitut à la généralisation temporelle*. La prudence initiale de project.md §9
(libellés plutôt que pourcentages) n'est plus une précaution : c'est une décision étayée.

Le calcul reste dans [`train_model.py`](../../src/train_model.py) et la courbe est stockée dans
`model_metrics.json` pour que la mesure soit reproductible ;
[`contact_queue.py`](../../src/contact_queue.py) ne l'applique délibérément pas.

### Ce qui est affiché maintenant

`contact_queue.py` exporte `proba_1/2/3` (probabilité brute par article, couverture 87 %).
La fiche client Streamlit affiche **« 90 % de chance de rachat »** sous chaque article
recommandé, avec la mention qu'il s'agit d'une estimation optimiste et le renvoi au rapport
de calibration. 51 tests au vert, dont un qui vérifie que le pourcentage et son avertissement
sont bien présents ensemble.

## 2026-08-04 — L'API livrable à un développeur front

Objectif : qu'un développeur front puisse consommer l'API **sans jamais recevoir `data/`**.
Trois changements, tous à défaut inchangé.

### `DATA_ROOT` — un jeu de données interchangeable

`DATA_ROOT` non défini → `data/` + `reports/`, exactement comme avant. Défini → il pointe
ailleurs. La variable est lue dans [`api/deps.py`](api/deps.py) **et**
[`src/validation.py`](../../src/validation.py) : ce second point n'était pas dans le plan initial et
aurait été un vrai défaut. `validation.py` est importé par l'API *et* par la page Streamlit ;
sans l'y ajouter, les quatre endpoints `/api/validation/*` auraient continué à servir les
vraies données malgré `DATA_ROOT=fixtures` — une API à moitié anonymisée, le pire des cas.

### `fixtures/` — copie anonymisée, commitée

[`tools/make_fixtures.py`](../../tools/make_fixtures.py) produit 10 vendeurs, 80 clients, ~65 K lignes
de faits, 3,7 Mo. Identifiants et noms remplacés ; **les montants sont conservés** : une fois
qu'un montant n'est plus attribuable à un client nommé il n'est plus confidentiel, et le garder
préserve la cohérence interne (les textes de raison citent des chiffres qui doivent correspondre
aux lignes qu'ils décrivent). `reports/model_metrics.json` est copié tel quel — agrégats
uniquement, déjà versionné.

**Un piège trouvé en vérifiant** : le motif `data/` de `.gitignore` s'applique à *tous* les
niveaux, donc `fixtures/data/` était ignoré — les fixtures n'auraient jamais été commitées et
le développeur ne les aurait jamais reçues. Corrigé en ancrant le motif : `/data/`. Vérifié
dans les deux sens avec `git check-ignore` (fixtures committables, `data/` et l'Excel toujours
exclus).

### CORS localhost

Le React du dépôt passe par le proxy `/api` de Vite et n'en a jamais eu besoin ; un front sur
son propre serveur de dev serait bloqué par le navigateur. Autorisé pour
`http://localhost:<port>` et `127.0.0.1:<port>` uniquement — pas de `*`, qui serait faux le
jour d'un déploiement.

### Vérification

Référence **67 tests** avant modification, **70** après (les 3 nouveaux couvrent
l'anonymisation). Le test de fuite existant est étendu : il balaie désormais `fixtures/` en plus
de `web/`, et un troisième test vérifie que tous les identifiants suivent le schéma synthétique
— un nom peut être remplacé alors qu'un `client_id` réel subsiste et suffit à relier les deux
jeux.

Bout en bout : les 11 endpoints répondent 200 sur les fixtures (10 GET + le POST), le CORS
renvoie bien l'origine, et **le POST `/api/feedback` écrit dans `fixtures/data/feedback.csv`,
`data/feedback.csv` restant intact**. Enfin les deux applications ont tourné **simultanément** —
Streamlit sur les vraies données (HTTP 200, log propre), l'API sur les fixtures — ce qui est la
preuve directe que la modification ne les couple pas.

### Reste à faire

- **Aucune route ne déclare de `response_model`** : dans `/openapi.json` les réponses
  apparaissent comme des objets vides. Le développeur voit les URL mais aucun nom de champ et
  ne peut pas générer de client typé. C'est la prochaine amélioration utile.
- **Aucune authentification, aucun cloisonnement par vendeur** : n'importe quel appelant peut
  lire la file de n'importe quel vendeur, ce que `project.md` §12 interdit. Sans conséquence en
  local, bloquant dès que l'API est partagée.
- `/api/validation/protocol` renvoie les métriques réelles (231 434 paires) alors que les
  fixtures en contiennent 22 493 : cohérent puisque `model_metrics.json` est copié tel quel,
  mais l'écart peut surprendre.

## 2026-08-03 — Textes simplifiés, montant compréhensible

### Comment les articles sont classés (question posée)

Par **valeur attendue = probabilité × CA de l'an dernier** — donc les deux, mais l'argent
pèse davantage. Mesuré sur la file : l'article n°1 est le plus **probable** dans 60.6 % des
cas, et le plus **cher** dans 74.0 % des cas. Corrélation entre probabilité et CA : 0.23,
donc les deux critères sont largement indépendants. Exemple : une huile à 15 590 TND / 83.7 %
passe devant un antigel à 1 896 TND / 84.6 %. C'est voulu — « n°1 » signifie *le plus
d'argent en jeu*, pas *le plus probable* — et c'est maintenant écrit dans l'interface.

### Le montant affiché : composite remplacé par un chiffre réel

`priority_score` = `max(estimation modèle, baisse de CA) × boost` ne correspond à rien de
physique : rien dans le monde ne vaut 50 608. Il continue de piloter **l'ordre** et la
synthèse direction, mais les écrans vendeurs affichent désormais `montant_cle` +
`montant_texte`, un chiffre que le vendeur peut répéter au téléphone :

| Situation | Affiché |
| --- | --- |
| Inactif | « 42 174 TND achetés l'an dernier, rien cette année » |
| En baisse | « 31 978 TND de baisse vs l'an dernier » |
| Actif | « X TND d'articles habituels non rachetés » |

Couverture : 100 % des clients ont un montant non nul. L'ordre de la file est inchangé.

### Textes

Réécriture complète des libellés Streamlit en langage courant : « Priorité » → « En jeu »
puis le montant concret, « Feedback » → « Après l'appel », « Historique CA » → « Ses achats,
mois par mois », « Articles en attente » → « Articles délaissés ». Raison d'article
raccourcie : « Acheté 4 mois en 2024, 2 mois en 2025 — 1 109 TND. Rien en 2026. »

Le libellé de confiance (« Confiance rachat: Élevée ») est retiré du texte de raison : il
faisait doublon avec le pourcentage affiché juste au-dessus, et « 51 % » à côté de
« Élevée » se lit comme une contradiction.

### Régression évitée

Retirer ce libellé aurait silencieusement vidé les fiches Excel de toute indication de
confiance — `vendor_sheets.py` lit les mêmes colonnes `reason_1..3`. Le pourcentage y est
donc désormais préfixé au texte de raison, avec la même mention « estimation optimiste ».
Fiches régénérées (38 fichiers). 52 tests au vert.
