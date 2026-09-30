# Le système de recommandation vendeurs — de l'Excel à l'IA

**Document de présentation.** Chaque `---` marque une rupture de diapositive.
Toutes les affirmations sont vérifiables dans le code (`src/`) ou dans les rapports générés (`reports/`).

> **Version v4 — fondation transactionnelle NAV.** Chiffres relus le 2026-08-25 contre
> `reports/rolling_metrics.json`, `reports/prospect_metrics.json`, `reports/reconciliation.md`
> et `reports/live_window.json`.
>
> Le projet a **changé de fondation de données en cours de route** : d'un export Excel de CA
> mensuel (15 mois troués, sans dates) vers le registre comptable NAV (30 mois continus, au
> jour près). Les comparaisons « v1 → v4 » du document sont **intentionnelles** — elles
> montrent ce qu'une meilleure extraction a rendu inutile. Voir **A7**, **B1** et **B10**.

> Conversion : `marp docs/presentation-ia.md -o presentation.pdf`
> ou `pandoc -t revealjs -s docs/presentation-ia.md -o presentation.html`

---

## Plan

| Partie | Question à laquelle elle répond | Public |
| --- | --- | --- |
| **A** | Qu'est-ce que l'IA fait, en français simple ? | Tout le monde |
| **B** | D'où viennent les données et qui décide de leur forme ? | Métier + technique |
| **C** | Quels algorithmes, quelles mathématiques ? | Technique |
| **D** | Quels graphiques servent à décider ? | Technique + jury |
| **E** | Comment se déroule un entraînement, concrètement ? | Technique |
| **F** | Quelles sont les limites — et lesquelles sont tombées ? | Tout le monde |

---
---

# PARTIE A — L'IA en langage simple

---

## A1. Ce que « IA » veut dire dans ce projet

Ce n'est **pas** ChatGPT. Rien ici ne comprend ni ne parle.

C'est de l'**apprentissage supervisé** (*supervised learning*), et ça tient en une phrase :

> On montre à la machine des milliers de situations passées **dont on connaît déjà la fin**,
> et elle cherche toute seule la formule qui relie le début à la fin.

Trois familles existent en apprentissage automatique ; nous sommes dans la première :

| Famille | Principe | Notre cas |
| --- | --- | --- |
| **Supervisé** | Apprendre d'exemples étiquetés | ✅ Les deux modèles |
| Non supervisé | Trouver des groupes sans étiquette | Partiellement (similarité des paniers) |
| Par renforcement | Apprendre par essai/erreur récompensé | ❌ Impossible : pas de boucle de retour |

---

## A2. L'analogie du nouveau vendeur

> Un nouveau vendeur arrive. Vous ne lui expliquez **aucune règle**.
>
> Vous lui donnez 231 000 fiches de 2024 : *« ce client achetait cette pièce 3 mois sur 5, pour 4 000 TND »*.
>
> Puis vous retournez chaque fiche : au dos est écrit s'il a racheté en 2025.
>
> Il lit les 231 000 fiches et se fabrique son intuition.

**L'IA, c'est exactement ce vendeur — sauf qu'il ne se fatigue pas et qu'il lit tout.**

---

## A3. Le point de départ : transformer le passé en exercices corrigés

Une machine ne peut apprendre que si on lui donne des questions **dont la réponse est déjà connue**.
C'est le rôle de `build_snapshot()` — `src/snapshots.py:88`.

| L'énoncé — ce qu'on savait en 2024 | La réponse — ce qui s'est passé en 2025 |
| --- | --- |
| Client A / Pièce B : 3 mois d'achat, 4 000 TND, dernier achat en mai… | **1** = il a racheté |
| Client C / Pièce D : 1 mois d'achat, 200 TND, achat en janvier… | **0** = il n'a pas racheté |

Le vocabulaire à employer devant un jury :

- **Features** (caractéristiques) = l'énoncé, les 23 colonnes de gauche.
- **Label** (étiquette) = la réponse, la colonne de droite : 0 ou 1.
- **Instance** = une ligne, ici un couple (client, article).

**2 051 182 lignes** pour la seule fenêtre de test, dont **11,4 % portent un « 1 »**.

---

## A4. La règle qui tient tout le projet : le décalage temporel

Le registre NAV étant journalier, la coupure n'est plus une **année** mais une **date** `T`,
choisie 16 fois de suite — c'est l'**instantané** (*snapshot*) :

```
   [ T − 12 mois  ──────► T )        [ T  ──────► T + 3 mois )
   ┌──────────────────────┐        ┌──────────────────────┐
   │      FEATURES        │ ─────► │        LABEL         │
   │     (l'énoncé)       │        │    (la réponse)      │
   └──────────────────────┘        └──────────────────────┘
    strictement AVANT T              strictement À PARTIR de T
        Ce que le modèle voit          Ce qu'il doit deviner
```

`src/snapshots.py:6-7` — 12 mois de recul, 3 mois d'horizon, **16 instantanés**
(T = 2025-01 … 2026-04). Les features viennent **uniquement** d'avant `T`, le label
**uniquement** d'après.

> **Pourquoi 16 dates plutôt qu'une seule ?** Un modèle validé sur un seul mois peut
> n'avoir appris qu'une saison. Répéter la découpe donne 16 mesures au lieu d'une, et
> c'est leur **stabilité** qui prouve quelque chose (voir A13).

**Si une seule information de 2025 se glissait dans l'énoncé de 2024**, le modèle « devinerait »
parfaitement en test et s'effondrerait en production. Cela s'appelle une **fuite de données**
(*data leakage*) : c'est l'erreur n°1 des projets d'apprentissage automatique.

---

## A5. Une fuite plus subtile, verrouillée dans le code

`src/snapshots.py:25-27` interdit un second piège, bien plus difficile à voir :
**la contamination entre les découpes**.

Le label d'un instantané court **3 mois vers l'avant**. Si l'entraînement s'arrête en
juin et que la validation commence en juillet, le label de juin regarde juillet, août et
septembre — **que la validation est censée n'avoir jamais vus.**

D'où **deux embargos**, pas un seul :

```
 TRAIN            EMBARGO      VAL          EMBARGO      TEST
 2025-01..06      07..08       09..10       11..12       2026-01..04
 (6 instantanés)  (jetés)      (2)          (jetés)      (4)
```

Les 4 instantanés d'embargo sont **calculés puis jetés**. C'est le coût assumé de la
propreté : on paie 4 mois de données pour garantir qu'aucune fenêtre de label ne
chevauche la découpe suivante.

---

## A6. Ce que le modèle « voit » — les 23 features traduites

Le modèle ne voit ni nom, ni pièce, ni client. Il voit **21 nombres + 2 catégories par ligne**
(`src/snapshots.py:58-67`).

**Sur le couple client × pièce — l'intensité ET le rythme de la relation**
`n_orders` nombre de commandes · `n_months` mois actifs · `qty_sum` quantités ·
`amount_sum` argent · `amount_per_order` panier moyen · `days_since_last` fraîcheur ·
`days_since_first` ancienneté · `median_interval` rythme habituel ·
`interval_cv` régularité de ce rythme · `overdue_ratio` **retard relatif** ·
`return_rate` taux de retour

**Sur le client — son profil global**
`client_amount` son poids · `client_n_articles` largeur du panier ·
`client_n_orders` fréquence · `client_active_months` présence ·
`client_days_since_last` **sa fraîcheur globale**

**Sur l'article — sa nature**
`article_n_clients` combien de clients l'achètent · `article_amount` son poids ·
`article_repurchase_share` consommable ou pièce à vie

**Relatives** : `share_of_client`, `share_of_article`
**Catégorielles** : `enseigne` (la marque), `vendor_code` (le vendeur)

> Six de ces features (`days_since_*`, `median_interval`, `interval_cv`, `overdue_ratio`,
> `client_days_since_last`) étaient **impossibles à calculer** avant le registre NAV.
> Voir A7 — c'est le changement le plus important du projet.

---

## A7. L'histoire à raconter : quand la donnée rend un bricolage inutile

**La version précédente du projet n'avait aucune date.** Les données étaient un agrégat
mensuel : « ce client a acheté pour tant, ce mois-là ». « Dernier achat il y a 12 jours »
était hors de portée.

Le contournement était un **substitut de fraîcheur** — la part du CA réalisée en fin de
fenêtre, en supposant qu'un client actif en avril-mai est encore vivant :

```python
# ANCIEN — un substitut, faute de mieux
late = pos[pos["mois"] >= 4].groupby("client_id")["ca"].sum()
cli["client_late_share"] = (late / cli["client_ca"]).clip(0, 1)
```

Le registre NAV a des **dates au jour près**. La fraîcheur se mesure directement, et le
rythme propre à chaque couple client×pièce devient calculable :

```python
# ACTUEL — src/snapshots.py:128, la vraie mesure
pair["overdue_ratio"] = pair["days_since_last"] / pair["median_interval"]
```

`overdue_ratio` répond à *« ce client a-t-il dépassé SON propre rythme ? »* — un garage qui
commande tous les 15 jours et qu'on n'a pas vu depuis 45 est en retard ; un autre qui
commande deux fois par an ne l'est pas. La même donnée brute, lue **relativement**.

**Le verdict des chiffres :** `client_late_share` pesait 14,3 (4ᵉ sur 20). Son remplaçant
`client_days_since_last` pèse **18,0 — 1ʳᵉ sur 23**, devant tout le reste.

> **La leçon, et elle vaut mieux que l'astuce :** le *feature engineering* compense la
> pauvreté de la donnée, il ne la remplace pas. Une bonne feature bricolée a été battue
> par la vraie mesure dès qu'on a pu l'obtenir. **Le meilleur gain de ce projet n'est pas
> venu d'un algorithme — il est venu d'une meilleure extraction.**

---

## A8. Comment CatBoost apprend — l'unité de base

Un **arbre de décision**, c'est une suite de questions oui/non :

```
Le client achète-t-il plus de 50 références ?
├── OUI → A-t-il acheté cette pièce 3 mois ou plus ?
│         ├── OUI → probabilité 71 %
│         └── NON → probabilité 34 %
└── NON → La pièce est-elle un consommable ?
          ├── OUI → probabilité 22 %
          └── NON → probabilité  6 %
```

`depth=6` : chaque arbre pose **6 questions imbriquées**, jamais plus.

---

## A9. Comment CatBoost apprend — le vrai mécanisme

Un seul arbre est bête. La force du *gradient boosting* est ailleurs :

1. **Arbre n°1** fait une prédiction grossière sur les 231 000 lignes.
2. On calcule **ses erreurs** : où a-t-il dit « il rachètera » alors que non ?
3. **Arbre n°2 n'apprend pas le problème — il apprend les erreurs de l'arbre n°1.**
4. Arbre n°3 apprend les erreurs restantes des deux premiers. Ainsi de suite.
5. Après **800 arbres**, on additionne tout le monde.

`learning_rate = 0.05` : chaque arbre ne corrige que **5 %** de l'erreur.
On avance à petits pas pour ne pas sur-réagir à un cas isolé.

> **La phrase pour un non-technicien :** « Ce n'est pas un expert. C'est une commission de
> 800 conseillers médiocres, où chacun ne s'occupe que des dossiers que les précédents ont ratés. »

---

## A10. Ce que « le modèle » est, physiquement

Le fichier `data/model_rolling.cbm` contient :

- **800 organigrammes** et leurs seuils de décision
- Les tables d'encodage des variables catégorielles

**Rien d'autre.** Pas de code, pas de raisonnement, pas de données clients — des nombres figés.

Le prédire, c'est parcourir 800 fois 6 questions et additionner 800 nombres.
Cela prend quelques microsecondes par client.

---

## A11. Le garde-fou : apprendre ≠ réciter par cœur

Le danger s'appelle le **surapprentissage** (*overfitting*) : l'élève qui apprend les corrigés
par cœur. 20/20 sur les exercices vus, 5/20 à l'examen.

Trois niveaux de protection — `src/train_rolling.py`, **découpés dans le temps**, jamais au hasard :

| Jeu | Instantanés | Rôle | Analogie scolaire |
| --- | --- | --- | --- |
| **Train** | 2025-01 … 06 (6) | Il apprend dessus | Les exercices d'entraînement |
| **Validation** | 2025-09 … 10 (2) | Dire *quand s'arrêter* (`early_stopping_rounds=200`) | Le contrôle blanc |
| **Test** | 2026-01 … 04 (4) | **Jamais touché.** Une seule fois, à la fin | L'examen final |

Entre chacun, **un embargo de 2 instantanés jeté** (voir A5).

> **Le point à défendre :** la version précédente coupait **au hasard** (80/20 sur la même
> année). Un tirage aléatoire met le même client des deux côtés et flatte le résultat.
> Ici la coupure est **chronologique** : le modèle est toujours jugé sur un futur qu'il
> n'a pas vu — la seule découpe qui ressemble à la production.

---

## A12. L'argument de sérieux à connaître par cœur

Deux choses à savoir défendre, et la seconde compte plus que la première.

**1. Le modèle a convergé — il ne s'est pas arrêté par manque de budget.**

| Paramètre | Valeur |
| --- | --- |
| Plafond d'itérations | 6 000 |
| Arrêt précoce | 200 rondes sans progrès |
| **Meilleure itération** | **1 345** |
| Écart train − test | **+0,037** |

L'arrêt précoce **s'est déclenché** : l'AUC de validation avait cessé de progresser bien
avant le plafond. Un modèle stoppé *au* plafond ne prouve rien — on ne sait pas s'il
aurait continué à s'améliorer. Ici, si.

**2. Le refus de comparer — l'argument le plus solide du dossier.**

L'ancien modèle affichait **AUC 0,762** sur la question annuelle janvier-mai. Le modèle
actuel affiche **0,774**. `reports/rolling_metrics.json` interdit explicitement de citer
cet écart (`not_comparable_to`) :

> Un horizon de 3 mois a un **taux de base plus faible** et pose un problème **plus
> difficile**. Un chiffre plus élevé sur une question différente n'est pas la preuve d'un
> meilleur modèle. La seule comparaison honnête serait de rejouer la question annuelle sur
> les nouvelles données — **ce qui n'a pas été fait.**

> **Ne jamais annoncer « +0,012 d'AUC ».** Deux nombres qui se ressemblent ne mesurent pas
> forcément la même chose. Savoir refuser une comparaison flatteuse est un résultat.

---

## A13. Le résultat, en langage humain

**L'AUC de 0,774 — la définition à donner :**

> Prenez au hasard une paire qui a vraiment racheté, et une qui n'a pas racheté.
> L'AUC est la probabilité que le modèle donne la note la plus haute à la bonne.
> **0,50 = pile ou face. 0,774 = juste dans 77 % des cas.**

**La comparaison qui compte** — contre ce que ferait un vendeur dans Excel, c'est-à-dire
trier par chiffre d'affaires passé :

| Méthode | Précision du top 30 par vendeur |
| --- | --- |
| Trier par CA passé | 31,9 % |
| **CatBoost** | **61,9 %** |

**Le modèle double la précision de la liste d'appel** à effort constant.

**Et la stabilité, qui vaut mieux que le chiffre global :**

| Instantané | 2026-01 | 2026-02 | 2026-03 | 2026-04 |
| --- | --- | --- | --- | --- |
| AUC | 0,775 | 0,773 | 0,774 | 0,775 |

> Quatre mois consécutifs à ±0,002. Un modèle qui ne marche qu'un mois n'est pas
> déployable ; celui-ci ne dépend pas du mois où on l'interroge.

---

## A14. Le chiffre à montrer à un décideur

Ce n'est pas l'AUC. C'est le **décile** :

| Décile (10 = probabilité la plus haute) | Taux de rachat réel | Lift |
| --- | --- | --- |
| **10** | **39,3 %** | **×3,45** |
| 9 | 20,5 % | ×1,81 |
| 8 | 14,6 % | ×1,29 |
| 7 | 11,2 % | ×0,98 |
| 6 | 8,5 % | ×0,75 |
| 5 | 6,7 % | ×0,58 |
| … | … | … |
| **1** | **1,2 %** | **×0,10** |

> Sur 100 paires au hasard, **11 rachètent** dans les 3 mois. Sur les 100 du **top 10 %,
> 39 rachètent.** Le vendeur travaille sur une liste **3,45 fois plus dense**, pour le
> même temps de téléphone.

Et le décile 1 à 1,2 % est aussi utile : **le modèle sait dire qui ne pas appeler.**
Un décile bas 10× sous le taux de base, c'est du temps de vendeur rendu.

---

## A15. Ce que le modèle a appris tout seul

| Feature | Importance | Ce que ça dit |
| --- | --- | --- |
| `client_days_since_last` | **18,0** | **Depuis quand on ne l'a pas vu** |
| `client_n_orders` | 9,8 | À quelle fréquence il commande |
| `article_n_clients` | 9,3 | La pièce est-elle courante |
| `enseigne` | 9,0 | La marque |
| `client_n_articles` | 7,6 | La largeur de son panier |
| `client_amount` | 6,5 | Son poids |
| `article_repurchase_share` | 5,9 | Consommable ou pièce à vie |

> **Conclusion métier, née des données et non programmée :**
> la première question n'est pas *« quelle pièce ? »* mais **« ce client est-il encore
> vivant, et est-il en retard ? »**. La récence et la fréquence dominent tout le reste.

**Le renversement à souligner :** dans la version précédente, les features de **volume**
(`client_n_articles`, `client_ca`) menaient. Depuis que la **récence** est mesurable, elle
prend la tête. Ce n'est pas le modèle qui a changé d'avis — c'est la donnée qui a enfin
permis de poser la bonne question.

---

## A16. Le deuxième modèle : même logique, question opposée

`src/prospect_model.py` répond à l'autre question du vendeur :
**« qui pourrait acheter cette pièce alors qu'il ne l'a jamais prise ? »**

| | Modèle 1 — rachat | Modèle 2 — prospection |
| --- | --- | --- |
| Population | Clients qui ont **déjà** la pièce | Clients qui ne l'ont **jamais** eue |
| Label | A racheté sous 3 mois | A adopté sur le semestre suivant |
| Découpe | Instantanés mensuels glissants | Fenêtres semestrielles (H1/H2) |
| Taux de base | 11,4 % | ~0,2 % avant échantillonnage |
| Difficulté | Modérée | Élevée — on part de zéro |
| Métrique | AUC, décile | precision@5 |

---

## A17. Modèle 2 — la décision qui pouvait tout invalider

`src/prospect_model.py:46-52` : les négatifs sont tirés **à l'intérieur de la même pièce**.

```
❌ Tirer au hasard dans tout le catalogue
   → le modèle apprend à reconnaître des PIÈCES plausibles
   → l'AUC explose, le classement ne sert à rien

✅ Tirer parmi les candidats de CETTE pièce
   → le modèle apprend à classer des CLIENTS
   → c'est exactement la question que pose l'écran
```

**L'entraînement doit poser la même question que l'interface.**
Sinon on optimise une métrique flatteuse et un produit inutile.

---

## A18. Modèle 2 — les résultats

Features 2025-H2 → labels 2026-H1, **jamais vus**, sur **1 649 articles** ayant réellement
gagné un acheteur. Un embargo d'un semestre sépare l'entraînement du test.

| Classement | precision@5 | recall@5 |
| --- | --- | --- |
| **Modèle** | **12,98 %** | **8,10 %** |
| Heuristique (signature) | 9,17 % | 3,59 % |
| Client le plus large | 7,87 % | 3,96 % |
| Client le plus gros | 4,46 % | 2,91 % |
| Hasard | 0,55 % | 0,19 % |

**AUC : 0,875** — et **près de 24× le hasard** sur la précision.

Les trois étalons intermédiaires comptent autant que le hasard : ils prouvent que le
modèle ne se contente pas de **proposer les gros clients**. « Le client le plus large »
atteint déjà 7,87 % — le modèle le bat quand même, donc il a appris autre chose que la taille.

Les deux features dominantes : `client_n_articles` (22,4) et **`cosine_basket_sim` (16,0)**,
la ressemblance entre le panier du client et celui des acheteurs actuels. Traduction pour
le vendeur : *« les garages qui te ressemblent prennent cette pièce. »*

---
---

# PARTIE B — L'ingestion des données

---

## B1. La source : le registre comptable NAV

Le projet a **changé de fondation en cours de route**. C'est le fait le plus important de
cette partie.

| | v1 — l'extraction initiale | **v2 — le registre NAV** |
| --- | --- | --- |
| Fichier(s) | `CA client vendeur article Janv-mai 24-25-26.xlsx` (51 Mo) | `vente 01-01-24 -- 28-05-2025.xlsx` (121 Mo) + `vente 29-05-2025 -- 30-06-2026.xlsx` (96 Mo) |
| Nature | Agrégat mensuel de CA | **Journal de transactions** |
| Grain | (vendeur, client, article, **mois**) | (**document, ligne**) — une écriture comptable |
| Dates | ❌ aucune | ✅ **au jour près** |
| Quantités | ❌ | ✅ `Quantité` |
| Coût / marge | ❌ | ✅ `Coût total (réel)`, `Discount Amount` |
| Couverture | Jan–Mai de 2024, 2025, 2026 (**15 mois troués**) | **2024-01-02 → 2026-06-30, 30 mois continus** |

Feuille `Écritures comptables article`, 17 colonnes. Deux fichiers parce que le premier
bute sur le **plafond de 1 048 576 lignes d'Excel** — la coupure (28/05 → 29/05) est une
limite technique, pas une frontière métier. Ils se concatènent directement.

S'y ajoutent deux référentiels : `deatils articles nav.xlsx` et `deatils clients nav.xlsx`.

> **Ce que ça débloque :** « dernier achat il y a 12 jours », le rythme propre à chaque
> client, et des labels qui ne sont plus des bornes inférieures. Voir A7 et F1.

---

## B2. Le problème de format : « pivoté » vs « long »

**L'ancienne extraction était pivotée (*wide*)** — une colonne par mois :

| vendeur | client | article | jan-24 | fév-24 | mars-24 | … | avr-26 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V01 | C205 | A9931 | 1 200 | – | 830 | … | 410 |

Il fallait la **dépivoter** (*unpivot* / *melt*) pour obtenir une ligne par observation.
Ajouter un mois obligeait sinon à réécrire le code ; en format long, c'est une ligne de plus.

**Le registre NAV arrive déjà en format long** — c'est la nature d'un journal : une ligne
= une écriture. Le dépivotage disparaît, et avec lui toute une classe de bugs.

| date | doc_type | doc_no | client_id | article_id | qty | amount |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-01-02 | Facture vente | F-10231 | C205 | A9931 | 4 | 1 200,000 |

**La leçon générale reste vraie et vaut d'être dite :** le format d'arrivée d'une donnée
détermine la difficulté de tout ce qui suit. Le travail n'a pas disparu parce qu'on a mieux
codé — il a disparu parce qu'on a **demandé une meilleure extraction**.

---

## B3. Qui décide de quel format ? — la chaîne de responsabilité

| Étape | Format | **Qui décide** | Négociable ? |
| --- | --- | --- | --- |
| Export ERP | Excel, feuille `Écritures comptables article` | **L'entreprise / l'éditeur NAV** | Non — c'est ce qui existe |
| Périmètre livré | 30 mois continus, transactions datées | **L'entreprise** (choix d'extraction) | **Oui — et ça a été fait** |
| Table de faits | `transactions.parquet`, grain `(document, ligne)` | **Le développeur** | Oui, choix technique |
| Agrégat mensuel | `ca_monthly.parquet`, dérivé du registre | **Le développeur** — `src/ledger.py` | Oui |
| Tables de référence | 3 CSV UTF-8-BOM | **Le développeur** — Excel doit pouvoir les ouvrir | Oui |
| Features | 21 numériques + 2 catégorielles | **Le développeur** — contrainte de l'algorithme | Oui |
| Sortie modèle | JSON + CSV + `.cbm` | **Le développeur** — contrainte des consommateurs | Oui |

> **Le point à dire en soutenance :** la ligne 2 était présentée comme non négociable dans
> la première version du projet. Elle l'était. **Le plafond a été levé en demandant** —
> et c'est ce qui a produit le plus gros gain du projet, pas un changement d'algorithme.

**La nuance à garder :** `ca_monthly.parquet` existe toujours, mais il est désormais
**dérivé** du registre (`src/ledger.py`) au lieu d'être la source. Les écrans qui
affichaient cinq mois par an lisent maintenant les trente.

---

## B4. Les 6 règles de nettoyage, et leur justification

`src/build_transactions.py:15-23` — chaque règle protège d'une erreur précise :

| Règle | Pourquoi |
| --- | --- |
| IDs conservés en **texte**, espaces retirés | `007` ≠ `7` : les zéros de tête disparaîtraient en numérique |
| Montants arrondis à **3 décimales** | Précision du millime ; supprime les artefacts de virgule flottante |
| Retours **conservés** et marqués `is_return` | Ce sont des faits sur un couple, pas du bruit — **chaque analyse décide** quoi en faire |
| Lignes **sans client ni article** supprimées | Elles ne peuvent alimenter aucune question |
| **Le signe de `cost` est laissé tel quel** | NAV écrit `Coût total (réel)` en **négatif** quand `Montant vente` est positif |

**La règle du coût mérite un arrêt.** Puisque NAV inverse le signe, la marge se calcule :

```
marge = amount + cost        ← et NON amount - cost
```

> Inventer ici une convention de signe « plus logique » aurait **masqué** celle de NAV.
> Le code documente la bizarrerie au lieu de la corriger en silence : toute personne qui
> calcule une marge plus tard tombe sur l'avertissement plutôt que sur un résultat faux.

---

## B5. Le contrôle de changement de fondation — et son résultat gênant

Changer de source oblige à répondre à : *« la nouvelle donnée raconte-t-elle la même
histoire que celle sur laquelle le système validé a été construit ? »*

`src/reconcile.py` pose deux questions dans l'ordre, et écrit `reports/reconciliation.md` :

**1. Les identifiants se recouvrent-ils ?** — sinon rien n'est comparable et le script s'arrête.

| Entité | v1 | registre | communs | v1 couvert |
| --- | --- | --- | --- | --- |
| clients | 2 060 | 2 191 | 2 059 | **100,0 %** |
| articles | 40 321 | 45 604 | 40 272 | **99,9 %** |
| vendeurs | 54 | 54 | 53 | 98,1 % |

**2. Sur les 15 mois communs, les totaux mensuels coïncident-ils ?**

| Mesure | Résultat |
| --- | --- |
| Tolérance retenue | 1 % par mois |
| Pire écart relatif | **9,89 %** (2025-03) |
| Mois hors tolérance | **11 sur 15** |

> **Et c'est écrit tel quel dans le rapport.** L'extraction v1 était un état pré-agrégé
> avec son propre périmètre et ses propres arrondis ; le registre est la comptabilité.
> Ils ne se recouvrent pas au millime, et **l'écart est une question ouverte pour
> l'entreprise**, pas un bug à faire taire.

**Ce qu'il faut savoir répondre :** les identifiants se recouvrent à ~100 %, donc les deux
sources parlent bien des mêmes clients et des mêmes pièces. Ce sont les **montants** qui
divergent. Un contrôle qui affiche « tout va bien » quand ce n'est pas le cas ne vaut rien —
celui-ci a rempli son office en **refusant** de valider.

---

## B6. Ce que produit l'ingestion

```
 vente 01-01-24 -- 28-05-2025.xlsx  +  vente 29-05-2025 -- 30-06-2026.xlsx
                        │
                        │  src/build_transactions.py
                        ▼
      ┌─────────────────────────────────────────────┐
      │  data/transactions.parquet                  │  ← LA table de faits
      │  1 881 560 lignes × 14 colonnes             │
      │  (date, doc_type, doc_no, article_id,       │
      │   client_id, vendor_code, qty, amount,      │
      │   cost, discount, is_return, …)             │
      └─────────────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │  src/ledger.py             │  src/build_masters.py
          ▼                            ▼
 data/ca_monthly.parquet         vendors / clients / articles .csv
 1 751 053 lignes, 30 mois         44 / 6 309 / 52 115 lignes
 (dérivé, plus une source)
```

| Mesure | Valeur |
| --- | --- |
| Transactions | **1 881 560** |
| Période | **2024-01-02 → 2026-06-30** — 30 mois **continus** |
| Vendeurs / Clients / Articles *(ayant réellement échangé)* | 54 / 2 191 / 45 604 |
| Lignes négatives (retours/avoirs) | 27 659 (1,6 %) |

> **Le contraste à montrer :** 896 K lignes sur 15 mois troués → **1,88 M lignes sur 30 mois
> continus**, avec dates et quantités. Le volume double ; ce qui change vraiment, c'est
> qu'il n'y a **plus de trou** entre juin et décembre.

**Pourquoi 6 309 clients au référentiel mais 2 191 dans les transactions ?** Le référentiel
NAV liste tous les comptes ouverts ; le registre ne contient que ceux qui ont **acheté** sur
la période. L'écart n'est pas une perte — c'est la différence entre un fichier client et une
base d'acheteurs actifs.

---

## B7. Pourquoi Parquet, et pas Excel ou CSV

| Format | Chargement de 896 K lignes | Types | Poids |
| --- | --- | --- | --- |
| Excel `.xlsx` | ~90 s | Devinés, fragiles | 51 Mo |
| CSV | ~4 s | **Perdus** — tout redevient du texte | ~40 Mo |
| **Parquet** | **~0,3 s** | **Conservés** (dates, décimaux, texte) | **~8 Mo** |

Parquet est **colonnaire** : lire uniquement `client_id` et `ca` ne charge pas les autres colonnes.
Les entraînements font des dizaines de `groupby` — le gain est décisif.

**Mais les tables de référence restent en CSV UTF-8-BOM** : elles doivent s'ouvrir dans Excel
côté métier. Le BOM (`utf-8-sig`) est ce qui empêche Excel de casser les accents.

> **Choisir un format, c'est choisir un consommateur.** Parquet pour la machine, CSV pour l'humain.

---

## B8. La couverture des noms — un trou qui s'est refermé

Dans la v1, les noms n'existaient que dans la feuille Mai : **38,8 % des articles et 13,7 %
des clients étaient anonymes**, et aucun retraitement ne pouvait les récupérer. La conclusion
d'alors était : *« c'est une demande de référentiel à l'entreprise, pas un correctif de code. »*

**La demande a été satisfaite** (`deatils articles nav.xlsx`, `deatils clients nav.xlsx`),
et `src/build_masters.py` projette ces référentiels :

| Entité | Présents dans les transactions | Nommés | Couverture |
| --- | --- | --- | --- |
| Clients | 2 191 | 2 191 | **100 %** |
| Articles | 45 604 | 45 585 | **100 %** (19 manquants) |
| Vendeurs | 54 | 44 | **81,5 %** — 10 sans nom |

**Il reste les vendeurs.** Dix codes vendeur actifs n'ont pas de nom dans le référentiel.
C'est le même type de demande qu'avant, sur un périmètre 100× plus petit — et l'interface
affiche le code brut plutôt que d'inventer un libellé.

> **La leçon, répétée :** un manque de donnée se corrige en **demandant la donnée**. Deux des
> trois trous ont été comblés ainsi ; aucun ne l'aurait été par du code.

---

## B9. Les faits structurels qui plafonnent tout le reste

**Rareté extrême** — 1 041 598 couples (client, article) distincts sur une grille possible
de 99,9 M : **1,04 % de remplissage**. Et **70,3 % des couples n'apparaissent qu'un seul mois**.
La plupart des relations client-article sont des coups uniques, pas des habitudes.

> C'est *la* raison pour laquelle le taux de base est de 11,4 % et pas de 50 % : on demande
> au modèle de retrouver une habitude dans une matrice presque vide.

**Concentration forte** — le top 10 % des articles = **74,4 %** du CA ;
le top 10 % des clients = **57,3 %**.

**Propriété multiple — et c'est pire qu'estimé** : 1 992 clients sur 2 191 (**90,9 %**)
apparaissent sous **plusieurs codes vendeur**, jusqu'à **37**. La v1 mesurait 78,4 % et un
maximum de 4 ; le registre complet montre l'ampleur réelle.

> *« À qui appartient ce client ? »* est une **règle métier**, pas une question technique :
> la donnée ne répond pas, et 30 mois continus le montrent bien mieux que 15 mois troués.
> C'est la question ouverte la plus coûteuse qui reste.

---

## B10. Le plafond d'hier, et ce qu'il en reste

**C'était la diapositive la plus importante du dossier v1** : elle définissait le plafond de
ce que tout modèle pouvait faire. La voici avec le verdict du registre NAV.

| Manquait en v1 | Statut aujourd'hui |
| --- | --- |
| Les **dates** (seulement des mois) | ✅ **Résolu** — `date` au jour près |
| Les **quantités** | ✅ **Résolu** — `qty` |
| Le **statut** de commande | ✅ **Résolu** — `doc_type` : 1 827 049 expéditions, 54 511 retours |
| **Juin à décembre** | ✅ **Résolu** — 30 mois continus, aucun trou |
| Le **coût / la marge** | ✅ **Bonus** — `cost`, `discount` (attention au signe, voir B4) |
| Stock, favoris, historique de contact | ❌ **Toujours absent** |

**Ce qui reste vraiment hors de portée :**

| Absent | Conséquence directe |
| --- | --- |
| **Stock** | Le modèle peut recommander une pièce indisponible |
| **Historique de contact** | Impossible de savoir si un client a déjà été appelé — donc pas d'apprentissage par retour |
| **2023** | Une seule fenêtre de test vraiment fraîche (voir F1) |

> **La phrase à dire, et elle a changé :** *« J'ai demandé des données transactionnelles.
> Ce qui existait était un Excel de CA mensuel — alors le projet a été refondé sur le
> registre comptable quand il est devenu disponible. »*
> Les quatre premières lignes du tableau v1 se sont refermées **par une extraction, pas par du code.**

---

## B11. Le point ouvert d'hier — tranché par la donnée

La v1 s'arrêtait en mai 2026 et ce mois était **le plus bas des 15** (−29 % de CA face à
mai 2025), alors qu'avril 2026 était le deuxième plus haut. L'hypothèse posée était :
*« une extraction faite en milieu de mois »*, avec un risque réel — si mai était partiel,
**tout le monde paraissait plus dormant qu'il ne l'était** et la file d'appel se remplissait
de fausses alertes.

**Le registre va jusqu'au 30 juin 2026 et tranche la question :**

| Mois | Lignes | CA |
| --- | --- | --- |
| 2026-03 | 58 384 | 8 477 004 |
| 2026-04 | 68 710 | 10 119 040 |
| **2026-05** | **54 232** | **7 726 506** |
| 2026-06 | 64 642 | 9 200 571 |

Juin 2026 est complet et **normal**. Mai 2026 était donc **un vrai mois creux**, pas une
troncature d'export. Aucun retrait ni proratisation n'est nécessaire.

> **Ce que ça enseigne sur la méthode :** l'hypothèse de la v1 était raisonnable et
> impossible à vérifier avec la donnée d'alors — c'est pour ça qu'elle avait été **posée
> comme question ouverte plutôt que corrigée en silence**. Une correction appliquée par
> prudence aurait effacé un fait métier réel.

---
---

# PARTIE C — Les algorithmes et les mathématiques

---

## C1. Les six algorithmes du projet

| # | Algorithme | Où | Rôle |
| --- | --- | --- | --- |
| 1 | **Régression logistique** | *(étalon v1, retiré en v4)* | Étalon linéaire — remplacé par le tri par CA passé |
| 2 | **Gradient boosting (CatBoost)** | les deux modèles | Le modèle retenu |
| 3 | **Lift / règles d'association** | `prospect_features.py:213` | Construire la « signature » d'un article |
| 4 | **Similarité cosinus** | `prospect_features.py:87` | Mesurer la ressemblance des paniers |
| 5 | **k plus proches voisins (k-NN)** | `prospect_features.py:306` | Compter les voisins acheteurs |
| 6 | **Régression isotonique (PAVA)** | *(mesuré en v1)* | Tenter de calibrer — **échec documenté** |

---

## C2. Algorithme 1 — La régression logistique

**Le modèle le plus simple qui répond à une question binaire.** Il cherche une droite.

```
        p = σ(w₁x₁ + w₂x₂ + … + w₁₈x₁₈ + b)

                        1
avec   σ(z) = ───────────────────        (fonction logistique / sigmoïde)
                   1 + e^(−z)
```

- `x` = les 18 features numériques
- `w` = les poids **appris**, un par feature
- `σ` écrase n'importe quel nombre réel dans l'intervalle ]0 ; 1[ — ce qui en fait une probabilité

**Ce qu'on minimise** — la *log-loss* (entropie croisée binaire) :

```
L = − Σ [ yᵢ · log(pᵢ) + (1 − yᵢ) · log(1 − pᵢ) ]
```

Cette fonction punit très durement la **confiance erronée** : prédire 0,99 pour un cas qui
vaut 0 coûte `−log(0,01) ≈ 4,6`, contre `−log(0,5) ≈ 0,7` pour une prédiction indécise.

---

## C3. Pourquoi `StandardScaler` avant la régression logistique

```python
scaler = StandardScaler()
lr.fit(scaler.fit_transform(train[NUMERIC]), train["label"])
```

```
         x − μ
   z = ─────────      moyenne 0, écart-type 1
           σ
```

**Deux raisons, toutes deux réelles :**

1. `ca_sum` va jusqu'à 100 000 ; `regularity` vaut entre 0 et 1. Sans normalisation,
   la descente de gradient zigzague et converge mal.
2. La régularisation L2 de scikit-learn (activée par défaut) pénalise les grands coefficients.
   Sans normalisation, elle **punirait plus fort les features à petite échelle** — un biais purement artificiel.

**Point important pour la soutenance :** CatBoost, lui, n'en a **pas besoin**. Les arbres
comparent des seuils (`ca_sum > 4000 ?`), et un seuil est insensible au changement d'unité.

---

## C4. Algorithme 2 — Le gradient boosting, la formule

Le modèle final est une **somme** :

```
   F_M(x)  =  F₀(x)  +  ν · h₁(x)  +  ν · h₂(x)  +  …  +  ν · h_M(x)

   M = 800 arbres        ν = 0,05 (learning_rate)
```

À chaque étape `m`, on calcule le **pseudo-résidu** de chaque ligne :

```
            ∂L
   rᵢ = − ────── = yᵢ − pᵢ        (pour la log-loss)
           ∂F(xᵢ)
```

**Traduction :** `rᵢ` est simplement *« de combien je me suis trompé sur cette ligne »*.
L'arbre `h_m` est ensuite entraîné **à prédire cette erreur**, pas l'étiquette.

C'est là qu'est le mot *gradient* : chaque arbre est un pas dans la direction de la plus
forte pente de descente de l'erreur.

---

## C5. Du score à la probabilité

Le boosting travaille en **log-odds** (logarithme de la cote), pas en probabilité :

```
                     p                              1
   F(x) = log ───────────      ⟺      p = ───────────────────
                   1 − p                        1 + e^(−F(x))
```

**Pourquoi ?** Les log-odds vont de −∞ à +∞ — on peut donc y additionner 800 arbres sans
jamais sortir de l'intervalle valide. En probabilité directe, additionner ferait dépasser 1.

`model.predict_proba()` applique la sigmoïde à la fin. C'est **la même fonction** que dans la
régression logistique : les deux algorithmes ne diffèrent pas par la sortie, mais par la façon
de construire `F(x)` — une droite d'un côté, une somme de 800 arbres de l'autre.

---

## C6. Ce que CatBoost fait de spécial — 1. Les arbres symétriques

Un arbre CatBoost est dit **oblivious** : **la même question à tous les nœuds d'un même niveau**.

```
Niveau 1 :          client_n_articles > 50 ?
                    /                        \
Niveau 2 :   months_n > 2 ?              months_n > 2 ?     ← MÊME question
              /        \                  /        \
Niveau 3 : enseigne∈{…}? enseigne∈{…}? enseigne∈{…}? enseigne∈{…}?
```

`depth = 6` → exactement **2⁶ = 64 feuilles**, toujours.

**L'avantage est la vitesse de prédiction :** trouver la feuille se réduit à construire un
nombre binaire de 6 bits, puis à lire une case de tableau. Pas de parcours d'arbre, pas de
branchement conditionnel. C'est ce qui rend le scoring de 251 342 paires quasi instantané.

**Le coût :** un arbre symétrique est moins expressif — compensé par leur nombre (800)
et par une régularisation implicite qui limite le surapprentissage.

---

## C7. Ce que CatBoost fait de spécial — 2. Les variables catégorielles

**Le vrai problème :** `enseigne` a 262 valeurs, `vendor_code` en a 54.

| Méthode classique | Pourquoi elle échoue ici |
| --- | --- |
| *One-hot encoding* | 262 colonnes de 0/1 → explosion et arbres inutilisables |
| *Label encoding* (BOSH=1, FEBI=2…) | Invente un ordre qui n'existe pas |
| *Target encoding* naïf | **Fuite** : la moyenne de la cible contient la ligne elle-même |

**La solution CatBoost — statistiques de cible ordonnées :**

```
                Σ  yⱼ  +  a · P
              j<i, xⱼ=xᵢ
      x̂ᵢ  =  ───────────────────
                 nᵢ  +  a
```

Chaque catégorie est remplacée par une moyenne de la cible calculée **uniquement sur les
lignes précédentes** d'une permutation aléatoire — jamais sur la ligne courante.
`a · P` est un lissage bayésien qui protège les catégories rares.

> **C'est précisément la raison du choix de CatBoost pour ce projet** : `enseigne` sort
> 3ᵉ feature la plus importante (14,6). Aucun encodage naïf n'aurait pu l'exploiter sans fuite.

---

## C8. Ce que CatBoost fait de spécial — 3. L'*ordered boosting*

Le même principe, appliqué au boosting lui-même : le résidu d'une ligne est calculé avec un
modèle qui **n'a jamais vu cette ligne**.

Cela corrige le *prediction shift* — un biais subtil du boosting classique, où les résidus
d'entraînement sont systématiquement trop optimistes parce que le modèle a déjà appris
sur ces mêmes lignes.

**Le fil conducteur de CatBoost est toujours le même :**
> *ne jamais utiliser l'information d'une ligne pour prédire cette ligne.*

C'est exactement le principe que le protocole d'entraînement applique à l'échelle du projet
avec la séparation chronologique. **Même idée, deux échelles.**

---

## C9. L'arrêt précoce (*early stopping*)

```python
iterations            = 6000
early_stopping_rounds = 200
```

Après chaque arbre, on mesure l'AUC sur la validation.
**Si 200 arbres consécutifs n'apportent aucune amélioration, on s'arrête.**

```
AUC validation
   ▲
   │         ╭──────────╮_________  ← plus rien à gagner : on s'arrête
   │     ╭───╯          ↑
   │   ╭─╯         best_iteration = 1345
   │ ╭─╯
   └─┴────────────────┴──────────►  arbres
     0              1345        6000  (plafond jamais atteint)
```

**Résultat mesuré : `best_iteration = 1345`** sur un plafond de 6 000.

> **Le contraste avec la version précédente est l'argument à retenir.** La v1 s'arrêtait à
> `best_iteration = 799` sur un plafond de 800 : le modèle **apprenait encore** quand on
> l'a coupé, et le plafond était la contrainte. Ici l'arrêt précoce **s'est déclenché de
> lui-même**, très loin du plafond.
>
> Un modèle stoppé *au* plafond ne dit rien : on ignore s'il aurait continué à progresser.
> Un modèle stoppé par l'arrêt précoce dit *« j'ai fini »*. **C'est ce qui permet d'écrire
> `converged: true` dans le rapport** (`reports/rolling_metrics.json`) — et de ne plus avoir
> à défendre le choix du plafond.

---

## C10. Algorithme 3 — Le *lift*, pour construire la signature d'un article

**Question :** quels articles sont caractéristiques des acheteurs d'une pièce donnée ?

```
                  P(article A | acheteur de X)         part chez les acheteurs de X
   lift(A | X) = ───────────────────────────────  =  ─────────────────────────────────
                        P(article A)                     part dans tout le catalogue
```

| Valeur | Interprétation |
| --- | --- |
| lift = 1 | Aucun lien — A est aussi courant partout |
| lift = 3 | A est **3× plus fréquent** chez les acheteurs de X |
| lift < 1 | Lien négatif |

**Les seuils, et pourquoi** — `src/article_prospects.py:22-25` :

```python
MIN_SHARE     = 0.15   # ≥ 15 % des acheteurs prennent l'article
MIN_LIFT      = 1.2    # ... et il est 1,2× plus courant chez eux
TOP_SIGNATURE = 30     # on garde les 30 meilleurs, triés par lift
MIN_BUYERS    = 3      # en dessous, le ratio est du bruit arithmétique
```

`MIN_SHARE` **sans** `MIN_LIFT` retiendrait les best-sellers que tout le monde achète.
`MIN_LIFT` **sans** `MIN_SHARE` retiendrait des coïncidences à 2 clients.
**Il faut les deux.**

---

## C11. Algorithme 4 — La similarité cosinus

Chaque client devient un **vecteur binaire** : 1 s'il achète l'article, 0 sinon.

```
                      u · v
   cos(u, v) = ───────────────────       ∈ [0 ; 1] pour des vecteurs binaires
                  ‖u‖ · ‖v‖
```

**Le cas binaire se simplifie remarquablement :**

```
                    |A ∩ B|
   cos(A, B) = ─────────────────         (coefficient d'Ochiai)
                 √(|A| · |B|)
```

- Numérateur : **combien d'articles en commun**
- Dénominateur : **normalisation par la taille des paniers**

> **Le dénominateur est tout l'intérêt.** Sans lui, le plus gros garage serait « similaire »
> à tout le monde, simplement parce qu'il achète tout. Le cosinus mesure la **forme** du panier,
> pas sa taille — et `client_ca` est déjà là pour mesurer la taille séparément.

**Binaire et non pondéré par le CA** (`prospect_features.py:72`) : la question est *« achète-t-il
cette pièce ? »*, pas *« combien dépense-t-il ? »*. Pondérer laisserait quelques lignes chères
dominer toute la similarité.

---

## C12. La contrainte mémoire, et sa solution

**Le problème :** la matrice client × article en dense.

```
1 600 clients × 40 000 articles × 8 octets  =  512 Mo
… pour 0,75 % de cases non nulles.
```

**La solution — matrice creuse CSR** (`scipy.sparse.csr_matrix`) : on ne stocke que les valeurs
non nulles et leurs coordonnées. **512 Mo → quelques Mo.**

**Mais la matrice de similarité, elle, est dense — et c'est volontaire :**

```
1 600 clients × 1 600 clients  ≈  10 Mo      ✅ acceptable
```

> Le résultat est client × client, pas client × article.
> **Creux là où c'est nécessaire, dense là où c'est possible.**
> Un choix d'ingénierie, pas un dogme.

---

## C13. Algorithme 5 — k plus proches voisins

`knn_buyers` = *parmi les 20 clients au panier le plus proche, combien achètent déjà cette pièce ?*

```python
KNN_K = 20
nearest = np.argpartition(-ctx.similarity[rows], take, axis=1)[:, :k]
out.loc[known, "knn_buyers"] = is_buyer[nearest].sum(axis=1)
```

**`argpartition` et non `argsort`** — la différence est mesurable :

| Opération | Complexité | Ce qu'elle fait |
| --- | --- | --- |
| `argsort` | O(n log n) | Trie **tout** — on jette 98 % du travail |
| `argpartition` | **O(n)** | Place juste les 20 premiers devant, sans les ordonner |

Et le calcul est **groupé** : une seule opération pour tous les candidats.
La version en boucle coûtait **1,2 s par article** — payé à chaque interaction de l'écran,
pas seulement à l'entraînement.

C'est la 3ᵉ feature du modèle prospects (importance 9,29), et elle donne au vendeur une phrase
qu'il peut dire au téléphone : *« 7 des 20 garages au panier le plus proche du vôtre prennent cette pièce. »*

---

## C14. L'échantillonnage négatif — et son prix

**Le problème :** pour une pièce donnée, presque personne ne l'adopte. Un modèle qui répond
toujours « non » aurait raison à 99,8 % et serait parfaitement inutile.

**La solution :** 20 négatifs tirés par positif (`NEG_RATIO = 20`).

```
752 621 lignes   dont   36 747 positives   =   4,88 %
```

**Le prix à payer, et il faut le dire :** rééchantillonner **modifie le taux de base**.
Les probabilités absolues sorties du modèle ne correspondent plus à la réalité du terrain.

> **Mais le classement à l'intérieur d'un article est préservé** — et c'est tout ce dont
> l'écran a besoin : il affiche les 5 meilleurs prospects, pas un pourcentage.
> **On n'a payé que ce dont on n'avait pas besoin.**

---

## C15. Algorithme 6 — La régression isotonique, et son échec

**L'objectif :** corriger l'excès de confiance pour afficher de vrais pourcentages.

La régression isotonique ajuste une **fonction en escalier croissante** minimisant l'erreur
quadratique, par l'algorithme **PAVA** (*Pool Adjacent Violators*) :

```
   min  Σ (yᵢ − f(pᵢ))²      sous contrainte   f croissante
```

**Croissante ⟹ le classement est inchangé ⟹ l'AUC est inchangée.**
Seuls les nombres bougent, pas l'ordre.

**Le résultat mesuré en v1 :**

| Écart de calibration | Avant | Après |
| --- | --- | --- |
| Maximum sur une tranche (MCE) | 10,7 % | **13,9 %** ❌ |
| Moyen pondéré par le volume (ECE) | 2,2 % | **2,5 %** ❌ |

**La correction dégradait la calibration. Elle n'a pas été appliquée.**

> **Et la suite donne raison à cette prudence.** Le modèle v4, sans aucune étape de
> calibration, mesure **MCE 3,2 % et ECE 0,7 %** sur sa fenêtre de test (voir D7). Le
> problème que la régression isotonique tentait de corriger a été résolu par **plus de
> données et un horizon mieux posé**, pas par un correctif a posteriori.

---

## C16. Pourquoi la calibration échoue — et ce que ça enseigne

Sur la validation, le modèle est bien calibré, voire **légèrement sous-confiant** :
cette validation partage la **même année** que l'entraînement, donc ses prédictions
confiantes se réalisent vraiment.

**La courbe apprend donc à pousser les probabilités vers le haut.**
Franchi le passage d'une année à l'autre, elles ne se réalisent plus à ce rythme :
la tranche haute passe de **375 à 1 525 paires**, dont seulement **78,9 %** rachètent.

**Cause racine :** il n'existe aucun jeu retenu représentant une **année fraîche**.
Calibrer honnêtement demanderait des instantanés 2023→2024, et **2023 n'est pas dans l'extraction**.

> **La leçon générale :** savoir **classer** et savoir **chiffrer** sont deux compétences
> distinctes. Ce modèle a la première et pas la seconde — d'où l'affichage
> *Élevée / Moyenne / Faible* plutôt que « 71 % ».
> **Un résultat négatif documenté vaut mieux qu'un chiffre flatteur non vérifié.**

---
---

# PARTIE D — Les graphiques de décision

---

## D1. Graphique 1 — La courbe ROC et l'AUC

On fait glisser un seuil de décision de 1 vers 0, et on trace deux taux :

```
              vrais positifs                        faux positifs
   TPR = ──────────────────────        FPR = ──────────────────────
          vrais pos. + faux nég.               faux pos. + vrais nég.
```

```
  TPR ▲
  1.0 │            ╭──────────  CatBoost v4 — AUC 0,774
      │        ╭───╯
      │     ╭──╯   ╭─────────   Tri par CA passé
      │   ╭─╯   ╭──╯
      │ ╭─╯  ╭──╯          ╱    ← diagonale = hasard, 0,500
      │╭╯ ╭──╯        ╱
  0.0 └────────────────────────► FPR
      0.0                    1.0
```

**L'AUC est l'aire sous cette courbe.** Sa définition probabiliste est la plus parlante :

```
   AUC = P( score(positif tiré au hasard) > score(négatif tiré au hasard) )
```

Elle est **égale à la statistique U de Mann–Whitney normalisée** : `AUC = U / (n₊ · n₋)`.
C'est un test de rang — d'où le fait qu'elle ne dépende **que de l'ordre**, jamais des valeurs.

---

## D2. Pourquoi l'AUC, et pas « le taux de bonnes réponses »

**Le piège de l'exactitude (*accuracy*) :** 11,4 % des paires rachètent.
Un modèle qui répond **toujours « non »** obtient **88,6 % d'exactitude** — et ne sert à rien.
Le déséquilibre s'est accentué avec l'horizon de 3 mois, donc le piège aussi.

| Métrique | Sensible au déséquilibre ? | Dépend d'un seuil ? |
| --- | --- | --- |
| Exactitude | ❌ Trompeuse ici | Oui |
| **AUC** | ✅ Robuste | **Non** |
| Précision@k | ✅ Robuste | Non — c'est un rang |

**Et surtout :** aucun seuil n'a besoin d'être choisi. Le vendeur ne demande pas
*« ce client va-t-il racheter, oui ou non ? »* — il demande *« qui j'appelle en premier ? »*.

> **La métrique doit poser la question du produit.**
> Ici, la question est un **classement**, donc la métrique est une métrique de rang.

---

## D3. Graphique 2 — La courbe de *lift* par décile

**C'est le graphique à montrer à un décideur.** On trie les paires par score,
on les coupe en 10 groupes égaux, et on mesure le taux de rachat réel dans chacun.

```
  Taux de rachat (test 2026-01..04)
   39% ┤ ██                                   décile 10
   21% ┤ ██ ██                                décile 9
   15% ┤ ██ ██ ██                             décile 8
   11% ┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  ← taux de base 11,4 %
    9% ┤ ██ ██ ██ ██ ██
    7% ┤ ██ ██ ██ ██ ██ ██
    5% ┤ ██ ██ ██ ██ ██ ██ ██
    4% ┤ ██ ██ ██ ██ ██ ██ ██ ██
    3% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██
    1% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██        décile 1
       └──────────────────────────────►
        10  9  8  7  6  5  4  3  2  1
```

**Il traduit l'AUC en temps de vendeur** : *« votre liste est 3× plus dense »*
se comprend sans aucun vocabulaire technique.

---

## D4. Le détail d'implémentation à connaître

```python
d["dec"] = pd.qcut(d["s"].rank(method="first"), 10, labels=range(1, 11))
```

**Pourquoi `.rank(method="first")` avant `qcut` ?**

Le score manuel `ca × mois / 5` produit d'énormes paquets d'**ex æquo**. `qcut` calcule ses
bornes sur les quantiles : avec des valeurs identiques, plusieurs bornes coïncident et
**`qcut` lève une exception**.

Le rang `"first"` départage les ex æquo par ordre d'apparition, garantissant 10 groupes
strictement égaux. **Sans cette ligne, le tableau comparatif ne pourrait pas être produit
pour l'étalon métier** — et on ne pourrait pas prouver que le modèle le bat.

---

## D5. Graphique 3 — Précision@k et rappel@k

La métrique du modèle prospects. L'écran affiche **5 clients** : seul ce top 5 compte.

```
                    hits dans le top k                    hits dans le top k
   precision@k = ────────────────────      recall@k = ──────────────────────────
                            k                          nombre total d'adoptants
```

**Un choix de conception documenté** (`prospect_model.py:87-95`) : on divise par **k**,
pas par la longueur de la liste.

> *« La page offre toujours cinq créneaux ; une liste courte qui n'en remplit que trois
> a quand même consommé cinq créneaux d'attention du vendeur. »*

**Et les articles sans aucune adoption sont exclus** de la moyenne : leur precision@k vaut 0
quel que soit le classement. Les inclure mesurerait le taux d'adoption du catalogue,
**pas la qualité du classement**. Leur nombre est publié (**1 649 articles retenus**) pour
que la couverture reste visible.

---

## D6. Graphique 4 — L'importance des features

Rendu en barres horizontales dans le tableau de bord (`web/src/pages/ModelsPage.tsx`, Recharts).

CatBoost utilise par défaut **`PredictionValuesChange`** : de combien la prédiction change,
en moyenne, quand la valeur de la feature change. **Les valeurs somment à 100.**

```
client_days_since_last  ██████████████████  18,0
client_n_orders         █████████▊           9,8
article_n_clients       █████████▎           9,3
enseigne                █████████            9,0
client_n_articles       ███████▌             7,6
client_amount           ██████▌              6,5
article_repurch.        █████▉               5,9
```

**Le renversement par rapport à la v1 est l'observation à faire ici.** Les features de
**volume** menaient (`client_n_articles` 16,9 · `client_ca` 15,0) ; ce sont maintenant les
features de **récence et de fréquence** qui dominent — et la première d'entre elles était
**impossible à calculer** avant le registre NAV.

**Deux usages, dont un seul est technique :**
1. Diagnostiquer — une feature à importance nulle est du bruit à retirer.
2. **Justifier auprès du vendeur** — les explications affichées sont ordonnées selon cette
   importance, pour que la raison la plus forte se lise en premier (`prospect_model.py:201-203`).

⚠️ **Corrélation, pas causalité.** *« `client_ca` est important »* ne veut pas dire
*« augmenter le CA d'un client le fera racheter »*.

---

## D7. Graphique 5 — La courbe de fiabilité (calibration)

On range les prédictions en 10 tranches et on compare **le prédit** à **le réalisé** :

```
  Réalisé ▲
    1.0   │                          ╱╳   ← diagonale = calibration parfaite
          │                     ╱ ╳
          │                ╱ ╳          ← v4 : quasiment SUR la diagonale
          │           ╳ ╱                 (écart max 3,2 %)
          │      ╳ ╱
          │  ╳╱
    0.0   └──────────────────────────►
          0.0                      1.0   Prédit
```

Les deux mesures de calibration portent des noms standards :

```
              nb                                          nb
   ECE = Σ  ────── · |prédit_b − réalisé_b|      MCE = max │prédit_b − réalisé_b│
         b     N                                       b
```

| Mesure | v1 | **v4** |
| --- | --- | --- |
| **ECE** — écart moyen pondéré par le volume | 2,2 % | **0,7 %** |
| **MCE** — pire écart sur une tranche | 10,7 % | **3,2 %** |

**Le modèle v4 est bien calibré**, et il l'est **sans aucune étape de calibration** — la
régression isotonique reste non appliquée (voir C15). Prédiction moyenne 10,74 % contre
11,37 % observés : le modèle est même légèrement **sous**-confiant, l'inverse du défaut v1.

> **Et pourtant l'API continue d'annoncer `score_is_calibrated: false`.** C'est délibéré :
> aucune étape de calibration n'est appliquée, donc rien ne *garantit* la propriété — elle
> est constatée sur une fenêtre, pas construite. Annoncer « calibré » engagerait un contrat
> que le pipeline ne tient pas. **Mesurer une bonne propriété ne suffit pas à la promettre.**

---

## D8. Récapitulatif — quel graphique pour quelle décision

| Graphique | Décision qu'il permet | Public |
| --- | --- | --- |
| **Courbe ROC / AUC** | Le modèle bat-il l'étalon ? | Technique |
| **Lift par décile** | Combien de temps vendeur est gagné ? | **Direction** |
| **Précision@k** | Le modèle mérite-t-il de remplacer l'heuristique ? | Technique + produit |
| **Importance des features** | Quoi retirer ; quoi dire au vendeur | Technique + terrain |
| **Courbe de fiabilité** | A-t-on le droit d'afficher un pourcentage ? | **Produit** |

> **Aucune de ces cinq courbes ne mesure la même chose, et aucune n'est optionnelle.**
> Un seul chiffre ne peut pas décider à la fois d'un déploiement et d'un libellé d'interface.

---
---

# PARTIE E — Le déroulé d'un entraînement

---

## E1. La chaîne complète

```
 1. INGESTION            src/build_transactions.py
    2 classeurs NAV (217 Mo) → transactions.parquet (1,88 M lignes)
                            │
 2. RÉFÉRENTIELS         src/build_masters.py
    masters NAV → articles.csv, clients.csv
                            │
 3. AGRÉGAT              src/ledger.py
    transactions → ca_monthly.parquet (30 mois, dérivé)
                            │
 4. INSTANTANÉS          src/snapshots.py
    16 dates T, 12 mois de recul, 3 mois d'horizon
    → train 3,0 M · val 1,0 M · test 2,05 M lignes
      + 4 instantanés d'embargo CALCULÉS PUIS JETÉS
                            │
 5. ENTRAÎNEMENT         src/train_rolling.py
    CatBoost, plafond 6 000, arrêt précoce 200
    → best_iteration 1345, converged: true
                            │
 6. ÉVALUATION           sur le test 2026-01..04, UNE SEULE FOIS
    AUC 0,774 · décile 10 = 39,3 % (×3,45) · top-30 = 61,9 %
    → data/pair_scores_rolling.csv  ← LE BACKTEST (avec label)
                            │
 7. SCORING LIVE         src/score_live.py
    instantané à T = le mois APRÈS la dernière transaction
    → data/pair_scores_live.csv     ← LA PRÉVISION (sans label)
      + reports/live_window.json    ← la fenêtre servie
                            │
 8. FILE D'APPEL         src/baseline_lists.py → src/contact_queue.py
    → data/clients_to_contact.csv
```

> **L'étape 7 est la plus facile à oublier, et la plus coûteuse.** L'étape 6 produit un
> **backtest** : chaque ligne porte déjà sa réponse, et sa fenêtre est terminée. Servir ce
> fichier à un vendeur, c'est lui recommander d'appeler pour un trimestre **déjà passé**.
> L'étape 7 rejoue le même modèle sur la fenêtre **qui n'a pas encore eu lieu**.
>
> Deux fichiers, même modèle, sens opposés. `pair_scores_rolling.csv` **prouve** ;
> `pair_scores_live.csv` **sert**.

---

## E2. Ce que produit un entraînement

| Fichier | Format | **Consommateur** |
| --- | --- | --- |
| `data/model_rolling.cbm` | Binaire CatBoost | `score_live.py` — le scoring |
| `reports/model_rolling.md` | Markdown | **Un humain** — versionnable, lisible en diff Git |
| `reports/rolling_metrics.json` | JSON | **Une machine** — l'API (`/meta`) et le tableau de bord |
| `data/pair_scores_rolling.csv` | CSV UTF-8-BOM | La page de validation — **le backtest, avec label** |
| `data/pair_scores_live.csv` | CSV UTF-8-BOM | **L'API et la file d'appel — la prévision** |
| `reports/live_window.json` | JSON | La fenêtre servie : `snapshot_date`, `window_end`, `horizon_months` |

> **Le même résultat est écrit deux fois, en Markdown et en JSON.** Ce n'est pas une
> redondance : un rapport Markdown se relit en revue de code un an plus tard,
> un JSON s'affiche dans une interface. Deux consommateurs, deux formats.

---

## E3. Les deux protocoles de validation, comparés

| | Modèle 1 — rachat | Modèle 2 — prospection |
| --- | --- | --- |
| Séparation | **Chronologique** — instantanés mensuels | **Chronologique** — fenêtres semestrielles |
| Ce que ça sépare | Des dates `T` | Des semestres, et **des articles entiers** |
| Embargo | **2 embargos** de 2 instantanés (jetés) | **1 semestre** (2025-H1, jeté) |
| Plafond / arrêt | 6 000 / 200 → **1 345** | CatBoost, graine `SEED = 7` |
| Test final | 2026-01 … 04 | features 2025-H2 → labels 2026-H1 |

**Ce qui a changé, et qu'il faut savoir défendre :** la v1 coupait le modèle 1 **au hasard**
(80/20 sur la même année). Les deux modèles sont désormais découpés **dans le temps**, avec
un embargo explicite — la seule découpe qui reproduise la production, où le futur n'est
jamais disponible.

**Le détail resté vrai pour le modèle 2 :** ses lignes sont empilées **article par article**,
donc la coupe sépare des **pièces entières**. Le test contient des articles **jamais vus** —
beaucoup plus proche de la tâche réelle qu'un tirage aléatoire, qui aurait laissé le modèle
voir chaque article des deux côtés.

---

## E4. La reproductibilité — et une erreur classique évitée

| Graine | Où | Ce qu'elle fixe |
| --- | --- | --- |
| `random_seed=42` | CatBoost — `train_rolling.py:114` | Les permutations de l'*ordered boosting* |
| `SEED = 7` | Modèle prospects — `prospect_model.py:30` | Échantillonnage des articles et des négatifs |
| `crc32(article_id)` | Étalon aléatoire — `prospect_model.py:151` | Le mélange, par article |

*(`random_state` pour `train_test_split` a disparu avec le tirage aléatoire : une découpe
chronologique n'a pas besoin de graine — elle est déterministe par construction.)*

**Le piège évité** (`prospect_model.py:147-148`) : `hash()` en Python est **randomisé à chaque
lancement** du processus, sauf si `PYTHONHASHSEED` est fixé. Utiliser `hash()` pour semer
l'étalon « aléatoire » aurait donné un étalon **différent à chaque exécution** — tout en
étant présenté comme reproductible. `zlib.crc32` est déterministe par construction.

> **Un étalon non reproductible ne prouve rien** : si le hasard change entre deux exécutions,
> l'écart mesuré peut venir du hasard plutôt que du modèle.

---

## E5. Les protections contre la triche, dans le code

| Risque | Protection | Où |
| --- | --- | --- |
| Fuite temporelle | Features avant `T`, label après `T` | `snapshots.py:6-7` |
| **Fuite entre découpes** | **Deux embargos**, 4 instantanés jetés | `snapshots.py:52-56` |
| Fuite par feature d'état courant | `assert_no_current_state_features()` — liste interdite | `train_rolling.py` |
| Fuite dans l'encodage catégoriel | *Ordered target statistics* de CatBoost | Algorithme |
| **Décalage train/service** | **Un seul constructeur de features** partagé | `prospect_features.py:2-6` |
| **Servir un backtest** | Le label est **retiré** du frame servi par l'API | `repository.py:109` |
| Réglage contre le test | Comparaison v1↔v4 explicitement interdite | `rolling_metrics.json` |

**Le décalage train/service (*train/serve skew*) mérite un mot :** si l'écran calculait une
feature autrement que l'entraînement, le modèle marcherait hors ligne et décevrait sur le
terrain — **silencieusement**, sans aucune erreur. La parade est structurelle : la page et le
script d'entraînement appellent **la même fonction**.

---
---

# PARTIE F — Limites et suites

---

## F1. Les limites à énoncer soi-même — mises à jour

**Deux des trois limites de la v1 sont tombées avec le registre NAV.** Les énoncer ainsi est
plus honnête que de les recopier.

| Limite v1 | Statut |
| --- | --- |
| *« Il ne prédit aucune date »* | ✅ **Levée** — le registre est journalier (voir A7) |
| *« Il classe bien, il chiffre mal »* (MCE 10,7 %) | ✅ **Largement levée** — MCE mesurée à **3,2 %**, ECE **0,7 %** |
| *« Une seule fenêtre de test honnête »* | ⚠️ **Atténuée** — 4 instantanés de test au lieu d'un, mais 2023 manque toujours |

**Les limites réelles aujourd'hui :**

**1. La prévision a une date de péremption.**
Les scores servis répondent pour une fenêtre **fixe** — actuellement 2026-07-01 → 2026-09-30.
Passée cette date, ce ne sont plus des recommandations mais un **enregistrement historique**,
sans que rien ne change dans le fichier. D'où `reports/live_window.json` et le champ
`evaluation_mode` de l'API, qui bascule de `live_forecast` à `expired_forecast` tout seul.
**Il faut re-scorer, pas ré-entraîner** — c'est `src/score_live.py`, quelques minutes.

**2. Aucune boucle de retour.**
Aucun historique de contact : impossible de savoir si un client a été appelé, ni ce que
l'appel a donné. Le modèle apprend du **comportement d'achat**, jamais du résultat de ses
propres recommandations. C'est ce qui interdit l'apprentissage par renforcement (voir A1).

**3. 2023 manque toujours.**
Le recul reste de 30 mois. Il suffit pour des instantanés glissants, pas pour un holdout
d'une année entière vraiment fraîche.

**4. Le stock est invisible.**
Le modèle peut recommander une pièce indisponible. C'est une jointure à faire côté ERP,
pas un problème de modèle.

---

## F2. Le lien entre les limites — une seule cause

Les limites restantes ne partagent plus une cause unique — elles en ont **deux**, et la
distinction commande la réponse à donner :

```
   CE QUI MANQUE DANS LA DONNÉE          CE QUI EST DE NATURE OPÉRATIONNELLE
   ┌──────────────────────────┐          ┌──────────────────────────────────┐
   │  2023  ·  contacts       │          │  la fenêtre servie expire        │
   │  stock                   │          │                                  │
   └───────────┬──────────────┘          └───────────────┬──────────────────┘
               ▼                                         ▼
   Se lève par une EXTRACTION              Se lève par une EXÉCUTION
   (demande à l'entreprise)                (re-scorer, pas ré-entraîner)
```

> **Le précédent qui rend l'argument crédible :** les quatre manques de la v1 (dates,
> quantités, statut, juin-décembre) se sont tous levés par une extraction. La colonne de
> gauche n'est donc pas une fatalité — c'est un devis.
>
> La colonne de droite est différente : aucune donnée supplémentaire ne la réglera. Elle
> demande qu'on **relance `score_live.py`** à intervalle régulier. Une limite qu'on lève en
> exécutant une commande n'est pas du même ordre qu'une limite qu'on lève en négociant.

---

## F3. La phrase de conclusion

> Sur 30 mois de registre comptable continu, le modèle répond à *« qui vaut la peine d'être
> appelé en premier, et pour quelle pièce »*. Sa liste des 30 priorités par vendeur est
> **juste à 61,9 % contre 31,9 % pour un tri par chiffre d'affaires** — le réflexe Excel
> qu'il remplace. Son décile haut est **3,45× plus dense** que le hasard, et sa performance
> est **stable à ±0,002 d'AUC sur quatre mois consécutifs** jamais vus.
>
> Le second modèle désigne les clients qui **n'ont jamais** acheté une pièce et vont
> l'adopter, **24× mieux que le hasard**.
>
> **Et le gain le plus important du projet n'est venu d'aucun algorithme :** il est venu du
> remplacement d'un export Excel agrégé par le registre comptable — ce qui a rendu inutile
> la moitié des contournements de la première version.

---

## F4. Les sept chiffres à connaître par cœur

| Chiffre | Ce qu'il prouve |
| --- | --- |
| **1 881 560** transactions, **30 mois continus**, 54 vendeurs, 2 191 clients, 45 604 articles | L'échelle réelle |
| **0,774** AUC de test, et **±0,002 sur 4 mois** | Le modèle marche, et pas qu'une fois |
| **61,9 %** de précision sur le top 30 par vendeur, contre **31,9 %** en triant par CA | Le gain, au niveau du vendeur |
| **39,3 %** de rachat dans le top 10 %, contre **11,4 %** de base — **×3,45** | Le gain, en langage métier |
| **12,98 %** precision@5 prospects, contre **0,55 %** au hasard | Le 2ᵉ modèle vaut **24×** le hasard |
| **1 345** itérations sur un plafond de **6 000** | Le modèle a **convergé**, il n'a pas été coupé |
| **18,0** d'importance pour `client_days_since_last` | La récence domine — et elle n'existait pas en v1 |

> **Le chiffre qu'il ne faut PAS citer :** l'écart entre l'AUC 0,762 de la v1 et le 0,774
> d'aujourd'hui. Horizons et labels différents — `rolling_metrics.json` l'interdit
> explicitement (voir A12). Savoir refuser cette comparaison vaut mieux que de l'utiliser.

---

## F5. Références du code

| Sujet | Fichier |
| --- | --- |
| Ingestion du registre NAV | `src/build_transactions.py` |
| Référentiels clients / articles | `src/build_masters.py` |
| Agrégat mensuel (dérivé) | `src/ledger.py` |
| Contrôle de changement de fondation | `src/reconcile.py` → `reports/reconciliation.md` |
| Instantanés glissants + découpes | `src/snapshots.py` |
| Modèle de rachat | `src/train_rolling.py` |
| **Scoring de la fenêtre à venir** | `src/score_live.py` |
| Features prospects | `src/prospect_features.py` |
| Logique prospects | `src/prospect_model.py` |
| Entraînement prospects | `src/train_prospect_model.py` |
| Résultats — humain | `reports/model_rolling.md`, `reports/prospect_model.md` |
| Résultats — machine | `reports/rolling_metrics.json`, `reports/prospect_metrics.json` |
| Cycle de vie, ré-entraînement, déploiement | `docs/model-lifecycle.md` |
| Contrats de l'API | `model_results_api/README.md` |
