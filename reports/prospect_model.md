# Prospect model — block 2

Label: the client bought an article they had NEVER bought in the snapshot window.
Features come from the snapshot window only, so train and test see the same shapes.

## Building windows
| Role | Features | Labels | Clients |
| --- | --- | --- | --- |
| train | 2024-H1 | 2024-H2 | 1,619 |
| train | 2024-H2 | 2025-H1 | 1,659 |
| embargo (dropped) | 2025-H1 | 2025-H2 | - |
| test | 2025-H2 | 2026-H1 | 1,683 |

- 2024-H1 -> 2024-H2: 833,253 rows, 40,799 positives
- 2024-H2 -> 2025-H1: 806,908 rows, 39,555 positives
## 2 training windows, 20 negatives per positive
- rows: 1,640,161   positives: 80,354 (4.90%)

- model saved to data/prospect_model.cbm

## Held-out evaluation — features 2025-H2, labels 2026-H1

| ranker | precision@5 | recall@5 | articles |
| --- | --- | --- | --- |
| model | 0.1298 | 0.0810 | 1,649 |
| heuristic | 0.0917 | 0.0359 | 1,649 |
| client size | 0.0446 | 0.0291 | 1,649 |
| client breadth | 0.0787 | 0.0396 | 1,649 |
| random | 0.0055 | 0.0019 | 1,649 |

AUC on a sampled held-out set: 0.8748

## Feature importance (top 12)

| feature | importance |
| --- | --- |
| client_n_articles | 22.43 |
| cosine_basket_sim | 16.02 |
| knn_buyers | 11.78 |
| client_n_families | 7.72 |
| client_mean_ca_per_article | 5.31 |
| brand_ca_share | 4.94 |
| client_n_months | 4.32 |
| article_n_buyers | 4.30 |
| client_ca | 3.64 |
| lift_weighted_overlap | 3.03 |
| enseigne | 2.63 |
| brand_n_held | 2.34 |

## Verdict: model beats the heuristic