# Rolling-snapshot model

Generated 2026-08-12 06:56. Lookback 12 months, horizon 3 months.

| Split | Snapshots | Range |
| --- | --- | --- |
| train | 8 | 2025-01 … 2025-08 |
| validation | 2 | 2025-09 … 2025-10 |
| embargo (dropped) | 2 | 2025-11 … 2025-12 |
| test | 4 | 2026-01 … 2026-04 |

| Split | Rows | Positive rate |
| --- | --- | --- |
| train | 4,021,611 | 11.17% |
| validation | 1,022,035 | 11.40% |
| test | 2,051,182 | 11.37% |

## AUC

| Split | AUC |
| --- | --- |
| train | 0.833 |
| validation | 0.788 |
| test | 0.771 |

Train − test gap: **+0.062**.

Iteration cap 6000, early stopping 200, best iteration **5970**. Early stopping fired, so validation AUC had stopped improving — this is a converged result.

> The v1 model scored AUC 0.762 on the January–May annual question. **That number is not comparable to the test AUC above.** A 90-day horizon has a lower base rate and is a harder problem. The comparable figure is the continuity benchmark below.

## Lift by decile (test)

| Decile | Purchase rate | Lift |
| --- | --- | --- |
| 1 | 1.2% | ×0.11 |
| 2 | 2.6% | ×0.23 |
| 3 | 3.9% | ×0.34 |
| 4 | 5.1% | ×0.45 |
| 5 | 6.7% | ×0.59 |
| 6 | 8.7% | ×0.77 |
| 7 | 11.6% | ×1.02 |
| 8 | 15.0% | ×1.32 |
| 9 | 20.6% | ×1.81 |
| 10 | 38.3% | ×3.37 |

Base rate: 11.4%.

## Precision at the top 30 per vendor (test)

- model: **61.7%**
- by past revenue alone: 31.9%

## Per-snapshot test AUC

| Snapshot | Rows | Positive rate | AUC |
| --- | --- | --- | --- |
| 2026-01 | 516,268 | 11.08% | 0.773 |
| 2026-02 | 511,291 | 11.36% | 0.770 |
| 2026-03 | 507,732 | 11.38% | 0.770 |
| 2026-04 | 515,891 | 11.67% | 0.772 |

Stability across snapshots matters more than the pooled number: a model that only works in one month is not deployable.

## Feature importance (top 12)

| Feature | Importance |
| --- | --- |
| client_days_since_last | 14.6 |
| client_n_orders | 9.3 |
| article_n_clients | 8.7 |
| enseigne | 8.5 |
| client_n_articles | 8.2 |
| client_amount | 8.0 |
| article_amount | 6.4 |
| article_repurchase_share | 5.7 |
| client_active_months | 4.2 |
| qty_sum | 4.0 |
| vendor_code | 3.9 |
| days_since_last | 2.6 |
