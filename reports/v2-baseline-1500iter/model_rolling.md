# Rolling-snapshot model

Generated 2026-08-12 00:38. Lookback 12 months, horizon 3 months.

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
| train | 0.807 |
| validation | 0.781 |
| test | 0.775 |

Train − test gap: **+0.032**.

> The v1 model scored AUC 0.762 on the January–May annual question. **That number is not comparable to the test AUC above.** A 90-day horizon has a lower base rate and is a harder problem. The comparable figure is the continuity benchmark below.

## Lift by decile (test)

| Decile | Purchase rate | Lift |
| --- | --- | --- |
| 1 | 1.1% | ×0.10 |
| 2 | 2.6% | ×0.23 |
| 3 | 3.8% | ×0.33 |
| 4 | 5.2% | ×0.45 |
| 5 | 6.5% | ×0.58 |
| 6 | 8.6% | ×0.76 |
| 7 | 11.3% | ×1.00 |
| 8 | 14.8% | ×1.30 |
| 9 | 20.7% | ×1.82 |
| 10 | 39.0% | ×3.43 |

Base rate: 11.4%.

## Precision at the top 30 per vendor (test)

- model: **62.2%**
- by past revenue alone: 31.9%

## Per-snapshot test AUC

| Snapshot | Rows | Positive rate | AUC |
| --- | --- | --- | --- |
| 2026-01 | 516,268 | 11.08% | 0.776 |
| 2026-02 | 511,291 | 11.36% | 0.774 |
| 2026-03 | 507,732 | 11.38% | 0.775 |
| 2026-04 | 515,891 | 11.67% | 0.776 |

Stability across snapshots matters more than the pooled number: a model that only works in one month is not deployable.

## Feature importance (top 12)

| Feature | Importance |
| --- | --- |
| client_days_since_last | 18.5 |
| article_n_clients | 8.9 |
| enseigne | 8.6 |
| client_n_orders | 8.2 |
| client_amount | 6.8 |
| client_n_articles | 6.6 |
| article_amount | 6.3 |
| article_repurchase_share | 5.6 |
| client_active_months | 5.2 |
| qty_sum | 4.4 |
| vendor_code | 3.4 |
| days_since_last | 3.0 |
