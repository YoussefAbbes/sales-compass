# Rolling-snapshot model

Generated 2026-08-12 11:18. Lookback 12 months, horizon 3 months.

| Split | Snapshots | Range |
| --- | --- | --- |
| train | 6 | 2025-01 … 2025-06 |
| embargo train→val (dropped) | 2 | 2025-07 … 2025-08 |
| validation | 2 | 2025-09 … 2025-10 |
| embargo val→test (dropped) | 2 | 2025-11 … 2025-12 |
| test | 4 | 2026-01 … 2026-04 |

| Split | Rows | Positive rate |
| --- | --- | --- |
| train | 2,996,823 | 11.30% |
| validation | 1,022,035 | 11.40% |
| test | 2,051,182 | 11.37% |

## AUC

| Split | AUC |
| --- | --- |
| train | 0.811 |
| validation | 0.771 |
| test | 0.774 |

Train − test gap: **+0.037**.

Iteration cap 6000, early stopping 200, best iteration **1345**. Early stopping fired, so validation AUC had stopped improving — this is a converged result.

> The v1 model scored AUC 0.762 on the January-May annual question. **That number is not comparable to the test AUC above.** A 90-day horizon has a lower base rate and is a harder problem, so a higher number here is not evidence of a better model on the v1 question.
>
> The only fair like-for-like comparison would be a continuity benchmark: v1's annual January-May question re-run on the NAV data with full-year labels. **That has not been run.** Until it is, treat 0.762 and the figure above as answers to two different questions, and do not quote a delta between them.

## Lift by decile (test)

| Decile | Purchase rate | Lift |
| --- | --- | --- |
| 1 | 1.2% | ×0.10 |
| 2 | 2.7% | ×0.23 |
| 3 | 3.9% | ×0.34 |
| 4 | 5.2% | ×0.46 |
| 5 | 6.7% | ×0.58 |
| 6 | 8.5% | ×0.75 |
| 7 | 11.2% | ×0.98 |
| 8 | 14.6% | ×1.29 |
| 9 | 20.5% | ×1.81 |
| 10 | 39.3% | ×3.45 |

Base rate: 11.4%.

## Precision at the top 30 per vendor (test)

- model: **61.9%**
- by past revenue alone: 31.9%

## Per-snapshot test AUC

| Snapshot | Rows | Positive rate | AUC |
| --- | --- | --- | --- |
| 2026-01 | 516,268 | 11.08% | 0.775 |
| 2026-02 | 511,291 | 11.36% | 0.773 |
| 2026-03 | 507,732 | 11.38% | 0.774 |
| 2026-04 | 515,891 | 11.67% | 0.775 |

Stability across snapshots matters more than the pooled number: a model that only works in one month is not deployable.

## Feature importance (top 12)

| Feature | Importance |
| --- | --- |
| client_days_since_last | 18.0 |
| client_n_orders | 9.8 |
| article_n_clients | 9.3 |
| enseigne | 9.0 |
| client_n_articles | 7.6 |
| client_amount | 6.5 |
| article_repurchase_share | 5.9 |
| client_active_months | 5.1 |
| article_amount | 4.7 |
| qty_sum | 4.4 |
| vendor_code | 3.4 |
| days_since_last | 2.9 |
