# Vendor Recommendation System Delivery Plan

## 1. Current direction

The project will begin with an explainable hybrid recommendation system:

- SQL-based reorder and eligibility logic.
- Purchase history and favorites as the primary signals.
- Optional item similarity or `implicit` ALS for candidate discovery.
- `CatBoostClassifier` as the first trained purchase-prediction model.
- Daily batch scoring written back to SQL.
- Existing vendor interface extended with recommendations and feedback actions.
- No generative language model or deep-learning dependency in the MVP.

## 2. Delivery principles

- Prove value with a simple baseline before adding model complexity.
- Prevent future-data leakage by using dated historical snapshots.
- Keep recommendations explainable and auditable.
- Apply availability, pricing, permission, compatibility, and contact rules after scoring.
- Capture vendor actions from the first pilot.
- Measure incremental business impact, not only offline accuracy.
- Release in small stages with explicit acceptance criteria.

## 3. Phase 0 — Business and data discovery

### Tasks

- [ ] Confirm the business objective and priority KPI.
- [ ] Define the prediction horizon for each major article category.
- [ ] Inventory source tables, relationships, keys, and statuses.
- [ ] Count clients, articles, completed order lines, favorites, and history length.
- [ ] Identify vendor-client ownership and territory rules.
- [ ] Document stock, pricing, contract, compatibility, and credit restrictions.
- [ ] Confirm how cancellations, returns, samples, and quotations are represented.
- [ ] Assess whether vendor contacts and outcomes are currently recorded.
- [ ] Profile missing values, duplicates, invalid timestamps, and orphan records.
- [ ] Map replacement and equivalent SKUs.

### Deliverables

- Source-to-feature data map.
- Data-quality report.
- Agreed prediction horizon and business KPI.
- List of commercial filters and permissions.
- Initial volume and infrastructure estimate.

### Exit criteria

- Client, article, purchase, favorite, and vendor ownership data can be joined reliably.
- Completed purchases can be separated from cancellations, returns, samples, and quotations.
- The first recommendation horizon and success metric are agreed.

## 4. Phase 1 — Explainable SQL baseline

### Tasks

- [ ] Build clean purchase-history views.
- [ ] Calculate per-client and per-article purchase summaries.
- [ ] Calculate client-article reorder intervals, regularity, and typical quantity.
- [ ] Implement the overdue ratio and sparse-history fallbacks.
- [ ] Add active-favorite candidates with time decay.
- [ ] Add simple category popularity and item co-purchase candidates.
- [ ] Implement stock, active-status, permission, compatibility, and recent-purchase filters.
- [ ] Create deterministic reason codes and explanations.
- [ ] Create recommendation and feedback tables.
- [ ] Generate the first daily ranked recommendations.

### Deliverables

- Reorder baseline.
- Favorites baseline.
- Simple cross-sell baseline.
- Populated `article_recommendations` table.
- Baseline offline metrics.

### Exit criteria

- Every displayed recommendation is eligible and has a factual explanation.
- Historical backtesting can reproduce recommendations for a past date.
- Baseline results are available for comparison with later models.

## 5. Phase 2 — Vendor interface and feedback loop

### Tasks

- [ ] Add a **Clients to contact today** view.
- [ ] Add client-first recommended article sections.
- [ ] Add an article-first list of likely clients.
- [ ] Show one to three recommendations per client with reasons and confidence.
- [ ] Add actions for contact, quotation, snooze, and dismiss.
- [ ] Add structured dismissal and refusal reasons.
- [ ] Automatically link resulting orders when possible.
- [ ] Apply contact-frequency limits and recent-contact penalties.
- [ ] Enforce vendor and territory access control.

### Deliverables

- Vendor opportunity queue.
- Client and article recommendation views.
- Feedback event capture.
- Audit trail for recommendations and actions.

### Exit criteria

- Vendors can act on a recommendation without leaving the normal workflow.
- Every action is recorded with vendor, recommendation, timestamp, and outcome.
- Unauthorized clients and articles cannot appear.

## 6. Phase 3 — Historical training dataset

### Tasks

- [ ] Choose weekly or monthly snapshot frequency.
- [ ] Build candidate sets as they would have existed at each snapshot date.
- [ ] Calculate all features using information strictly before the snapshot.
- [ ] Label purchases occurring within the selected future horizon.
- [ ] Create meaningful negative candidates rather than random impossible products.
- [ ] Create chronological training, validation, and untouched test periods.
- [ ] Version the dataset definition and feature calculations.
- [ ] Add automated checks for future-data leakage.

### Deliverables

- Versioned `client_article_features` dataset.
- Training labels and candidate-source indicators.
- Chronological train, validation, and test partitions.
- Dataset quality and leakage report.

### Exit criteria

- A historical snapshot can be recreated deterministically.
- No feature uses data later than its snapshot date.
- Positive and negative label definitions reflect eligible purchasing opportunities.

## 7. Phase 4 — CatBoost purchase model

### Tasks

- [ ] Train a simple logistic-regression benchmark.
- [ ] Train `CatBoostClassifier` on the snapshot dataset.
- [ ] Tune only after establishing a stable baseline.
- [ ] Compare against reorder-only, favorite-only, and popularity baselines.
- [ ] Evaluate by client segment, article category, and history density.
- [ ] Inspect feature importance and recommendation examples.
- [ ] Calibrate probabilities if they will be used for expected-value calculations.
- [ ] Define confidence thresholds and suppress weak recommendations.
- [ ] Version and store the approved model and feature definition.
- [ ] Add daily scoring and write results back to SQL.

### Deliverables

- Trained and versioned CatBoost model.
- Offline comparison report.
- Scoring pipeline.
- Model monitoring baseline.

### Exit criteria

- The model improves top-K recommendation quality over the SQL baseline on the untouched time period.
- Recommendations remain commercially valid after filtering.
- Probability or confidence labels are sufficiently calibrated for the interface.
- Scoring completes within the required batch window.

## 8. Phase 5 — Collaborative candidate generation

This phase should proceed only when purchase data has meaningful overlap between clients and articles.

### Tasks

- [ ] Build the client-article interaction matrix.
- [ ] Define confidence weights for completed purchases and favorites.
- [ ] Train and evaluate `implicit` ALS.
- [ ] Compare ALS, item-to-item similarity, and co-purchase rules.
- [ ] Add the strongest collaborative candidates to the candidate union.
- [ ] Add collaborative scores as CatBoost features.
- [ ] Measure candidate Recall@100 before evaluating final Recall@K.
- [ ] Validate cold-start behavior for new clients and articles.

### Deliverables

- Collaborative candidate service or batch job.
- Candidate coverage comparison.
- Updated ranking model using collaborative features.

### Exit criteria

- Collaborative candidates add relevant new articles beyond reorder and favorites.
- Candidate recall improves without unacceptable popularity bias.
- Sparse and new clients retain safe fallback recommendations.

## 9. Phase 6 — Controlled vendor pilot

### Tasks

- [ ] Select a representative vendor pilot group.
- [ ] Define a comparison or holdout group.
- [ ] Train vendors on reasons, confidence, and feedback actions.
- [ ] Monitor recommendation availability and data freshness daily.
- [ ] Review false positives, dismissals, and missing opportunities weekly.
- [ ] Measure recommendation usage and vendor time saved.
- [ ] Measure incremental conversion, revenue, and margin.
- [ ] Monitor returns, complaints, discounts, and contact fatigue.
- [ ] Collect qualitative vendor and manager feedback.

### Deliverables

- Pilot dashboard.
- Weekly quality and business-impact review.
- Go, revise, or stop recommendation for broader rollout.

### Exit criteria

- The pilot demonstrates positive incremental value or a clear path to improvement.
- No unresolved permission, pricing, stock, or contact-safety failures remain.
- Vendors understand and use the recommendations.

## 10. Phase 7 — Production hardening and rollout

### Tasks

- [ ] Automate feature generation, scoring, and expiration.
- [ ] Add pipeline failure alerts and stale-data detection.
- [ ] Monitor feature drift, prediction drift, and recommendation coverage.
- [ ] Schedule periodic retraining based on business cadence.
- [ ] Document rollback to the SQL baseline.
- [ ] Add model-version and data-version auditability.
- [ ] Load-test the vendor interface and recommendation endpoints.
- [ ] Roll out by vendor team, region, or category.
- [ ] Establish ownership for data quality, model quality, and business rules.

### Deliverables

- Reliable production pipeline.
- Monitoring and alerting.
- Rollback procedure.
- Operations and retraining guide.

### Exit criteria

- Production recommendations meet freshness and availability targets.
- Failures degrade safely to the rule-based baseline.
- Model, data, and business-rule owners are assigned.

## 11. Later enhancements

These items are intentionally outside the MVP:

- [ ] Survival or time-to-event model for precise reorder timing.
- [ ] Learning-to-rank model optimized for top-three or top-five recommendations.
- [ ] Promotion and price-response modeling.
- [ ] Client churn and reactivation model.
- [ ] Contact-channel and contact-time recommendations.
- [ ] Bundle and substitute recommendations.
- [ ] Text embeddings for new-article descriptions.
- [ ] Language-model assistant for summaries and draft messages.
- [ ] Uplift modeling to predict which clients buy because the vendor contacts them.
- [ ] Two-tower deep retrieval only if scale requires it.

## 12. Testing checklist

### Data tests

- [ ] No future timestamps in features.
- [ ] No duplicate recommendations for the same client and article.
- [ ] Returns and cancellations are excluded from positive labels.
- [ ] Replaced SKUs are handled consistently.
- [ ] Inventory and price timestamps are valid.

### Recommendation tests

- [ ] Out-of-stock and discontinued articles are excluded.
- [ ] Recently purchased equivalents are excluded.
- [ ] Client-specific catalogue and compatibility rules are respected.
- [ ] Old favorites decay appropriately.
- [ ] Recent refusals and contacts reduce priority.
- [ ] Low-confidence recommendations are hidden or clearly labeled.

### Interface and security tests

- [ ] Vendors only see authorized clients.
- [ ] Feedback actions are idempotent and auditable.
- [ ] Recommendations expire and refresh correctly.
- [ ] The interface handles empty, loading, stale, and error states.
- [ ] No other client's identity is exposed in explanations.

## 13. Immediate next actions

1. [ ] Export or document the anonymized SQL schema and table relationships.
2. [ ] Record the number of clients, articles, order lines, favorites, and years of history.
3. [ ] Identify the main article categories and their typical reorder cycles.
4. [ ] Confirm how vendors, contacts, quotations, returns, and stock are represented.
5. [ ] Select the first business KPI and pilot vendor group.
6. [ ] Complete Phase 0 before selecting final model parameters.

