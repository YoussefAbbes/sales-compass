# The January-May blind spot, measured

Pairs active in January-May of the feature year, and whether they came back in the label year — as the v1 basis could see it (January-May only) versus the full twelve months the NAV ledger now covers.

| Feature year | Label year | Pairs | Rate Jan-May | Rate full year | Understated by |
| --- | --- | --- | --- | --- | --- |
| 2024 | 2025 | 232,101 | 19.8% | 31.0% | ×1.57 |
| 2025 | 2026 * | 251,760 | 18.2% | 20.5% | ×1.13 |

\* 2026 is only observed to 30 June, so its full-year figure is itself a lower bound. The 2024->2025 row is the complete measurement.

## What this means for the reported numbers

- Observed repurchase rate, January-May only: **19.8%**
- True repurchase rate over the full year: **31.0%**
- The v1 label understated repurchase by a factor of **×1.57**.

Consequence for training, which matters more than the reported rate: pairs that did repurchase outside January-May were labelled 0 and taught the model that their pattern does not lead to a purchase. That is label noise in the target, and no amount of feature work or tuning could have corrected it.