# Champion vs. Candidate Governance Report

Generated: 2026-09-02T20:07:19.910430+00:00

- **Champion**: `fraud-risk-lightgbm-v1` (CHAMPION)
- **Candidate**: `mock-candidate-better-v1` (CANDIDATE)
- **Evaluation dataset**: validation

## Recommendation: **PROMOTE**

All promotion gates passed.

## Promotion Gates

| Gate | Result | Explanation |
|---|---|---|
| required_metric_availability | PASS | Both champion and candidate have valid PR-AUC/ROC-AUC on the evaluation dataset. |
| feature_schema_compatibility | PASS | Feature schema versions match (phase1-v1). |
| candidate_quality_threshold | PASS | Candidate PR-AUC (0.8700) meets the project minimum (0.3). |
| no_unacceptable_degradation | PASS | Candidate PR-AUC (0.8700) is at or above the champion's (0.6147), or within the negligible 0.005 review-free tolerance. |
| operational_compatibility | PASS | Legitimate-customer blocked rate changes by -0.0374% under the candidate — within the 0.50% tolerance. |
| decision_policy_compatibility | PASS | Both models are evaluated against the same frozen decision policy ('balanced'). |

## Ranking Metrics

| Metric | Champion | Candidate | Delta |
|---|---|---|---|
| PR-AUC | 0.6147 | 0.8700 | +0.2553 |
| ROC-AUC | 0.9253 | 0.9957 | +0.0705 |