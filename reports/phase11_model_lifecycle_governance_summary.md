# Phase 11 Summary — Model Lifecycle & Performance Governance

This report documents a lightweight local framework for governing model
versions and deciding whether a candidate model should replace the
current champion. All numbers are from real runs — a fast synthetic test
suite, the real champion registered from actual Phase 4/5 evaluation
data, a real demonstration comparing the real champion against two
clearly-labeled mock candidates, and a real Docker build/run with the new
endpoint validated live.

**This is a lightweight local governance framework — not a full
enterprise model registry, not an automated deployment system, and not a
claim that any promotion happens without explicit human approval.**

## 1. Phase objective

Answer "if a new model is trained or proposed, how do we decide whether
it is good enough to become the approved production candidate?" — a
governance and evaluation capability, never automatic model replacement.

## 2. Why governance is needed

Phases 1-10 built a rigorous, leakage-safe, reproducible pipeline for
one model. Nothing in that pipeline answers "what happens when someone
proposes a second one?" — without an explicit framework, a new model
could silently replace the champion with no comparison, no audit trail,
and no way to undo the change. Phase 11 exists specifically to close that
gap: every model this project has ever trained or will train gets a
documented identity, every champion/candidate comparison is auditable,
and no promotion happens without an explicit, attributable human action.

## 3. Model registry design

```
src/governance/
    __init__.py
    metadata.py       - metadata construction/validation, real SHA-256 hashing
    registry.py         - ModelRegistry: JSON-file-backed, explicit lifecycle actions
    evaluation.py          - champion vs. candidate comparison (reuses Phase 2A/5 functions)
    compatibility.py         - schema/policy compatibility checks
    gates.py                   - 6 explainable promotion gates + aggregation
    report.py                    - governance report generation
```

`artifacts/models/registry.json` — a single JSON file, human-readable,
tracking every registered model's metadata and status
(`CANDIDATE`/`CHAMPION`/`REJECTED`/`RETIRED`). This is explicitly not a
claim of a full enterprise model registry: no database, no access
control, no distributed storage, no MLflow/cloud integration. The core
governance guarantee is structural, not procedural: `champion_model_id`
in the registry data can ONLY change inside `ModelRegistry.promote()` or
`ModelRegistry.rollback()` — no other method in the entire codebase
writes to that field, and both require an explicit `approved_by` and
`reason` string supplied by the caller.

## 4. Model metadata

Required fields (`REQUIRED_METADATA_FIELDS` in `src/governance/metadata.py`):
`model_id`, `model_version`, `model_type`, `status`, `created_at`,
`feature_schema_version`, `training_data_reference`,
`validation_data_reference`, `evaluation_metrics`, `artifact_reference`,
`decision_policy_version`. Built from actual repository information —
never invented. Genuinely unrecorded historical fields (e.g. this project
never logged an exact training wall-clock duration) are marked with an
explicit `UNAVAILABLE` sentinel, distinguishable from a real zero or
empty value — verified directly by
`test_unavailable_historical_fields_marked_explicitly`.

## 5. Existing champion registration

`python -m src.run_phase11_register_champion` — registered
`fraud-risk-lightgbm-v1` as the initial `CHAMPION`, using:

- Real evaluation metrics: Phase 4's saved validation/test ranking
  metrics and budget tables, Phase 5's saved decision-policy evaluation —
  loaded from `data/interim/phase4_metrics.json` /
  `data/interim/phase5_metrics.json`, not recomputed.
- Real hyperparameters: `src/models/lightgbm_model.py`'s actual
  `DEFAULT_PARAMS` (`random_state=42`, etc.).
- A real SHA-256 hash of the actual model bundle file:
  `bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342`
  (`data/interim/phase6_model_bundle.pkl`) — computed by
  `compute_artifact_sha256`, not fabricated.
- Real data-partition references: exact row-range boundaries from
  `data/interim/phase1_split_metadata.json`.

No retraining or re-evaluation occurred — this is metadata registration
of an already-approved model.

## 6. Candidate workflow

`registry.register_candidate(metadata)` — requires `status="CANDIDATE"`
metadata; duplicate registration without `overwrite=True` raises
(`RegistryError`), preventing silent metadata clobbering. A candidate's
mere registration has zero effect on the champion — verified directly
(`test_evaluation_alone_does_not_change_champion`).

## 7. Champion/challenger evaluation methodology

`src/governance/evaluation.py` reuses the project's already-approved
evaluation functions directly — `compute_ranking_metrics` (Phase 2A),
`compute_threshold_metrics` (Phase 2A), `evaluate_policy` (Phase 5) —
rather than reimplementing metric computation. `compare_champion_vs_candidate`
requires an explicit `dataset_label` for every comparison, making the
evaluation partition auditable in every report rather than implicit.
Dataset discipline: both the real champion registration (§5) and the
demonstration (§10) use VALIDATION for gate decisions; the one-time Phase
4 TEST evaluation is carried in champion metadata purely as a historical
record, explicitly labeled "never used again as a tuning or
promotion-gate signal" — this project's test set is not touched again
here, consistent with every prior phase's discipline.

## 8. Promotion gates

Six gates (`src/governance/gates.py`), each returning exactly one of
`PASS`/`FAIL`/`REQUIRES_REVIEW` with a human-readable explanation:

| Gate | Checks |
|---|---|
| `required_metric_availability` | Both models have valid, non-NaN PR-AUC/ROC-AUC |
| `feature_schema_compatibility` | Champion/candidate feature schema versions match |
| `candidate_quality_threshold` | Candidate PR-AUC >= `MIN_CANDIDATE_PR_AUC` (0.30) |
| `no_unacceptable_degradation` | Candidate PR-AUC not more than `MAX_ACCEPTABLE_PR_AUC_DEGRADATION` (0.02) below champion |
| `operational_compatibility` | Legitimate-customer blocked-rate increase within `MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE` (0.5pp) |
| `decision_policy_compatibility` | Champion/candidate reference the same frozen Phase 5 policy version |

All numeric thresholds are explicitly documented in the module docstring
as project governance thresholds chosen for this repository, not
universal fraud-industry standards. Aggregation (`aggregate_gate_results`)
is a stated, simple rule — any `FAIL` -> `REJECT`; no `FAIL` but any
`REQUIRES_REVIEW` -> `REQUIRES_REVIEW`; all `PASS` -> `PROMOTE` — never a
hidden weighted score (the explanation always names the actual gate(s)
responsible, confirmed by test).

## 9. Compatibility checks

`src/governance/compatibility.py`: `check_schema_compatibility` and
`check_decision_policy_compatibility`, both feeding directly into gates 2
and 6 above. A real, deliberate mismatch (candidate declaring an
incompatible `feature_schema_version`) was tested end-to-end and
correctly produces `REJECT` regardless of how good the candidate's
metrics look (`test_schema_mismatch_fails_promotion_gate`) — schema
compatibility is a hard gate, not a soft signal.

## 10. Example governance recommendation (real run)

From `scripts/demo_model_governance.py`, comparing the real champion
against two mock candidates on real validation labels:

**Mock "worse" candidate** (champion scores + Gaussian noise, PR-AUC
0.5617 vs. champion's 0.6147): REJECT — failed
`no_unacceptable_degradation` ("Candidate PR-AUC (0.5617) is 0.0530 below
the champion's (0.6147), exceeding the 0.02 maximum tolerated
degradation"); all five other gates passed.

**Mock "better" candidate** (a synthetic, explicitly-labeled label-informed
blend, PR-AUC 0.8700): PROMOTE — all six gates passed.

Full reports saved to `reports/model_governance/champion_vs_mock_worse_candidate_report.json`
and `.../champion_vs_mock_better_candidate_report.{json,md}`.

## 11. Promotion workflow

```
Register Candidate -> Evaluate -> Generate Report -> Recommendation -> Explicit Approval -> Update Registry
```

`ModelRegistry.promote(candidate_model_id, approved_by, reason)` — real
run: `registry.promote("mock-candidate-better-v1", approved_by="demo-script
(simulated human reviewer)", reason="Demonstration: mock candidate passed
all promotion gates.")`. Result: the previous champion's status flips to
`RETIRED` (never deleted — `retirement_history` records when/why/by whom),
the candidate's status flips to `CHAMPION`, and `promotion_history`
records the same. Verified structurally that report generation itself
never touches the registry (§6) — only this explicit call does.

## 12. Rollback strategy

`ModelRegistry.rollback(target_model_id, approved_by, reason)` — only
works on a currently-`RETIRED` model (never deleted by `promote()`, so
this is always possible for any prior champion). Real run: after promoting
the mock "better" candidate, `registry.rollback("fraud-risk-lightgbm-v1",
approved_by="demo-script (simulated human reviewer)", reason=
"Demonstration: rolling back to the real champion after the mock
promotion.")` correctly restored the real champion, retiring the mock
candidate in turn, with `rollback_history` recording the action.
Limitation, documented plainly: this lightweight local implementation
has no automatic trigger, no canary/gradual rollout, and no
production-traffic-based health check — rollback is a manual, explicit
action based on external judgment (e.g. a human noticing a problem), not
an automated safety system.

## 13. Version compatibility checks

Covered in §9 — schema and policy compatibility are both hard gates (2
and 6), each independently testable and each verified to correctly block
promotion on a real mismatch.

## 14. Drift-to-governance relationship

Structurally enforced separation, not just a design intention: a
dedicated test (`test_drift_module_never_imports_governance_registry`)
parses every file in `src/drift/` with Python's `ast` module and asserts
none of them import anything from `src/governance/` — this is checked by
walking the actual import statements in the actual source files, not by
convention alone. `ModelRegistry.promote()`/`rollback()` require an
`approved_by`/`reason` string signature that a drift severity level (e.g.
"HIGH_DRIFT") cannot satisfy on its own — there is no code path by which
a drift report could programmatically supply real values for those
required human-facing fields. Drift detection may inform a human that
review is warranted (the conceptual HIGH_DRIFT -> MODEL_REVIEW_RECOMMENDED
signal from the brief), but nothing in this codebase wires that
information to an automatic registry action.

## 15. Lifecycle status reporting

`GET /model-governance/summary` — read-only (verified:
`test_governance_summary_never_promotes_or_mutates_champion` calls it 5
times and confirms the champion never changes). Real response (live
Docker container):

```json
{"available":true,"champion_model_id":"fraud-risk-lightgbm-v1","champion_version":"v1",
 "champion_status":"CHAMPION","n_registered_models":1,"n_candidates":0,
 "n_rejected":0,"n_retired":0,"registry_version":"v1",
 "last_updated_at":"2026-09-02T20:03:28.816448+00:00"}
```

No route in `src/api/main.py` calls `promote()`, `rollback()`, or
`register_candidate()` — the API layer only ever reads the registry.

## 16. Non-interference validation — the critical result

Two dedicated tests directly address the brief's core requirement:

- `test_governance_registry_does_not_change_predict_behavior`: the same
  transaction, scored once through an API with no governance registry
  loaded, and once through an API with a registry loaded AND a candidate
  registered in between predict calls — risk scores match to `<1e-9`,
  decisions identical.
- `test_governance_summary_never_promotes_or_mutates_champion`: confirmed
  above (§15).

Both passed. Model governance genuinely does not touch the inference
path.

## 17. Docker compatibility — REAL, EXECUTED

Rebuilt the Phase 8-10 image with Phase 11's new files (`src/governance/`,
the registry artifact) and validated live:

```
$ docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest -t fraud-risk-api:local .
Successfully built a25c6d594edd

$ docker run -d --name fraud-risk-api-test -p 8123:8000 fraud-risk-api:local
$ docker inspect --format='{{.State.Health.Status}}' fraud-risk-api-test
healthy   # after 5 seconds

$ curl http://127.0.0.1:8123/model-governance/summary
{"available":true,"champion_model_id":"fraud-risk-lightgbm-v1", ...}

$ curl http://127.0.0.1:8123/drift/summary       # still works
$ curl http://127.0.0.1:8123/monitoring/summary   # still works, request counts include the new route
```

`/health` and `/predict` unaffected; the new governance endpoint confirmed
working alongside Phases 9/10's endpoints, all correctly tracked by
Phase 9's request-metrics middleware. `requirements.txt` did not change
for this phase (only the standard library `hashlib`/`json` are used).

## Demonstration results

`python scripts/demo_model_governance.py` — full real run, using a
throwaway demo registry (`artifacts/models/demo_registry.json`) so none
of its promote/rollback actions ever touched the real registry (confirmed:
the real `artifacts/models/registry.json` still shows exactly 1 model,
the real champion, after running the demo). Registered the real champion
(real metrics, real hash), registered 2 explicitly-labeled MOCK candidates
(synthetic scores, never claimed to be a real competing fraud model),
evaluated both on real validation labels, generated two full governance
reports (§10), explicitly promoted the passing mock candidate, then
explicitly rolled back to the real champion — end to end, exit code 0.

## Test suite results

**290/290 Python tests passed** (248 from Phases 0-10 + 37 unit tests for
metadata/registry/evaluation/gates/compatibility/promotion/rollback/drift-separation
+ 5 FastAPI integration tests including the two non-interference tests in
§16). All prior tests re-run unchanged and still passing.

## Errors encountered and fixes

1. **Floating-point boundary bug in the degradation gate.** A real test
   (`test_gate_degradation_boundary_behavior`) placed a candidate's PR-AUC
   exactly at the `MAX_ACCEPTABLE_PR_AUC_DEGRADATION` boundary and found
   the gate returned `FAIL` instead of the intended `REQUIRES_REVIEW`/`PASS`
   boundary behavior. Root cause: `0.50 - 0.02` does not evaluate to
   exactly `-0.02` in binary floating point (`-0.020000000000000018`), so
   the boundary comparison failed by a sub-1e-9 rounding error. This is
   not a hypothetical edge case — any candidate landing near a configured
   threshold could be judged by which way an arbitrary rounding error
   happened to fall, rather than by the intended threshold. Fixed by
   adding a documented `1e-9` epsilon to both boundary comparisons in
   `gate_no_unacceptable_degradation`, verified fixed by re-running the
   exact failing test.

## Known limitations

- **Local JSON file, not a database** — no concurrent-write protection, no
  access control, no audit-log tamper-resistance beyond what a plain file
  provides.
- **No automated canary/gradual rollout** — promotion is immediate and
  total (all future traffic uses the new champion) once explicitly
  approved; there is no staged-traffic mechanism.
- **No automatic rollback trigger** — a human must notice a problem and
  explicitly call `rollback()`; nothing in this project monitors the live
  champion's real-world performance and rolls back on its own.
- **Compatibility checks are metadata-based, not artifact-based** — schema
  and policy compatibility are checked by comparing the declared
  `feature_schema_version`/`decision_policy_version` strings in metadata,
  not by actually loading and diffing the underlying `FeaturePipeline`/
  `LightGBMPreprocessor` objects; a mislabeled candidate could in
  principle slip past this check.
- **Governance is entirely separate from real inference artifact
  swapping** — promoting a model in the registry does NOT automatically
  make `src/api/dependencies.py`'s `RiskDecisionEngine` use the new
  model; wiring an approved promotion into the live serving path is
  future work, explicitly out of scope here.

## What was intentionally NOT implemented

Automatic retraining, automatic model promotion, automatic rollback,
cloud model registry, MLflow server deployment, Kubernetes, Kafka,
database infrastructure, CI/CD redesign, automatic threshold
optimization, new fraud-model research, feature-engineering redesign —
all reserved for later phases, per the Phase 11 scope restrictions.
