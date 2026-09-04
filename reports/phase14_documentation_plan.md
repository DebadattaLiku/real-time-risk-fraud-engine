# Phase 14 Documentation Plan (PLAN ONLY — no README changes yet)

This plan is the output of inspecting the actual repository state, not a
rewrite. Every number in Section 2 was read directly from a real artifact
file in this session — nothing here is copied from memory of earlier
phase reports without re-verification.

## 1. Current README Assessment

**What is already good:**
- Accurate, honest tone throughout — no inflated claims found anywhere in
  the current README (e.g. "This is a local containerized deployment for
  development/demonstration — not a cloud or production deployment").
- Each major subsystem (Docker, Monitoring, Drift, Governance, Dashboard)
  already has its own section with a real command block and a link to the
  full phase report.
- Correctly distinguishes real vs. mock results already, e.g. "the mock
  candidate's PR-AUC is never presented as a real model result" appears
  in the Dashboard section.
- `## Principles this project follows` is a strong, distinctive section —
  exactly the kind of thing that signals engineering rigor to a reviewer.

**What is missing:**
- No "what/why" framing at the top for a reader with 2-5 minutes — it
  opens directly into dataset/setup mechanics rather than a short pitch
  (what this system does, why the problem matters, headline real numbers).
- No architecture diagram beyond fragments scattered across sections
  (Docker's mini-diagram, Monitoring's mini-diagram, Drift's mini-diagram)
  — no single end-to-end picture of the full Phase 0-13 system in one
  place.
- No model performance table anywhere in the README itself (the real
  numbers exist in `data/interim/phase4_metrics.json` /
  `phase5_metrics.json` and in `reports/phase4_behavioral_features_summary.md`,
  but are not surfaced at the top level).
- No mention of the ML methodology at all in the README (LightGBM vs.
  Logistic Regression baseline, Isolation Forest rejection, behavioral
  features, chronological split discipline) — a reader has to go into
  `reports/` to learn ANY of this.
- No "How to run locally" consolidated section — Docker instructions
  exist, dashboard instructions exist, but there's no single "quickest
  path to see this working" walkthrough combining API + dashboard.
- No testing/CI section — 316 passing tests and a real CI workflow
  (`.github/workflows/tests.yml`, added in Phase 13) exist but are
  entirely unmentioned in the README.
- No table of contents — the README is long (380 lines) with no
  navigation aid.

**What is unclear:**
- The opening paragraph says "Combines a supervised fraud model, an
  anomaly detector, and behavioral/transaction risk features into a fused
  risk score" — this describes the originally planned architecture, not
  what was actually built and shipped. Phase 3 concluded Isolation Forest
  should NOT be used (confirmed: `reports/phase3_anomaly_detection_summary.md`
  states "Isolation Forest is NOT recommended for the Version 1 hybrid
  engine"), and the real champion (`fraud-risk-lightgbm-v1`) is LightGBM +
  behavioral features only, with no anomaly-detection fusion. This is the
  single most important correction needed — not because the current text
  is dishonest (Phase 3's rejection is real and well-documented
  elsewhere), but because the README's own opening line contradicts the
  project's own later, more careful finding, and a reader who stops at
  paragraph one would come away with an inaccurate picture of the actual
  system.
- "The current phase's findings" / "Do not assume later phases exist
  until their report files are present" (lines 8-10) is leftover
  scaffolding language from very early in the project's life (when it was
  genuinely being built phase-by-phase and later phases genuinely didn't
  exist yet) — no longer accurate now that all 13 phases are complete and
  documented.

**What is outdated:**
- `## Status` says "Phases 0-12 complete" — Phase 13 (engineering audit,
  approved) is not mentioned at all.
- The most-recent-report pointer says
  `reports/phase12_dashboard_summary.md` — should point to
  `reports/phase13_engineering_audit_summary.md`.

**What should be retained (unchanged in substance):**
- Every existing "Run with Docker" / "Monitoring & Observability" /
  "Drift Detection" / "Model Lifecycle & Governance" / "Dashboard" section
  — these are accurate, well-sourced, and already correctly hedge their
  claims (e.g. Drift's "Limitations" paragraph about cumulative
  behavioral features, Governance's mock-candidate warning). Phase 14
  should REORGANIZE and ADD CONTEXT around these, not rewrite their
  content.
- The `## Principles this project follows` list, verbatim.
- The Dataset section's honest explanation of why the raw CSVs aren't
  auto-downloaded (sandbox network restriction) — accurate and worth
  keeping, though it can be tightened.

## 2. Evidence Inventory

All values below were read directly from the named artifact file in this
session (not copied from a prior report from memory).

| Metric | Value | Dataset/Partition | Source Artifact | Safe to Claim? |
|---|---:|---|---|---|
| Champion model | `fraud-risk-lightgbm-v1` | — | `artifacts/models/registry.json` (`champion_model_id`) | Yes |
| Champion artifact hash | `bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342` | — | `artifacts/models/registry.json` | Yes |
| LightGBM (+behavioral) Test PR-AUC | 0.5483 (0.5482543868701517) | Final TEST (one-time) | `data/interim/phase4_metrics.json` -> `model_b_behavioral.test_ranking.pr_auc` | Yes |
| LightGBM (+behavioral) Test ROC-AUC | 0.9058 (0.9057988067195211) | Final TEST | same artifact -> `.roc_auc` | Yes |
| LightGBM (+behavioral) Val PR-AUC | 0.6147 (0.6146879274607402) | VALIDATION | same artifact -> `model_b_behavioral.val_ranking.pr_auc` | Yes |
| Recall@1% (Test) | 25.30% (0.2530003243593902) | Final TEST | same artifact -> `test_budget_table[budget=0.01].recall_at_k` | Yes |
| Recall@2% (Test) | 40.90% (0.4090171910476808) | Final TEST | same artifact -> `test_budget_table[budget=0.02]` | Yes |
| Recall@5% (Test) | 60.14% (0.6013623094388583) | Final TEST | same artifact -> `test_budget_table[budget=0.05]` | Yes |
| Precision@1% / @2% / @5% (Test) | 88.04% / 71.16% / 41.85% | Final TEST | same artifact, same rows | Yes |
| Transaction-only baseline Test PR-AUC | 0.5428 (0.5427906495270152) | Final TEST | `data/interim/phase4_metrics.json` -> `model_a_baseline.test_ranking.pr_auc` | Yes |
| Behavioral-feature improvement (Val PR-AUC delta) | +0.0055 (0.005510599775985892) | VALIDATION (decision basis) | same artifact -> `validation_pr_auc_delta_b_minus_a` | Yes — but must be framed as small/modest, matching Phase 4's own report language, not oversold |
| Isolation Forest Test PR-AUC | 0.0449 | Final TEST | `reports/phase3_anomaly_detection_summary.md` | Yes, framed as "why it was rejected," never as a used component |
| No-skill baseline PR-AUC (fraud base rate) | 0.0348 | Final TEST | same report | Yes |
| Frozen decision policy — approve threshold | 0.32061562877852007 | — | `data/interim/phase5_metrics.json` -> `policy_config.approve_threshold` | Yes |
| Frozen decision policy — block threshold | 0.631136794838037 | — | same artifact | Yes |
| Test-set decision distribution | APPROVE 97.91% / REVIEW 0.82% / BLOCK 1.27% | Final TEST | `data/interim/phase5_metrics.json` -> `test_eval.buckets` | Yes |
| Fraud captured (REVIEW+BLOCK) | 1,291 / 3,083 (41.87%) | Final TEST | same artifact | Yes |
| Fraud captured by BLOCK alone | 958 | Final TEST | same artifact | Yes |
| Fraud missed in APPROVE | 1,792 (58.1% of all fraud) | Final TEST | same artifact | Yes — this is an important HONEST limitation to keep visible, not hide |
| Precision among BLOCKED | 84.85% | Final TEST | same artifact | Yes |
| Legitimate customers blocked | 171 (0.20% of legitimate traffic) | Final TEST | same artifact | Yes |
| Offline/online behavioral feature parity | 2,997 / 3,000 transactions (99.9%) fully matching | Real-time simulation, 3,000-txn slice | `data/interim/phase6_metrics.json` -> `feature_parity` | Yes, WITH the caveat below |
| Offline/online risk-score correlation | 0.999999996 (max abs diff 0.00046) | Same simulation | `data/interim/phase6_metrics.json` -> `score_parity` | Yes |
| Offline/online decision parity | 3,000 / 3,000 (100%) | Same simulation | `data/interim/phase6_metrics.json` -> `decision_parity` | Yes |
| Simulation-subset PR-AUC/ROC-AUC | 0.5057 / 0.8930 | 3,000-txn TEST subset, NOT the full test set | `data/interim/phase6_metrics.json` -> `simulation_ranking` | Yes, but must NOT be conflated with the full-test-set 0.5483/0.9058 numbers — different, smaller population, already flagged in Phase 6's own report |
| Test suite size | 316 passed, 0 failed | — | live `pytest -q` run, this session | Yes |
| Docker build context size | ~4.29MB (Phase 8's original measurement) | — | `reports/phase8_containerization_summary.md` | Yes, but should be phrased as "Phase 8 measured," not re-verified fresh this session |
| API endpoints | 9 total (`/predict`, `/health`, `/metadata`, `/metrics`, `/monitoring/summary`, `/drift/summary`, `/drift/analyze`, `/model-governance/summary`, `/dev/reset-state`) | — | `src/api/main.py`, grepped this session | Yes |
| Drift reference partition | VALIDATION, 88,581 transactions | — | `artifacts/drift/reference_profile.json` -> `metadata` | Yes |
| Monitored drift signals | 6 numeric + 2 categorical + 4 behavioral + risk score + decisions | — | same artifact | Yes |
| Mock "better" governance candidate PR-AUC | 0.8700 (0.8700143560447657) | VALIDATION, SYNTHETIC scores | `reports/model_governance/champion_vs_mock_better_candidate_report.json` | MUST be labeled synthetic/mock every single time it appears — never presented as real model performance |
| Mock "worse" governance candidate PR-AUC | 0.5617 (0.5616811530670596) | VALIDATION, SYNTHETIC scores | `reports/model_governance/champion_vs_mock_worse_candidate_report.json` | Same caveat — synthetic, demonstration-only |
| Governance recommendations (real runs) | worse -> REJECT, better -> PROMOTE | — | same two files | Yes, as demonstrations of the GATE MECHANISM, not as model comparisons |
| Numerical-stability fix (Phase 13) | Old formula gave negative variance / NaN std on an adversarial case; Welford's gives error ~4.9e-12 vs. numpy ground truth | Synthetic adversarial test, real code | `reports/phase13_engineering_audit_summary.md`, `tests/test_phase6_state.py` | Yes — concrete, quantified, genuinely interesting for an MLOps-reviewer audience |
| Dataset scale | 590,540 rows, 3.499% fraud rate | Raw dataset | `reports/phase0_eda_summary.md` | Yes |
| Chronological split | 70% / 15% / 15% (train/val/test), by `TransactionDT` | — | `data/interim/phase1_split_metadata.json` | Yes |

**Caution flag on the parity number**: the 2,997/3,000 (99.9%) offline/
online parity figure in `data/interim/phase6_metrics.json` was recorded
BEFORE Phase 13's Welford's-algorithm fix to `src/engine/state.py`. The
artifact was not regenerated after that fix (regenerating it requires a
real-data simulation run, out of scope for this documentation-only
phase). The 99.9% figure and its "one outlier explained by catastrophic
cancellation" story remain accurate as a historical record of what Phase
6 found, and the README should say exactly that — it should NOT claim the
parity is now "100%" or "even better" without a fresh run to back that
up, even though that is very likely true given the fix. This is a real
gap Phase 14's README should flag honestly (e.g. "as measured in Phase 6,
before a later numerical-stability fix — not re-measured since") rather
than silently update the number or silently leave the impression that
the artifact is current.

## 3. Architecture Inventory (as actually found in the repository)

```
Data:      IEEE-CIS Fraud Detection (Kaggle), 590,540 rows, 394 columns, 3.499% fraud
Split:     Chronological 70/15/15 (train/val/test) by TransactionDT — never shuffled
Baselines: Logistic Regression (Phase 2A) — established as a weak baseline
           Isolation Forest (Phase 3) — evaluated as a complementary anomaly
             signal, found NOT to add value, explicitly NOT used in the champion
Champion:  LightGBM, transaction-level features + card1-based behavioral
           features (Phase 4) — fraud-risk-lightgbm-v1
Policy:    Frozen 3-way APPROVE/REVIEW/BLOCK decision policy (Phase 5),
           thresholds selected on VALIDATION only
Engine:    Stateful RiskDecisionEngine (Phase 6) — behavioral state updated
           AFTER prediction/decision, never before; validated offline/online parity
API:       FastAPI service (Phase 7) — 9 endpoints, thin wrapper around the engine
Container: Docker (Phase 8) — local containerized deployment, not cloud/production
Observability: Prometheus-style /metrics + /monitoring/summary (Phase 9), passive
Drift:     PSI + KS batch drift detection vs. a VALIDATION-partition reference
           (Phase 10), passive, investigation-only
Governance: Local JSON model registry, explicit promotion gates, human-approved
           promotion/rollback only (Phase 11)
Dashboard: Streamlit, 7 pages, presentation layer only, reuses the real API/engine
           (Phase 12)
Audit:     Engineering/reproducibility/security polish pass (Phase 13) — fixed a
           real numerical-stability bug, added .gitignore + CI, no ML changes
```

This matches what's on disk; nothing here is aspirational.

## 4. Recommended README Structure

1. Title + one-paragraph pitch (what it does, why the problem is real,
   2-3 headline real numbers inline)
2. Table of contents
3. Architecture diagram (single, complete, ASCII — consistent with the
   project's existing style; no new diagramming tool needed)
4. Key results (a single table: the real champion's test-set numbers,
   sourced exactly as in Section 2 above)
5. What this project demonstrates (short bullets: leakage-safety
   discipline, chronological evaluation, behavioral features, governed
   promotion, drift monitoring, full test coverage — framed as
   engineering competencies, not just "we did stuff")
6. Repository layout (keep, lightly updated)
7. How to run it locally (consolidated: setup -> tests -> API ->
   dashboard -> Docker, one flow, cross-referencing the existing detailed
   sections rather than duplicating their command blocks)
8. ML approach (new — baseline -> Isolation Forest rejection -> LightGBM
   -> behavioral features, with the real numbers, and a link to each
   phase report)
9. Decision policy (condensed version of existing content)
10. Real-time serving (condensed version of existing Docker/engine content)
11. Monitoring & Observability (existing section, lightly trimmed)
12. Drift Detection (existing section, lightly trimmed)
13. Model Governance (existing section, lightly trimmed, extra-careful
    with the mock-candidate framing)
14. Dashboard (existing section, lightly trimmed)
15. Testing & CI (new — 316 tests, what's covered, the new CI workflow)
16. Engineering audit (new — one paragraph pointing to Phase 13, since
    it's currently invisible)
17. Principles this project follows (keep verbatim)
18. Limitations (new, consolidated — currently limitations are scattered
    per-section; a reviewer benefits from seeing them gathered: 58% of
    fraud still missed in APPROVE, no ground-truth-label-based accuracy
    monitoring, no authentication, in-memory-only state, etc.)
19. Dataset / setup (existing content, condensed)
20. Full phase report index (a table: phase -> one-line summary -> report link)

## 5. Missing Portfolio Assets

- Architecture diagram: not literally missing (fragments exist per
  section) but no single end-to-end one exists — should be created as an
  ASCII diagram (consistent with the rest of the project; no new tooling
  dependency).
- Dashboard screenshots: genuinely missing, and Phase 12's own report
  already documents why (no browser automation available in this
  environment) — should NOT be fabricated. Recommend either (a) leaving a
  clearly-labeled placeholder noting screenshots require a local run, or
  (b) omitting screenshots entirely and relying on the real API JSON
  examples instead, which ARE genuine.
- API examples: NOT missing — real, working `curl` examples already exist
  in the current README and in Phase 7's report; should be reused, not
  invented.
- Sample prediction: NOT missing — a real, actually-executed example
  response exists in `reports/phase7_api_service_summary.md` and was
  re-confirmed live in Phase 13's Docker validation
  (`risk_score: 0.05003272790754602, decision: "APPROVE"`); safe to quote
  directly.
- Performance tables: missing from the top-level README (exists only
  buried in `reports/phase4_...md`) — should be added using Section 2's
  verified numbers.
- Model comparison table: partially missing — the Phase 4 A/B comparison
  (transaction-only vs. +behavioral) exists in the report but not the
  README; worth a small table.
- Drift example: NOT missing — real scenario results exist in
  `reports/phase10_drift_detection_summary.md` (Scenario A/B/C/D); a
  short excerpt is a good, honest addition, but must preserve the
  "controlled/synthetic scenario" labeling exactly.
- Governance example: NOT missing, exists in
  `reports/model_governance/*.json` and Phase 11's report — must be
  presented with the mock-candidate caveat every time.
- Existing figures: 25 real PNGs already exist under `reports/figures/`
  (PR/ROC curves, feature importance, decision distributions,
  drift/parity plots) from actual runs — these are a significant,
  currently-underused asset; several are strong README candidates (e.g.
  `phase4_pr_curve_test.png`, `phase2b_feature_importance.png`,
  `phase6_offline_online_score_parity.png`).
- LICENSE file: does not exist. Common expectation for a public GitHub
  portfolio repo; flagged here as a gap, not fixed in this plan-only phase.

## 6. Claims That Need Caution

- "Production" language: every existing section already correctly says
  "local"/"not production-deployed" — this discipline must be preserved
  everywhere in the rewritten README, including any new headline/pitch
  language. No new "production-ready" or "production system" framing
  should be introduced even informally.
- `card1` pseudo-entity: every phase report is careful to say this is NOT
  a verified customer ID (Vesta's anonymized card identifier). The README
  currently doesn't over-claim here, but a rewritten "behavioral
  features" section must repeat this caveat, not drop it for brevity.
- Historical dataset vs. live traffic: the entire system was validated
  against a historical, static, already-labeled dataset via chronological
  backtesting — never real, live production transaction traffic. This
  needs to stay explicit in any "real-time" framing (the system processes
  transactions one-at-a-time and updates state correctly — genuinely
  real-time-CAPABLE — but has never seen live traffic).
- "Natural" behavioral drift: Phase 10's own finding (cumulative
  behavioral features structurally drift across chronological periods
  even with "stable" underlying behavior) must be preserved with its full
  nuance — a shortened README version must not accidentally imply all
  behavioral PSI movement is either "fine" or "a problem"; the original
  conditional framing needs to survive condensing.
- Mock governance challenger: as flagged repeatedly above — the 0.8700
  PR-AUC number is a synthetic, label-informed blend, constructed ONLY to
  exercise the promotion-gate code path, and MUST carry a
  "synthetic/mock — not a real model" label at every single appearance,
  including in any summary table, not just in prose.
- Monitoring architecture vs. real production monitoring volume: the
  Prometheus-style `/metrics` endpoint and Phase 9's `MetricsRegistry`
  are real, working code — but all "requests observed" numbers anywhere
  in this project come from small demonstration/test runs (tens of
  requests), never real production-scale traffic. Any README framing of
  the monitoring system should describe its DESIGN and DEMONSTRATED
  correctness, not imply it has handled production volume.
- Offline/online parity "99.9%": as detailed in Section 2's caution flag —
  this number predates Phase 13's numerical-stability fix and should be
  presented as a historical Phase 6 finding, not silently updated or
  implied to be freshly re-measured.
- The "+0.0055" behavioral-feature improvement: real and directionally
  positive, but genuinely small — Phase 4's own report calls it "a small
  but real, validation-confirmed improvement" and explicitly warns
  against overselling it. The README rewrite must not inflate this into
  stronger language than the source report itself uses.
- Simulation-subset metrics vs. full-test-set metrics: the Phase 6
  simulation's PR-AUC (0.5057, on a 3,000-row subset) must never be
  quoted alongside or in place of the full-test-set PR-AUC (0.5483)
  without explicitly noting they're different, differently-sized
  populations — Phase 6's own report is careful about this distinction
  and the README rewrite must preserve it.

---

STOPPING HERE per instructions. No README changes, no source code
changes, no dashboard changes, no test changes, no Docker changes were
made in this phase. Awaiting review before proceeding to any actual
rewrite.
