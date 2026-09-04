# Phase 12 Summary — Fraud Risk Intelligence Dashboard

This report documents a Streamlit dashboard presenting the existing
Phases 0-11 system end-to-end. All numbers and validation results below
are from real, independently-executed runs in this session — a fresh
full test suite run, a real live FastAPI service exercised through the
dashboard's actual (unmocked) API client, and a real `streamlit run`
process smoke-tested against every page.

**The dashboard is a presentation layer only.** No inference, feature
generation, drift detection, or governance logic was implemented inside
it — every number and decision it shows comes from calling the real
FastAPI service or reading real, already-computed project artifacts.

## Starting point: existing work found on inspection

Per the brief's instruction to inspect the repository before modifying
anything, the repository was found to already contain a complete
dashboard implementation (`dashboard/app.py`, 6 pages, 5 utility modules,
a test suite, and a refresh script) — apparently from earlier work on this
exact phase. Rather than rebuild it, this session's work was inspection,
independent validation, and verification that nothing was fabricated:
every page's source was read in full, the real test suite was re-run from
a clean state, real API integrations were exercised against a live
service (not mocks), and a real Streamlit process was launched and
smoke-tested. One piece of genuinely missing work was found and completed
in this session: the README's Dashboard section did not exist yet and has
been added.

## Files added/changed (this session)

- Modified: `README.md` — added the `## Dashboard` section (purpose,
  launch instructions, integration summary, offline-fallback behavior),
  updated the `## Status` line and `## Repository layout` diagram to
  include `dashboard/`, `artifacts/`, and the Phase 6+ `src/` subpackages.
- Regenerated: `artifacts/dashboard_test_summary.json` — refreshed via
  `python scripts/refresh_dashboard_test_summary.py` to reflect this
  session's real test run.
- Created: this report.

Everything else under `dashboard/`, `tests/test_phase12_dashboard.py`, and
`scripts/refresh_dashboard_test_summary.py` was found already complete and
correct on inspection (detailed below) and required no changes.

## Dashboard architecture

```
dashboard/
    app.py                    # Executive Overview (entry point)
    pages/
        1_Live_Scoring.py       # interactive transaction scoring + decision signals
        2_Analytics.py            # risk/decision analytics from real evaluation artifacts
        3_Monitoring.py             # live Phase 9 observability
        4_Drift.py                    # live + offline Phase 10 drift monitoring
        5_Governance.py                 # real champion + labeled mock governance demo
        6_Architecture.py                 # static system-flow diagram
    utils/
        api_client.py            # thin HTTP client for the real FastAPI service
        engine_fallback.py         # local RiskDecisionEngine fallback (same engine, same method)
        artifacts.py                 # loaders for real, already-computed JSON/YAML artifacts
        session.py                     # shared sidebar state, cached local-engine resource
        ui.py                            # presentation-only badges/formatting helpers
```

Standard Streamlit multipage-app convention (`pages/` directory,
number-prefixed filenames) — chosen because it's the natural fit for a
project already organized in numbered phases, and requires no additional
framework beyond what `streamlit` itself provides.

## Dashboard features actually implemented

**A. Executive Overview** — champion model/version/status (from the real
Phase 11 registry), frozen decision policy configuration (from
`config/decision_policy.yaml`), real Phase 4 test-set PR-AUC/ROC-AUC/
Recall@1/2/5%, real Phase 5 test-set decision distribution, live API/
drift/governance status when reachable, and a cached (not live) automated
test count with its recorded timestamp.

**B. Live Transaction Risk Scoring** — a form (with two example presets)
that submits a transaction to the real `POST /predict` endpoint when the
API is reachable, or the real `RiskDecisionEngine.process_transaction()`
directly otherwise, and displays risk score, decision, policy thresholds,
and (when using the local engine, which returns them) the raw behavioral
signals that fed into the decision — explicitly labeled "Decision
Signals," not "explanation," since this project has not implemented SHAP
or any causal attribution method.

**C. Risk & Decision Analytics** — APPROVE/REVIEW/BLOCK distribution,
fraud-vs-legitimate counts per bucket, Recall@K/Precision@K bar charts,
and operational workload/friction metrics — all read directly from the
real, saved `data/interim/phase4_metrics.json` / `phase5_metrics.json`
final test-set evaluation, never recomputed.

**D. Monitoring Dashboard** — request/prediction/decision/risk-score/
error metrics from the real, live `GET /monitoring/summary` and raw
`GET /metrics` text — requires the live API (documented honestly: there
is no meaningful offline substitute for in-memory, per-process counters).

**E. Drift Monitoring** — live `GET /drift/summary` status, the real
Phase 10 reference-profile artifact (monitored numeric/categorical/
behavioral signals and their reference distributions), and an interactive
"run a demo batch" button against the real `POST /drift/analyze`.
Explicitly preserves the Phase 10 finding that some behavioral-feature PSI
movement reflects the features' cumulative, time-accumulating design
rather than a data-quality problem — the exact caveat from
`reports/phase10_drift_detection_summary.md`, not diluted.

**F. Model Governance** — the real current champion (live API or the real
`artifacts/models/registry.json`), including its real SHA-256 artifact
hash, clearly separated by a prominent warning banner from Phase 11's
mock-candidate governance demonstration reports — the mock "better"
candidate's PR-AUC (0.8700) is labeled "synthetic" everywhere it appears
and is never presented as a real model result. Also documents, in plain
text, that promotion/rollback are human-invoked-only and that a dedicated
test confirms `src/drift/` cannot import `src/governance/`.

**G. Architecture / System Flow** — a static, text-diagram page showing
the real `RiskDecisionEngine.process_transaction()` step sequence
(explicitly highlighting that state updates happen LAST), how
Observability/Drift/Governance sit outside the inference path as
read-only observers, and the full phase map.

## Integration — exactly which existing modules/APIs are reused

| Dashboard component | Reuses |
|---|---|
| Live Scoring (API path) | `POST /predict` (Phase 7, unchanged) |
| Live Scoring (fallback path) | `src.api.dependencies.build_engine()` + `RiskDecisionEngine.process_transaction()` (Phases 6/7, unchanged) |
| Executive Overview / Analytics | `data/interim/phase4_metrics.json`, `phase5_metrics.json`, `config/decision_policy.yaml` (Phases 4/5, unchanged) |
| Monitoring | `GET /monitoring/summary`, `GET /metrics` (Phase 9, unchanged) |
| Drift | `GET /drift/summary`, `POST /drift/analyze`, `artifacts/drift/reference_profile.json` (Phase 10, unchanged) |
| Governance | `GET /model-governance/summary`, `artifacts/models/registry.json`, `reports/model_governance/*.json` (Phase 11, unchanged) |

No file under `dashboard/` computes a risk score, generates a behavioral
feature, runs a PSI/KS test, or evaluates a promotion gate — every one of
those operations happens exactly once, in its already-approved module,
and the dashboard only ever reads the result.

## Validation — all real, executed this session

### 1. Full test suite

```
$ pytest -q
315 passed, 58 warnings in 17.08s
```

Re-run independently from a clean `__pycache__`/`.pytest_cache` state (not
just trusting the pre-existing cached summary). 25 of these are
`tests/test_phase12_dashboard.py`: dashboard imports, page-file syntax
validity, 7 API-client tests against a mocked HTTP layer (`responses`
library — no live server needed for these), 8 real artifact-loader tests
against the actual repository files, 4 UI-helper tests, and 3 tests
against the REAL local `RiskDecisionEngine` (valid transaction, invalid/
negative-amount rejection, `isFraud`-present rejection) — no mocking for
those three, a real engine is built and scores a real transaction.

### 2. Live FastAPI + real (unmocked) dashboard API-client integration

Started a real `uvicorn src.api.main:app` process, then called the
dashboard's actual `dashboard.utils.api_client` functions (the same code
the pages import) against it — not the `responses`-mocked test versions:

```
health: True {'status': 'ok', 'model_loaded': True, 'policy_loaded': True}
metadata: True {'model_version': 'phase4_lightgbm_transaction_plus_behavioral_v1', 'policy_name': 'balanced', ...}
predict: True {'transaction_id': 9500001, 'risk_score': 0.05574999751084692, 'decision': 'APPROVE', ...}
monitoring available: True total requests: 5
drift available: True {'available': True, 'batches_analyzed': 0, ...}
governance available: True {'available': True, 'champion_model_id': 'fraud-risk-lightgbm-v1', ...}
```

### 3. Real Streamlit launch + smoke test

```
$ streamlit run dashboard/app.py --server.headless true --server.port 8501 &
$ curl http://127.0.0.1:8501/                      -> 200
$ curl http://127.0.0.1:8501/_stcore/health          -> ok
$ curl http://127.0.0.1:8501/Live_Scoring            -> 200
$ curl http://127.0.0.1:8501/Analytics                -> 200
$ curl http://127.0.0.1:8501/Monitoring                 -> 200
$ curl http://127.0.0.1:8501/Drift                       -> 200
$ curl http://127.0.0.1:8501/Governance                    -> 200
$ curl http://127.0.0.1:8501/Architecture                    -> 200
```

Run TWICE (two separate processes, two separate ports) with a full log
inspection after the second run — grepped for "traceback"/"exception"/
"error" (case-insensitive) across the complete server log: none found.
This ran with the real FastAPI service simultaneously live, so every page
had genuine access to real API data during the smoke test, not just the
offline-fallback path.

Browser automation was not available in this environment, so visual
rendering/interactivity (form submission, chart display) was not verified
by a real browser session — see Limitations.

## Limitations

- No browser automation available — page HTTP-200 responses and clean
  server logs confirm the Streamlit app starts, routes correctly, and
  renders each page's Python code without raising an exception, but do
  not confirm pixel-level visual correctness or interactive browser-side
  behavior (e.g. actually clicking the "Score Transaction" button in a
  real browser). The underlying logic those interactions call
  (`api_client.predict`, `score_transaction_locally`) is independently
  tested and was independently exercised live in validation step 2 above.
- Monitoring page has no offline mode — by design (in-memory, per-process
  counters have no meaningful offline substitute), not an oversight;
  documented plainly on the page itself.
- Cached, not live, test count on Executive Overview — refreshed
  explicitly via `scripts/refresh_dashboard_test_summary.py`, not
  recomputed on every dashboard page load (would be far too slow for an
  interactive UI); the page shows the exact recorded timestamp so this is
  never presented as a live number.
- No Docker packaging for the dashboard itself — Phase 8's container
  packages the FastAPI service only; running the dashboard still requires
  a local Python environment with `streamlit` installed. Not requested by
  this phase's brief.

## What was intentionally NOT implemented

Consistent with the brief's engineering constraints: the trained model
was not modified or retrained, no threshold or decision policy was
changed, no feature definition was changed, chronological evaluation was
not altered, no data leakage was introduced, LightGBM was not replaced,
no performance metric was invented, no production traffic was fabricated,
and the mock Phase 11 governance candidate's synthetic PR-AUC is never
presented as a real result anywhere on the Governance page.
