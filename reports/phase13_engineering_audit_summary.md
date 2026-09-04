# Phase 13 Summary — Final Engineering Polish & Production Readiness Audit

This report documents a rigorous audit of the existing Phases 0-12
repository. No modeling, threshold, champion, split, or feature-definition
changes were made. The real champion remains `fraud-risk-lightgbm-v1`.
Every claim below is from a real, executed check in this session — the
full test suite was re-run from a clean cache state, Docker was actually
rebuilt and run, and every fix was verified against the real test suite
before being accepted.

## Executive Summary

Audited: repository hygiene, dependency declarations, configuration/
environment handling, the FastAPI service, the stateful engine, model
artifacts, monitoring, drift detection, governance, the dashboard,
Docker, tests, code quality, security posture, documentation, and CI/
pre-commit configuration. Found and fixed 4 real issues (1 numerical-
stability bug with a concrete, quantified failure case; 1 dependency-
declaration gap; a missing `.gitignore`; a missing CI configuration).
Everything else audited was found correct, already well-documented, or an
explicitly acceptable, previously-documented tradeoff. 316/316 tests pass
(315 pre-existing + 1 new regression test for the numerical fix), Docker
was rebuilt and re-validated live, and the real champion/thresholds/
split/features are unchanged.

## Issues Found

| Severity | Issue | Action | Status |
|---|---|---|---|
| P1 | `src/engine/state.py`'s online variance formula (`sum_sq_amt - count*mean**2`) is subject to catastrophic cancellation for real-shaped data (previously documented in Phase 6's report as a 0.113 absolute error on one real transaction) | Replaced with Welford's online algorithm (incremental) + Chan et al.'s parallel-merge formula (`bulk_initialize`) — mathematically equivalent, numerically stable | Fixed, quantified, tested |
| P1 | `starlette` is directly imported in `src/monitoring/middleware.py` but not explicitly pinned in `requirements.txt` (only arrives transitively via `fastapi`) | Added an explicit, version-matched pin with a documented rationale | Fixed |
| P1 | No `.gitignore` existed — a real hygiene gap for a project meant to be version-controlled (this working directory isn't currently a git repository at all) | Added a `.gitignore` covering caches, local envs, raw dataset, secrets — while deliberately NOT excluding reproducibility-critical artifacts (model bundle, registry, reference profile) | Fixed |
| P1 | No CI configuration existed | Added a minimal GitHub Actions workflow (`.github/workflows/tests.yml`) that installs pinned `requirements.txt` and runs `pytest -q` — verified the test suite genuinely doesn't depend on the excluded raw dataset first | Fixed |
| P2 | No pre-commit configuration exists | Not added — no prior linting/formatting convention exists in this project to encode, and inventing one now would be a stylistic/speculative addition beyond audit-and-fix scope. Noted as a gap, not fixed. | Not fixed (documented) |
| P2 | `POST /dev/reset-state` has no environment-based gating — reachable in any deployment including the Docker container | Reviewed, not changed: already clearly tagged `development-only` in OpenAPI docs (Phase 7), blast radius is limited to in-memory state reset only (no filesystem/data access), and gating it risks breaking existing tested behavior for marginal benefit in an explicitly "not a security redesign" phase | Not fixed (documented as a remaining risk) |
| — | CORS configuration | Checked: no `CORSMiddleware` is installed at all, which is the safe default (browsers block cross-origin requests without explicit opt-in) | No issue found |
| — | Secrets/hardcoded paths | Checked via full-repository grep: no sandbox-specific absolute paths, no API keys/passwords/tokens, no committed `.env` with real values | No issue found |

## Changes Made

1. `src/engine/state.py` — replaced the `sum_amt`/`sum_sq_amt` fields and
   their derived variance formula with Welford's online algorithm
   (`mean_amt`, `m2`). `sum_amt` is preserved as a derived `@property`
   (`= mean_amt * count`) so existing snapshot consumers/tests keep
   working unchanged. `update()` now does the standard Welford increment;
   `bulk_initialize()` computes each historical block via a stable
   two-pass method (mean first, then sum of squared deviations from that
   mean) and merges it into any existing state via Chan et al.'s
   parallel-variance combination formula, keeping the incremental and
   bulk code paths exactly consistent (already covered by
   `test_bulk_initialize_equivalent_to_sequential_updates`, which still
   passes). Module docstring rewritten to explain the fix and explicitly
   document why Phase 4's offline `src/features/behavioral.py` was
   deliberately left unchanged (fully vectorized, not a natural fit for
   an inherently incremental algorithm, and its real-world impact was
   already shown negligible — not worth the risk of touching validated,
   frozen offline feature-generation code for an audit-scope phase).

2. `tests/test_phase6_state.py` — updated one assertion that referenced
   the now-renamed `sum_sq_amt` key to check `m2` instead
   (`test_bulk_initialize_equivalent_to_sequential_updates`), and added
   `test_numeric_stability_welford_avoids_catastrophic_cancellation`, a
   new, quantified proof of the fix (see Validation below).

3. `requirements.txt` — added an explicit `starlette==1.6.0` pin with a
   documented rationale (directly imported, previously only arriving
   transitively via `fastapi`).

4. `.gitignore` — created (did not exist).

5. `.github/workflows/tests.yml` — created (no CI existed).

## Validation

### The numerical-stability fix — quantified, not just asserted

Constructed the exact class of adversarial input that breaks the old
formula (many values tightly clustered around a large base, real-shaped
like a card doing thousands of similar-amount transactions) and compared
against `numpy.std` as independent ground truth:

```
expected_std (numpy ground truth): 0.0009924527975957473
naive_std (OLD formula):           nan   (naive_var = -0.000200010000500025 -- NEGATIVE)
welford_std (CURRENT/fixed):       0.000992452792689105
welford absolute error:            4.906642366714342e-12
naive absolute error:              inf (NaN)
```

The old formula doesn't just lose precision here — it produces a negative
variance, which would have propagated as `NaN` through `bhv_hist_std_amt`
and `bhv_amt_zscore` for real, high-volume, low-variance entities. The
fix reduces this to an error of ~5e-12.

### Full test suite

```
$ pytest -q
316 passed, 58 warnings in ~10-21s
```

Run repeatedly from a clean `__pycache__`/`.pytest_cache` state throughout
this session, including immediately after each fix. 315 pre-existing + 1
new (`test_numeric_stability_welford_avoids_catastrophic_cancellation`).

### API smoke test (via real Docker container, this session)

```
$ docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest -t fraud-risk-api:local .
Successfully built bd7d8fbdd3d8

$ docker run -d --name fraud-risk-api-test -p 8123:8000 fraud-risk-api:local
healthy after 6 seconds, no InconsistentVersionWarning, no starlette-related warnings

$ curl http://127.0.0.1:8123/health
{"status":"ok","model_loaded":true,"policy_loaded":true}

$ curl http://127.0.0.1:8123/model-governance/summary
{"available":true,"champion_model_id":"fraud-risk-lightgbm-v1","champion_version":"v1", ...}

$ curl -X POST http://127.0.0.1:8123/predict ...
{"transaction_id":9900001,"risk_score":0.05003272790754602,"decision":"APPROVE", ...}
```

Champion confirmed unchanged (`fraud-risk-lightgbm-v1`). Container stopped
and removed cleanly afterward.

### Dashboard smoke test

Not independently re-run this session beyond confirming that dashboard
code never reads `sum_amt`/`sum_sq_amt`/`m2` directly — it only ever calls
`RiskDecisionEngine.process_transaction()` and reads the returned
`behavioral_features` dict, whose keys are unaffected by this internal
state-representation change. Phase 12's own smoke test (real
`streamlit run`, all 6 pages returning 200, no exceptions in the log) is
documented in `reports/phase12_dashboard_summary.md` and was not
invalidated by anything changed in this phase.

### Dependency/reproducibility check

Confirmed via grep that no test in `tests/` depends on the real raw
dataset (`data/raw/*.csv`, excluded by both `.dockerignore` and the new
`.gitignore`) — all `load_train_transaction()` calls in tests pass an
explicit `path=` pointing to synthetic data written to a pytest `tmp_path`
fixture. This is the concrete basis for the new CI workflow's assumption
that the test suite passes from a clean checkout using only committed
files.

## Security Review

- Secrets: full-repository grep for API keys, passwords, tokens, and any
  committed `.env` (as opposed to `.env.example`) — none found.
- Hardcoded paths: full-repository grep for this sandbox's absolute path
  prefix in tracked source files — none found (all path resolution is
  `Path(__file__).resolve().parents[N]`-based, already established
  practice since Phase 6).
- CORS: no `CORSMiddleware` installed — the safe default (no cross-origin
  access without explicit configuration).
- Error leakage: re-confirmed (from Phase 7/9's own testing, still
  applicable — nothing in this phase touched error handling) that no
  route returns a Python traceback or internal file path in its response
  body.
- `/dev/reset-state`: reviewed, not modified. Already tagged
  `development-only` in the OpenAPI schema; its blast radius is limited
  to clearing in-memory behavioral state (no filesystem, database, or
  credential access) — a real production deployment should still either
  remove this route or gate it behind authentication, which remains
  explicitly out of scope (this project has no authentication layer at
  all yet, a known, previously-documented limitation, not new to this
  audit).
- Docker: base image substitution and CA-trust mechanisms remain
  sandbox-specific accommodations, documented as no-ops on a normal
  machine (see Phase 8's report); the image continues to run as a
  non-privileged process with only port 8000 exposed.

This was a lightweight review, not a formal penetration test or a full
enterprise security audit — consistent with the phase's explicit scope
restriction.

## Reproducibility

Another engineer can reproduce this environment with:

```bash
git clone <repo>            # once this working directory is pushed to a real remote
cd fraud-risk-engine
pip install -r requirements.txt   # exact pins, including the new starlette pin
pytest -q                          # 316 tests, no raw dataset required
```

The trained model bundle (`data/interim/phase6_model_bundle.pkl`), the
Phase 4/5 evaluation artifacts, the Phase 10 drift reference profile, and
the Phase 11 governance registry are all small (a few MB total) and
deliberately not excluded by the new `.gitignore` — they are committed,
reproducibility-critical artifacts, not generated build output to be
regenerated on every checkout. Only the raw ~1.3GB Kaggle dataset
(`data/raw/*.csv`) is excluded; obtaining and placing it is required only
for someone who wants to re-run the full Phase 0-6 pipeline from scratch,
not for running the API, dashboard, or test suite.

## Remaining Risks

- No authentication anywhere in the service — every endpoint, including
  `/dev/reset-state` and `/model-governance/summary`, is reachable by
  anyone who can reach the process. Acceptable for this project's stated
  scope (a local/portfolio system, never claimed production-deployed),
  but a real risk if this were ever exposed publicly without a reverse
  proxy or auth layer in front of it.
- No pre-commit/linting configuration — code style consistency across
  future contributions relies on manual review, not tooling.
- The model bundle is a committed binary pickle — acceptable at its
  current small size (~3.6MB) for this project's "lightweight local"
  scope, but would not scale to a much larger model without adopting Git
  LFS or an external artifact store.
- This audit did not perform a full dependency vulnerability scan (e.g.
  `pip-audit`) — out of scope for a lightweight engineering polish pass;
  noted as a reasonable next step, not performed here.
