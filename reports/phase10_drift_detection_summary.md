# Phase 10 Summary — ML Drift Detection & Data Monitoring

This report documents a passive batch drift-monitoring layer that compares
incoming data and model-output distributions against an approved
historical reference. All numbers are from real runs — a fast synthetic
test suite, a real reference profile built from actual validation data,
a real demonstration with both real and controlled-synthetic scenarios,
and a real Docker build/run with the new endpoints validated live.

## 1. Phase objective

Answer "has the statistical behavior of incoming transactions changed
enough that the model or decision system may require investigation?" —
a detection and reporting capability, never automatic retraining,
threshold changes, or decision changes.

## 2. Observability vs. drift detection — the difference

Phase 9 (observability) answers "what is the service doing right now"
(request counts, latency, error rates). Phase 10 (drift detection)
answers a different question: "does what the service is *seeing and
producing* still look like what it was *approved and calibrated on*?"
Phase 9 has no concept of a reference distribution or a comparison; Phase
10 exists specifically to make that comparison, on a batch, against a
frozen historical baseline. They integrate (`/drift/summary` follows the
same lightweight-summary pattern as `/monitoring/summary`) but are
conceptually and architecturally distinct — `src/drift/` never imports
`src/monitoring/` or vice versa.

## 3. Drift-monitoring architecture

```
src/drift/
    __init__.py
    reference.py     # reference profile construction, save/load
    detectors.py       # PSI, KS, severity thresholds
    monitor.py           # DriftMonitor.analyze_batch() - the one public entry point
    report.py              # severity aggregation + explanation building
```

`DriftMonitor.analyze_batch()` is deliberately a **pure, passive analysis
over data the caller already has** — it does not call the model, the
feature pipeline, or the behavioral-state engine. The caller supplies a
batch that already contains the monitored raw columns, the
already-computed Phase 4 `bhv_*` columns, and the already-computed risk
scores/decisions. This is the concrete mechanism behind "do not create a
second inconsistent behavioral-feature implementation": `src/drift/` has
no feature-generation code of its own to diverge from the approved
engine.

## 4. Reference-data strategy

**VALIDATION**, not TRAIN, not TEST — built once via
`python -m src.run_phase10_build_reference`, saved to
`artifacts/drift/reference_profile.json` (real run: 88,581 transactions,
~950KB JSON).

- TEST is never used (explicitly forbidden, and would break this
  project's long-standing "test set is evaluation-only" rule).
- Raw TRAIN is not used either: it spans a much longer, less
  deployment-representative period, and was also used to fit
  preprocessing objects — using it as reference would conflate "what the
  model was fit on" with "what recent, representative traffic looks
  like."
- VALIDATION is the partition Phase 5 already treated as the best
  available proxy for approved, near-deployment data (policy thresholds
  were calibrated on it), making it the most defensible reference without
  new data collection.

The saved artifact contains quantile bin edges/counts and a bounded
5,000-value random sample per numeric feature (for PSI and KS
respectively), plus category-frequency tables — not the raw dataset, so
drift analysis never needs the ~683MB CSV at runtime.

## 5. Features selected for monitoring

Documented, non-exhaustive, from actual repository feature names:

| Type | Features | Why |
|---|---|---|
| Numeric | `TransactionAmt`, `C1`, `C13`, `C14`, `D2`, `V258` | Business-meaningful amount + Phase 2B's own top-10 LightGBM features by gain |
| Categorical | `ProductCD`, `R_emaildomain` | Low-cardinality business signal + another Phase 2B top-10 feature |
| Behavioral (Phase 4) | `bhv_prev_txn_count`, `bhv_hist_mean_amt`, `bhv_time_since_prev_txn`, `bhv_prior_count_24h` | One representative feature per category (count/amount/recency/velocity) |
| Outputs | risk score, decision distribution | The model's own outputs - most directly answer "did downstream behavior change" |

## 6. Numerical drift methodology

**PSI** (primary) on reference-quantile-based bins (10 bins, standard
practice: each reference bin starts with ~equal count), plus a
**complementary KS test** (`scipy.stats.ks_2samp`) against the reference's
bounded sample — KS answers a genuinely different question than PSI
(sensitive to shape differences a coarse 10-bin histogram might miss).
Zero-frequency bins are handled by clipping percentages to a small floor
(`1e-4`) before the PSI log-ratio, never dropping bins or renormalizing
away real structure. Severity thresholds (`PSI < 0.10` no significant
drift, `< 0.20` low, `< 0.30` moderate, `>= 0.30` high) are explicitly
documented as project-configurable conventions, not universal truth.

## 7. Categorical drift methodology (including a real fix)

Same PSI mechanics, extended for two categorical-specific concerns:

- **New/unseen categories are surfaced explicitly** (`new_categories` in
  every result), never silently dropped — tracked against the reference's
  full original category set, independent of any grouping below.
- **Rare-category grouping (`__OTHER__` bucket)**: categories below 1%
  reference frequency are grouped into a single bucket before computing
  PSI. This was not a hypothetical concern — it was found and fixed
  during real validation (see §17): `R_emaildomain` has 55 reference
  categories with a long tail of rare ones, and a modest current batch
  will show 0 occurrences for many of them purely from sampling noise,
  inflating naive PSI to 0.78 without reflecting any real distributional
  change. Grouping brought it down to a more representative 0.19-0.31
  depending on the batch, while still surfacing genuinely new categories
  by name, unaffected by the grouping.

## 8. Behavioral-feature drift approach — and a genuine methodological finding

Behavioral features are monitored with the exact same numeric PSI/KS
machinery, using the exact existing Phase 4 names (§5) — no reimplemented
feature logic. Real validation surfaced an important, honestly-reported
finding: **Phase 4's behavioral features are cumulative counters that grow
over the dataset's timeline by construction** (`bhv_prev_txn_count`,
`bhv_hist_mean_amt`, etc. accumulate more history the further into the
timeline a transaction sits). Comparing the VALIDATION reference against a
sample from the TEST partition (a genuinely later chronological period)
showed real, measurable "drift" in these features (`bhv_hist_mean_amt`
PSI=0.37, HIGH) — not because anything anomalous happened, but because
test-partition entities have structurally accumulated more prior history
than validation-partition entities had at their point in time. This is a
property of the feature design on a chronologically-split dataset, not a
data-quality problem — reported here explicitly rather than treated as
either a bug to silently fix or a real alert to raise.

## 9. Risk-score drift — results

Scenario D (controlled): reference mean 0.0293 vs. a synthetic
`uniform(0.7, 1.0)` batch mean 0.8592 -> PSI=8.28, `HIGH_DRIFT`. The
properly-constructed stable scenario (§11) showed no risk-score drift
(PSI=0.04, `NO_SIGNIFICANT_DRIFT`).

## 10. Decision-distribution drift — results

Reference (validation, real): `{APPROVE: 97.9996%, REVIEW: 0.6999%, BLOCK: 1.3005%}`
— matches Phase 5's saved validation numbers exactly, confirming
reproducibility. Scenario D's shifted-score batch produced decisions
dominated by BLOCK/REVIEW -> PSI=13.32, `HIGH_DRIFT`. The stable scenario
showed PSI=0.0018, `NO_SIGNIFICANT_DRIFT`.

## 11. Stable-batch demonstration result (the corrected Scenario A)

500 transactions randomly sampled from the reference partition itself
(validation) — per the brief's own definition of Scenario A ("sampled
similarly to reference"), not from a different chronological period.
Real result: `LOW_DRIFT` overall, driven only by `R_emaildomain`
(PSI=0.19, a residual long-tail sampling effect even after `__OTHER__`
grouping); every other monitored signal — all 6 numeric, `ProductCD`, all
4 behavioral features, risk score, and decision distribution — showed
`NO_SIGNIFICANT_DRIFT`. This is the expected, correct outcome for a
genuinely stable batch.

## 12. Synthetic drift scenario results — all four, real runs

| Scenario | Manipulation | Overall Result | Primary Driver(s) |
|---|---|---|---|
| A — Stable | None (random sample from reference partition) | `LOW_DRIFT` | `R_emaildomain` only |
| B — Amount shift | `TransactionAmt * 15 + 500` | `HIGH_DRIFT` | `TransactionAmt` (PSI=8.29) |
| C — Categorical shift | `ProductCD` forced to `'C'` for all rows | `HIGH_DRIFT` | `ProductCD` (PSI=9.84) |
| D — Risk-score shift | Scores replaced with `uniform(0.7, 1.0)` | `HIGH_DRIFT` | `risk_score` (PSI=8.28), `decision_distribution` (PSI=13.32) |

Each scenario is explicitly labeled CONTROLLED/SYNTHETIC in the
demonstration's own output — never presented as naturally observed
project data. Scenarios B and C each isolate cleanly to their intended
feature (no other feature crosses into MODERATE/HIGH_DRIFT), and Scenario
D isolates to the two output signals — exactly the behavior a
correctly-working detector should show.

## 13. Small-batch behavior

`MIN_BATCH_SIZE = 30` (documented project convention, not a rigorously
derived value). A real 2-record batch sent to `POST /drift/analyze`
(inside the live Docker container) produced a full, real result — PSI and
severity were still computed, never skipped — but with
`"low_confidence": true` and an explicit warning: "Batch size (2) is
below the minimum recommended size (30); drift results below have limited
statistical confidence and should not be treated as a reliable signal on
their own." The system never pretends more certainty than a 2-observation
comparison can support.

## 14. Overall drift severity logic

**Worst-feature-wins**: the overall batch severity is the maximum severity
across every monitored feature and output, full stop — not a weighted
score, not a black box. `overall_explanation` names exactly which
features/outputs are at that maximum level (`contributing_signals`),
so a person reading the report immediately sees what's driving it (e.g.
Scenario B: "High Drift driven primarily by: TransactionAmt").

## 15. Observability integration

`GET /drift/summary` — read-only, cheap, reports the latest known
state (batches analyzed, severity histogram, latest status/explanation);
verified it does NOT itself trigger a new analysis
(`test_drift_summary_does_not_trigger_new_analysis`). `POST /drift/analyze`
is the only way a new batch gets analyzed — a deliberately separate,
explicit, batch-level operation, never triggered automatically by
`/predict`. No new metrics counters were added to Phase 9's
`MetricsRegistry` (drift has its own lightweight in-`DriftMonitor`
tracking instead, matching Phase 9's own in-memory-only, no-database
design).

## 16. Non-interference validation — the critical result

Two dedicated tests directly address the brief's core requirement:

- `test_drift_monitoring_does_not_change_predict_behavior`: the same
  transaction, scored once through an API with no drift monitor loaded at
  all, and once through an API with a drift monitor loaded AND actively
  exercised (a real `POST /drift/analyze` call in between) — risk scores
  match to `<1e-9`, decisions identical.
- `test_drift_analyze_does_not_touch_behavioral_state`: confirms
  `POST /drift/analyze` never calls the engine and never mutates
  behavioral state for any entity, even when a batch happens to reference
  a real `card1` value.

Both passed. Drift monitoring genuinely does not touch the inference
path.

## 17. Docker compatibility — REAL, EXECUTED

Rebuilt the Phase 8/9 image with Phase 10's new files (`src/drift/`, the
reference-profile artifact) and validated live:

```
$ docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest -t fraud-risk-api:local .
Successfully built 3f8d8f86209b

$ docker run -d --name fraud-risk-api-test -p 8123:8000 fraud-risk-api:local
$ docker inspect --format='{{.State.Health.Status}}' fraud-risk-api-test
healthy   # after 5 seconds - unchanged from Phase 8/9

$ curl http://127.0.0.1:8123/health
{"status":"ok","model_loaded":true,"policy_loaded":true}

$ curl -X POST http://127.0.0.1:8123/predict ...
{"transaction_id":7000001,"risk_score":0.0616832113578294,"decision":"APPROVE", ...}

$ curl http://127.0.0.1:8123/drift/summary
{"available":true,"batches_analyzed":0,"severity_counts":{},"latest_status":null, ...}

$ curl -X POST http://127.0.0.1:8123/drift/analyze -d '{"records": [...2 extreme records...]}'
{"batch_size":2,"low_confidence":true,"warnings":["Batch size (2) is below..."], ...
 "overall_severity":"HIGH_DRIFT", ...}

$ curl http://127.0.0.1:8123/drift/summary
{"available":true,"batches_analyzed":1,"severity_counts":{"HIGH_DRIFT":1}, ...}
```

`/health` and `/predict` unaffected; both new drift endpoints confirmed
working with accurate, real-time-updated state inside the container.
`requirements.txt` did not change for this phase (only `numpy`/`pandas`/
`scipy`, already present, are used).

## 18. Demonstration results

`python scripts/demo_drift_monitoring.py` — full real run: loads the real
reference profile, the real model bundle, real validation and test data;
Scenario A (§11) and all four scenarios (§12) executed and confirmed via
explicit assertions in the script (all passed, exit code 0). The
cross-partition (validation-vs-test) informational comparison from §8 is
also printed, clearly labeled as informational context, not one of the
four required scenarios.

## Test suite results

**248/248 Python tests passed** (206 from Phases 0-9 + 42 new: 32 unit
tests for reference-profile construction, detectors, aggregation, and
`DriftMonitor` in isolation using synthetic data; 10 FastAPI integration
tests including the two non-interference tests in §16). Phase 0 and
Phase 1 scripts re-verified working on real data.

## Errors encountered and fixes

1. **`NameError: OTHER_BUCKET_LABEL` not imported.** Trivial — a missing
   import in `src/drift/detectors.py` after adding the rare-category
   grouping (§7). Caught immediately by the test suite (16 test failures)
   before any real-data use.
2. **Categorical PSI inflated by long-tail rare categories (real finding,
   real fix).** The first real run of the demonstration script showed
   `R_emaildomain` at PSI=0.78 (`HIGH_DRIFT`) even for data drawn from the
   reference partition's own neighborhood — investigated rather than
   accepted, traced to `R_emaildomain`'s 55 reference categories with many
   below 1% frequency, each contributing sampling-noise "surprise" to a
   modest current batch. Fixed with the standard `__OTHER__`-bucket
   grouping technique (§7), which reduced the same measurement to a more
   representative 0.19-0.31 while still surfacing genuinely new categories
   by name.
3. **Scenario A methodological flaw (not a code bug, a demo-design
   error).** The first version of the demonstration built "Scenario A:
   stable" from a real TEST-partition sample — which, per §8's finding,
   is NOT a fair stable-data test for behavioral features, since test and
   validation sit at different points in the dataset's timeline and
   cumulative behavioral counters differ structurally between them. The
   script's own assertion ("Scenario A should be stable") caught this
   immediately (`HIGH_DRIFT` instead of the expected low/none). Fixed by
   sampling Scenario A from the reference (validation) partition itself,
   per the brief's own definition, and keeping the original
   validation-vs-test comparison as clearly-labeled informational context
   instead of presenting it as the stable-data scenario.

## Known limitations

- **Batch-only, by design** — no single-transaction drift verdict exists
  or is meaningful; `MIN_BATCH_SIZE=30` is a documented convention, not a
  rigorously derived statistical minimum for this specific application.
- **In-memory `DriftMonitor` history** — like Phase 9's metrics, batch
  counts and severity history reset on process restart; no persistence.
- **PSI/KS assume the monitored feature's type doesn't change** — a
  column that silently switches from numeric to string upstream, for
  example, isn't specifically detected as a schema-drift event (only
  categorical vs. numeric are handled as configured, not auto-detected).
- **The `__OTHER__` grouping threshold (1%) is a fixed, project-chosen
  constant**, not tuned per feature — a feature with a very different
  natural cardinality/skew profile might warrant a different threshold.
- **Behavioral-feature "drift" between different time periods is
  partially structural** (§8) — any future comparison against a batch
  from meaningfully later in the timeline should expect some baseline
  behavioral movement that isn't necessarily an operational problem, and
  this system does not currently separate that structural component from
  a genuine behavioral change.

## What drift detection does NOT prove

**Drift does not automatically mean model performance degradation.** A
high PSI on `TransactionAmt` means the current batch's amounts look
different from the reference period's — it does not, by itself, mean
fraud is being missed, the model is miscalibrated, or the policy
thresholds are wrong. Drift is an investigation signal, not a conclusion.

**Without delayed ground-truth fraud labels, this system cannot directly
measure online model accuracy.** Every risk score and decision monitored
here is the model's own output — never compared against a true `isFraud`
outcome, because that outcome isn't known at prediction time and often
isn't known for a considerable delay in real fraud systems (chargebacks,
disputes, etc. arrive later). This phase therefore cannot and does not
claim to know whether the model is "still accurate" — only whether the
data it's seeing and the decisions it's making look statistically
consistent with the approved reference period.

## Future path toward model lifecycle management

A genuine model lifecycle system would need, at minimum: delayed-label
ingestion (once true fraud outcomes become available) to measure actual
online precision/recall against the model's historical validation
performance, automated alerting integrated with a real monitoring
platform (not just a JSON endpoint), a defined retraining trigger policy
(explicitly NOT implemented here — Phase 10 detects, it does not decide
to retrain), a model registry to track which model/policy version served
which traffic, and a documented human-in-the-loop investigation workflow
for what happens after a HIGH_DRIFT report. None of this exists yet — all
explicitly out of scope for Phase 10, per the brief.

## What was intentionally NOT implemented

Automatic retraining, automatic model replacement, automatic threshold
changes, automatic decision changes, feature-engineering redesign, new
fraud models, a model registry, a cloud monitoring platform, database
infrastructure, Kafka, Kubernetes, Grafana deployment, full CI/CD redesign
— all reserved for later phases, per the Phase 10 scope restrictions.
