# Phase 15 — Interview Preparation & Project Storytelling

Personal interview-preparation material for the **Real-Time Risk Decision
& Fraud Intelligence Engine** project. Every number here was re-verified
against the actual repository artifacts in this session
(`data/interim/phase4_metrics.json`, `phase5_metrics.json`,
`phase6_metrics.json`, `artifacts/models/registry.json`, source files).
Where the repository does not establish something, this report says so
explicitly rather than guessing.

---

# PART 1 — PROJECT STORY

## A. 30-Second Version

> "I built an end-to-end fraud risk platform on the IEEE-CIS fraud
> dataset — about 590,000 transactions with severe class imbalance,
> around 3.5% fraud. I trained a LightGBM model with leakage-safe
> chronological evaluation, added historical behavioral features per
> card, and got 0.5483 test PR-AUC. But the interesting part is
> everything around the model: a stateful real-time engine that scores a
> transaction *before* updating that entity's history, wrapped in a
> FastAPI service, with monitoring, drift detection, and a model
> governance layer that gates any future model promotion behind explicit
> checks and human approval." *(≈95 words)*

## B. 2-Minute Version

**Problem.** Fraud detection is a severe class-imbalance problem — only
3.499% of transactions in this dataset are fraud — so accuracy is
meaningless and even ROC-AUC can be misleading; I optimized for PR-AUC
and recall at realistic review budgets instead.

**Dataset.** IEEE-CIS Fraud Detection, 590,540 transactions, 394 columns,
anonymized transaction and identity data with a relative timestamp
(`TransactionDT`), not a random ID.

**Leakage-safe evaluation.** I split strictly chronologically — 70%
train, 15% validation, 15% test, by time, never shuffled — because random
splitting would let the model implicitly see the future.

**Model.** I baselined with Logistic Regression, then trained LightGBM as
the champion, and separately evaluated Isolation Forest as a
complementary anomaly signal — it didn't help (PR-AUC 0.0449, barely
above the 0.0348 no-skill baseline), so I explicitly excluded it rather
than force an ensemble.

**Behavioral features.** I added 12 features aggregated per card —
transaction count, historical mean/std amount, recency, velocity —
computed strictly from *prior* transactions only. That gave a modest but
real +0.0055 validation PR-AUC improvement.

**Decision policy.** Rather than a binary call, I built a three-way
APPROVE/REVIEW/BLOCK policy, thresholds selected only on validation data,
achieving 84.85% precision on blocked transactions with only 0.20% of
legitimate customers affected.

**Real-time engine.** I built a stateful engine that reads history,
scores, decides, and *then* updates state — never the other way — and
validated 100% offline/online decision parity on a 3,000-transaction
simulation.

**API/Docker.** Wrapped in a FastAPI service, containerized in Docker
with the raw dataset excluded from the image.

**Monitoring/Drift/Governance.** I added Prometheus-style monitoring,
PSI/KS drift detection against a validation-period reference, and a
local model governance registry with six explainable promotion gates —
so a new model can never silently replace the current one.

## C. 10-Minute Deep Dive

### 1. Problem
**What** — predict which transactions are fraudulent. **Why** — fraud
causes direct financial loss and erodes customer trust; false blocks
cause customer friction and lost legitimate revenue, so this is a
precision/recall balancing problem, not a pure accuracy problem. **How**
— framed as binary classification with a downstream three-way decision
layer. **Result** — 3.499% fraud prevalence established class imbalance
as the central technical constraint from the start.

### 2. Dataset
**What** — IEEE-CIS Fraud Detection (Kaggle/Vesta), 590,540 rows, 394
columns. **Why this dataset** — real, large-scale, genuinely anonymized
transaction data with the exact properties (imbalance, anonymized
categorical/numeric features, a relative timestamp) that make fraud
modeling hard in practice. **How** — loaded via a config-driven loader
(`src/data/load.py`), inspected for structure/temporal/missingness
patterns before any modeling (`reports/phase0_eda_summary.md`).
**Result** — established `TransactionDT` (not a calendar date, a relative
offset) as the ordering key for everything downstream.

### 3. Data split
**What** — 70/15/15 train/validation/test by `TransactionDT`. **Why** —
chronological, not random, to mirror deployment (predicting the future
from the past, never the reverse). **How** — a single ordered cut, no
shuffling, no k-fold (`src/data/split.py`, `data/interim/phase1_split_metadata.json`).
**Result** — 413,378 / 88,581 / 88,581 rows; test touched exactly once,
for final reporting only.

### 4. Leakage prevention
**What** — a set of structural rules, not just good intentions: no
preprocessing object fit outside TRAIN, no feature computed from
future-relative-to-itself data, no threshold selected on TEST. **Why** —
leakage silently inflates offline metrics and produces a model that fails
in deployment. **How** — enforced in code (e.g. behavioral features only
ever look backward; `FeaturePipeline` is fit once, on TRAIN, and reused).
**Result** — every phase report documents this discipline explicitly; it
is the foundation the rest of the project's credibility rests on.

### 5. Baseline
**What** — Logistic Regression. **Why** — establish a simple,
interpretable floor before investing in a more complex model. **How** —
standard preprocessing (scaling/encoding) plus L2-regularized logistic
regression. **Result** — Test PR-AUC 0.1604, Test ROC-AUC 0.7940
(`reports/phase2a_logistic_regression_summary.md`) — a real, weak
baseline that makes LightGBM's improvement meaningful by comparison.

### 6. LightGBM
**What** — gradient-boosted decision trees, the champion model. **Why**
— strong, well-established performance on structured/tabular data with
mixed numeric/categorical features and imbalance, faster to iterate than
deep learning at this scale. **How** — trained on transaction-level
features first, then combined with behavioral features (Phase 4); native
categorical handling, no manual one-hot for high-cardinality columns.
**Result** — Test PR-AUC 0.5483, Test ROC-AUC 0.9058, the project's
headline number.

### 7. Isolation Forest experiment
**What** — an unsupervised anomaly detector, evaluated as a potential
complementary signal to LightGBM. **Why** — the hypothesis that an
anomaly detector might catch fraud patterns a supervised model misses.
**How** — trained independently, compared its flagged transactions and
PR-AUC against LightGBM's on the same test set
(`reports/phase3_anomaly_detection_summary.md`). **Result** — Test PR-AUC
0.0449, barely above the 0.0348 no-skill baseline for this fraud rate,
with its own flagged transactions overwhelmingly already caught by
LightGBM — **explicitly rejected** and not part of the shipped system.

### 8. Behavioral features
**What** — 12 `bhv_*` features per `card1` (transaction count, historical
mean/std/min/max amount, recency, velocity, z-score). **Why** — a single
transaction is less informative than "how does this compare to this
card's own recent pattern." **How** — computed strictly from transactions
before the current one (`src/features/behavioral.py` offline,
`src/engine/state.py` online). **Result** — validation PR-AUC improved
+0.0055 (0.6092 → 0.6147) — real, modest, not oversold.

### 9. Decision policy
**What** — a frozen three-way APPROVE/REVIEW/BLOCK policy on top of the
risk score. **Why** — a probability alone isn't an operational decision;
REVIEW routes ambiguous cases to a human rather than forcing a binary
automated call. **How** — thresholds selected on VALIDATION only, then
evaluated once on TEST (`src/decision/`). **Result** — 84.85% precision
among BLOCKed transactions, 0.20% of legitimate customers blocked, 41.9%
of all fraud captured by REVIEW+BLOCK combined.

### 10. Stateful inference
**What** — a `RiskDecisionEngine` maintaining per-entity behavioral state
across calls. **Why** — behavioral features need history; recomputing
from scratch on every request would require full transaction history
lookup on every call. **How** — read state → generate features → predict
→ decide → *then* update state (`src/engine/risk_engine.py`,
`src/engine/state.py`). **Result** — 100% offline/online decision parity,
0.999999996 score correlation, across a 3,000-transaction simulation.

### 11. FastAPI
**What** — a 9-endpoint REST service wrapping the engine. **Why** — a
clean, documented, typed interface between the trained system and any
consumer (dashboard, tests, a future real caller). **How** — Pydantic
request/response models, the model/engine loaded once at startup, not
per-request (`src/api/main.py`). **Result** — real, tested example: a
live request returned `risk_score: 0.05003272790754602, decision:
"APPROVE"` in ~155ms (one measured call, not a load-tested figure).

### 12. Docker
**What** — a container image for the API. **Why** — reproducible,
portable local deployment; not a cloud deployment claim. **How** — copies
only `src/`, `config/`, `scripts/`, the trained model bundle, and small
drift/governance artifacts; the raw ~1.3GB dataset is excluded via
`.dockerignore`. **Result** — ~4.29MB build context (measured Phase 8).

### 13. Monitoring
**What** — Prometheus-style `/metrics` + JSON `/monitoring/summary`. **Why**
— you can't operate what you can't observe: request/prediction counts,
latency, decision distribution, error rates. **How** — a passive
middleware + registry, verified via dedicated non-interference tests to
never alter a prediction. **Result** — real, working code, exercised at
demo/test scale (tens of requests) — not production-volume-tested.

### 14. Drift detection
**What** — PSI + KS comparison of incoming batches against a
VALIDATION-partition reference. **Why** — a model's inputs can shift over
time even if the model itself doesn't change. **How** — reference-
quantile binning for numeric features, rare-category grouping for
categoricals to avoid long-tail sampling noise (`src/drift/`). **Result**
— real methodology bug found and fixed during development (see Part 10).

### 15. Governance
**What** — a local JSON model registry with six explainable promotion
gates. **Why** — no model should silently replace the current one — every
promotion needs an auditable reason and an explicit human approval.
**How** — `src/governance/gates.py`; `promote()`/`rollback()` require
`approved_by`/`reason` arguments. **Result** — real champion registered
with a real SHA-256 artifact hash
(`bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342`).

### 16. Dashboard
**What** — a 7-page Streamlit app presenting the whole system. **Why** —
a single place to see model results, live scoring, monitoring, drift, and
governance without hitting raw JSON endpoints. **How** — every page calls
the real API or the real engine directly; zero duplicated inference/
feature/drift/governance logic (`dashboard/`). **Result** — real Docker +
Streamlit smoke test: all pages return HTTP 200, zero exceptions logged.

### 17. Engineering lessons
The two strongest lessons are the categorical-PSI long-tail bug and the
Welford's-algorithm numerical-stability fix — both real, both found
during this project's own validation, both detailed as STAR stories in
Parts 10-11 below. The broader lesson: **rigorous testing at every phase
surfaced real bugs a purely "does it run" check would have missed.**

---

# PART 2 — ARCHITECTURE EXPLANATION

```text
Transaction
     │
     ▼
Validation                        ← Pydantic schema check; rejects malformed/missing fields (422)
     │
     ▼
Historical State Lookup            ← READ-ONLY lookup of this entity's existing behavioral state
     │
     ▼
Behavioral Features                 ← computed from that state, NOT from the current transaction
     │
     ▼
Preprocessing                        ← the frozen Phase 1 pipeline + Phase 4 LightGBM preprocessor
     │
     ▼
LightGBM                              ← the trained champion model
     │
     ▼
Risk Score
     │
     ▼
Decision Policy                        ← frozen Phase 5 thresholds
     │
     ▼
APPROVE / REVIEW / BLOCK
     │
     ▼
State Update                            ← ONLY NOW does this transaction get added to history
     │
     ▼
Monitoring / Drift / Governance          ← passive observers, outside the inference path
```

**Every arrow, explained:**
- Transaction → Validation: a malformed request (missing required field,
  wrong type, or an `isFraud` field present) is rejected with a 422
  before it ever reaches the engine.
- Validation → State Lookup: the engine looks up (never mutates) the
  requesting entity's (`card1`) existing summary statistics.
- State Lookup → Behavioral Features: the 12 `bhv_*` features are derived
  purely from that looked-up state — the CURRENT transaction's own amount/
  time have not yet been incorporated into anything.
- Behavioral Features → Preprocessing → LightGBM → Risk Score: the frozen,
  already-validated inference path — unchanged since Phase 4.
- Risk Score → Decision Policy → APPROVE/REVIEW/BLOCK: the frozen Phase 5
  thresholds map the continuous score to one of three actions.
- Decision → State Update: **only now** is the current transaction's
  amount/timestamp folded into that entity's running statistics.
- State Update → Monitoring/Drift/Governance: these observe the
  already-finalized result — they cannot change it (verified by dedicated
  non-interference tests for each subsystem).

**Why state update happens AFTER prediction, specifically**: if the
current transaction's own amount were included in "this entity's
historical average" before scoring, the transaction would partially be
scored against itself — a subtle, easy-to-miss form of leakage. Ordering
the code so state mutation is structurally the *last* step (not just
documented as a rule) makes this leakage impossible by construction, not
just by convention.

**Conceptual example** (illustrative only — not real transaction data):

```text
Transaction A (card1 = X, amount = 50)
   │
   ▼ score using card X's history BEFORE A (empty, first transaction)
   ▼ decision: APPROVE (no history to be suspicious about)
   ▼ NOW add A (amount=50, time=t_A) to card X's history

Transaction B (card1 = X, amount = 500, shortly after A)
   │
   ▼ score using card X's history INCLUDING A (count=1, mean=50, ...)
   ▼ decision: likely REVIEW/BLOCK — 500 is 10x this card's only prior amount
   ▼ NOW add B to card X's history
```

This illustrates exactly why the ordering matters: B's risk score is
informed by A (a real prior transaction), but A's own risk score was
never informed by B (which hadn't happened yet) or by itself.

---

# PART 3 — ML INTERVIEW QUESTIONS

### Q1. Why fraud detection instead of a generic classification example?
**Short answer**: it's a genuinely hard, realistic applied ML problem —
severe imbalance, temporal structure, and a real operational decision
layer, not just a clean toy dataset.
**Deep answer**: most "generic classification" tutorials use balanced or
mildly imbalanced data with i.i.d. splits; fraud detection forces you to
confront class imbalance (PR-AUC over accuracy), temporal leakage
(chronological splits), and the fact that a probability isn't a decision.
**Project-specific**: this project's entire structure — the metric
choice, the split strategy, the three-way policy — exists *because* of
these fraud-specific constraints, not despite them.
**Follow-up**: "Would this approach generalize to another imbalanced,
temporal problem, like churn prediction?" **Follow-up answer**: yes — the
chronological-split, PR-AUC-first, tiered-decision approach generalizes
directly to any problem with rare positive events and a meaningful time
axis (churn, intrusion detection, credit default).

### Q2. Why chronological split?
**Short answer**: to mirror how the model will actually be used —
predicting the future from the past, never the reverse.
**Deep answer**: with i.i.d. random splitting, information from
transactions temporally after a given test point can leak into training
(e.g. via aggregate features, or simply because patterns evolve and a
random split blends periods together), producing offline metrics that
don't reflect real deployment performance.
**Project-specific**: split by `TransactionDT` into TRAIN (70%,
0-413,377), VALIDATION (15%, 413,378-501,958), TEST (15%,
501,959-590,539) — verified boundaries, never shuffled.
**Follow-up**: "What if fraud patterns are highly seasonal and your test
window happens to be atypical?" **Follow-up answer**: that's a real risk
of a single chronological split — a production system would monitor for
exactly this via the drift-detection layer, and a single historical
backtest is not a substitute for monitoring live performance over time.

### Q3. Why not random train/test split?
**Short answer**: random splitting would let the model implicitly learn
from the future to predict the past, inflating offline metrics
unrealistically.
**Deep answer**: this is a specific instance of a well-known pitfall —
in any dataset with temporal structure, i.i.d. cross-validation optimistically
biases performance estimates because held-out points can be
temporally *interleaved* with training points rather than strictly
following them.
**Project-specific**: never used anywhere in this project — confirmed by
`src/data/split.py`'s single ordered cut and repeated leakage-guard tests.
**Follow-up**: "Did you measure how much worse a random split would have
looked?" **Follow-up answer**: **Not established by the project
evidence** — no random-split comparison was run; the decision was made a
priori as a leakage-prevention discipline, not validated empirically
against the alternative.

### Q4. What is temporal leakage?
**Short answer**: information from the future being available (directly
or indirectly) when making a prediction about the past, causing offline
performance to look better than real deployment performance would.
**Deep answer**: it can be direct (a feature literally derived from
future rows) or indirect (a fitted preprocessing statistic — mean,
encoding — computed across a time-mixed dataset that includes future
information).
**Project-specific**: prevented via chronological split, TRAIN-only
preprocessing fitting, and behavioral features restricted to strictly-prior
transactions (enforced in both offline and online code paths).
**Follow-up**: "Give a concrete example of indirect leakage you avoided."
**Follow-up answer**: a naive behavioral-feature implementation might
compute "this card's average transaction amount" using its ENTIRE history
including transactions after the one being scored — this project computes
it using only transactions strictly before, verified by dedicated Phase 4
leakage tests.

### Q5. Why PR-AUC?
**Short answer**: PR-AUC focuses on performance on the positive
(minority) class, which is what actually matters under severe imbalance;
ROC-AUC can look good even when precision on fraud is poor.
**Deep answer**: PR-AUC is the area under the precision-recall curve —
precision = TP/(TP+FP), recall = TP/(TP+FN) — and is sensitive to the
positive-class base rate in a way that makes it a harsher, more
informative metric when positives are rare, unlike ROC-AUC's false
positive rate term (FP/(FP+TN)), which is diluted by the huge negative
class.
**Project-specific**: used as the primary metric throughout — model
selection, behavioral-feature validation, and the Key Results table all
lead with PR-AUC (0.5483), not ROC-AUC.
**Follow-up**: "If PR-AUC is 0.5483, is 0.9058 ROC-AUC misleading?"
**Follow-up answer**: not misleading exactly, but incomplete on its own —
it shows the model separates classes well overall, but PR-AUC is the
number that tells you how that separation holds up specifically on the
rare positive class, which is the one that matters operationally.

### Q6. Why not accuracy?
**Short answer**: with 3.499% fraud, a model predicting "not fraud" for
every transaction would be 96.5% accurate while catching zero fraud.
**Deep answer**: accuracy = (TP+TN)/total is dominated by the majority
class under imbalance; it doesn't distinguish a genuinely good model from
a trivial majority-class predictor.
**Project-specific**: accuracy is not reported as a primary metric
anywhere in this project's evaluation artifacts.
**Follow-up**: "So why report ROC-AUC at all if it can be misleading?"
**Follow-up answer**: it's still informative as a general ranking-quality
signal and is a common reference point reviewers expect to see — reported
alongside PR-AUC, never in place of it.

### Q7. Why ROC-AUC?
**Short answer**: it measures the model's ability to rank a random
positive above a random negative, independent of any specific threshold
— a useful general discrimination measure to report alongside PR-AUC.
**Deep answer**: ROC-AUC = P(score(positive) > score(negative)) for a
randomly drawn pair; it's threshold-independent and less sensitive to
class-imbalance ratio changes than accuracy, but more optimistic than
PR-AUC when positives are rare because it's diluted by the huge
true-negative count.
**Project-specific**: reported as a secondary metric (0.9058) alongside
PR-AUC everywhere in this project — never used alone to make a modeling
decision.
**Follow-up**: "Could a model have high ROC-AUC and low PR-AUC?" **Follow-up
answer**: yes, and that's exactly the pattern under severe imbalance —
this project's own numbers (0.9058 ROC-AUC vs. 0.5483 PR-AUC) illustrate
it directly.

### Q8. What does PR-AUC = 0.5483 actually mean?
**Short answer**: averaged across all possible score thresholds, the
model's precision-recall tradeoff traces a curve whose area is 0.5483 —
meaningfully better than a random/no-skill classifier (which would score
≈0.035, the fraud base rate) but far from perfect (1.0).
**Deep answer**: PR-AUC is computed as the area under the precision(recall)
curve, roughly a weighted average of precision across all recall levels;
because a no-skill classifier's PR curve is a flat line at the positive
base rate (0.0348 in this project's case — confirmed empirically in Phase
3), 0.5483 represents roughly a 15-16x improvement over random ranking.
**Project-specific**: verified directly in `data/interim/phase4_metrics.json`;
contextualized against the Phase 3 no-skill baseline of 0.0348.
**Follow-up**: "Is 0.5483 'good' in absolute terms?" **Follow-up
answer**: see the Adversarial section (Part 14) for the full honest
answer — it's a real, meaningful lift over baseline, not a
state-of-the-art claim, and this project doesn't compare it against
external published benchmarks on a different train/test protocol.

### Q9. Why LightGBM?
**Short answer**: strong, well-established performance on structured/
tabular data with mixed feature types and class imbalance, fast to train
and iterate.
**Deep answer**: gradient-boosted trees natively handle mixed numeric/
categorical features, are robust to feature scale, and (via
`scale_pos_weight` or similar) handle imbalance more gracefully than many
alternatives, without requiring the large data volumes or extensive
tuning deep learning typically needs for tabular data.
**Project-specific**: trained with fixed hyperparameters
(`src/models/lightgbm_model.py`'s `DEFAULT_PARAMS`, `random_state=42`),
selected over Logistic Regression based on a direct PR-AUC comparison.
**Follow-up**: "Why not XGBoost or CatBoost?" — see Part 14 (Adversarial).

### Q10. Why Logistic Regression as a baseline?
**Short answer**: to establish a simple, fast, interpretable floor before
justifying the added complexity of LightGBM.
**Deep answer**: Logistic Regression is a linear model with a known,
well-understood failure mode (can't capture non-linear feature
interactions) — a useful reference point precisely because its
limitations are well understood.
**Project-specific**: Test PR-AUC 0.1604 vs. LightGBM's 0.5483 — a large,
meaningful gap that justifies the more complex model.
**Follow-up**: "Would a simpler LightGBM (fewer trees) have been almost
as good?" **Follow-up answer**: **Not established by the project
evidence** — no hyperparameter sweep/ablation over tree count vs.
performance was run; `DEFAULT_PARAMS` were fixed and used directly.

### Q11. Why did Isolation Forest fail to add value?
**Short answer**: its Test PR-AUC (0.0449) was barely above the no-skill
baseline (0.0348) for this fraud rate, and its flagged transactions were
overwhelmingly already caught by LightGBM alone.
**Deep answer**: Isolation Forest detects generic outliers, not
specifically fraud — an unusual-but-legitimate transaction and an unusual
fraudulent one look similar to a purely unsupervised anomaly score, so
without label information it struggles to specifically target the rare
class of interest.
**Project-specific**: measured directly via a complementarity analysis —
decision parity showed its flagged transactions added ~0 unique fraud
catches beyond LightGBM (`reports/phase3_anomaly_detection_summary.md`).
**Follow-up**: "Would a supervised anomaly-adjacent method (e.g. one-class
SVM with label-informed threshold tuning) have done better?" **Follow-up
answer**: **Not established by the project evidence** — only Isolation
Forest was evaluated as the anomaly-detection candidate.

### Q12. What is class imbalance?
**Short answer**: when one class (here, fraud) is much rarer than the
other, so naive optimization (e.g. for accuracy) is biased toward
ignoring the rare class.
**Deep answer**: with a 3.499% positive rate, a classifier can achieve
96.5% accuracy trivially; imbalance also affects which metrics are
informative (PR-AUC over ROC-AUC/accuracy) and how a decision threshold
should be chosen (not the default 0.5).
**Project-specific**: the central constraint shaping metric choice
(PR-AUC), threshold selection (validation-based, not default 0.5), and
the very existence of a three-way policy (a single threshold at 0.5 would
be a poor operational fit).
**Follow-up**: "Did you use any resampling (SMOTE, undersampling)?"
**Follow-up answer**: **Not established by the project evidence** — no
resampling technique was applied; imbalance was addressed via metric
choice and (implicitly) LightGBM's own handling of imbalanced leaf
statistics, not explicit resampling.

### Q13. What does Recall@1% mean?
**Short answer**: if you can only review the top 1% highest-risk
transactions (a fixed review budget), you'd catch 25.30% of all actual
fraud in the test set.
**Deep answer**: rank all transactions by risk score descending, take the
top 1% by volume, and measure what fraction of true fraud cases fall
within that top slice — directly answers "given limited review capacity,
how much fraud do we actually catch."
**Project-specific**: verified in `data/interim/phase4_metrics.json`
(`test_budget_table`, `budget: 0.01`); this framing is what connects a
ranking metric to a real staffing/capacity constraint.
**Follow-up**: "How was the 1% budget chosen?" **Follow-up answer**: it's
one of three illustrative budgets (1%/2%/5%) reported to show the
recall-vs-capacity curve shape — **not established** that 1% specifically
reflects any real analyst-team capacity; it's a standard reporting
convention, not a business-derived constraint in this project.

### Q14. What does Recall@2% mean?
**Short answer**: reviewing the top 2% highest-risk transactions catches
40.90% of all fraud.
**Deep answer**: same mechanism as Q13, at double the review budget —
recall increases with budget (25.30% → 40.90% → 60.14% as budget goes
1%→2%→5%), illustrating diminishing-but-real returns to additional review
capacity.
**Project-specific**: from the same `test_budget_table`.
**Follow-up**: "Is the recall gain from 1%→2% (a doubling of budget)
proportional?" **Follow-up answer**: no — 25.30%→40.90% is roughly a
1.6x gain for a 2x budget increase, showing diminishing marginal returns,
consistent with a well-calibrated ranking model (the highest-risk cases
are concentrated near the top).

### Q15. What does Recall@5% mean?
**Short answer**: reviewing the top 5% highest-risk transactions catches
60.14% of all fraud.
**Deep answer**: same mechanism, largest of the three reported budgets —
even at 5% review volume, nearly 40% of fraud is still missed, which is
an honest, important limitation to state plainly.
**Project-specific**: from the same `test_budget_table`; precision at
this budget drops to 41.85% (more false positives mixed in as the net
widens).
**Follow-up**: "What's the tradeoff of increasing the budget further, say
to 10%?" **Follow-up answer**: **Not established by the project
evidence** — only 1%/2%/5% were reported; the general pattern (recall up,
precision down, diminishing returns) would be expected to continue but
wasn't specifically measured at 10%.

### Q16. How did behavioral features improve the model?
**Short answer**: they gave the model historical context per entity —
transaction count, typical amount, recency, velocity — improving
validation PR-AUC by +0.0055 (0.6092 → 0.6147).
**Deep answer**: a transaction-only model has no way to know "is this
amount unusual FOR THIS CARD specifically" — behavioral features encode
exactly that relative signal, which a tree model can then split on (e.g.
`bhv_amt_zscore` being sharply high).
**Project-specific**: verified via a controlled A/B comparison, same
model architecture, transaction-only features vs. transaction+behavioral,
same evaluation protocol.
**Follow-up**: "+0.0055 is small — was it worth the added complexity?"
**Follow-up answer**: it's a real, validation-confirmed, reproducible gain
— modest, honestly described as such, not claimed as transformative; the
decision to keep it was evidence-based (Phase 4's own report explicitly
frames it this way), not a foregone conclusion.

### Q17. How was behavioral leakage prevented?
**Short answer**: features are computed strictly from transactions
before the current one, enforced in both the offline batch code and the
online stateful engine.
**Deep answer**: offline, a strict `(TransactionDT, TransactionID)`-
ordered cumulative computation excludes the current row from its own
aggregate; online, the engine reads existing state BEFORE prediction and
writes new state only AFTER the decision is finalized — structurally
impossible to leak, not just documented as a rule.
**Project-specific**: verified by dedicated Phase 4 leakage tests and the
Phase 6 predict-before-update engine design; a specific pair of genuine
bugs were found and fixed during Phase 4/6 development (documented in
those phase reports) where an early implementation accidentally included
the current row.
**Follow-up**: "How would you catch a subtle leakage bug like that in
review?" **Follow-up answer**: exactly the way this project did — a
dedicated unit test asserting a known, hand-computed small example's
exact feature values, not just "the code runs without error."

### Q18. Why use `card1`?
**Short answer**: it's the closest thing to a stable per-card identifier
available in the anonymized dataset, so it was used as the aggregation
key for behavioral features.
**Deep answer**: the dataset provides no verified customer or account ID;
`card1` is Vesta's own internal card-related identifier, chosen because it
had zero missing values and a reasonable per-entity transaction volume
(median ~4 transactions/entity, per Phase 0's EDA) — enough history to
make aggregation meaningful.
**Project-specific**: documented explicitly as a "pseudo-entity" in every
relevant phase report and feature docstring, never presented as a
verified identity.
**Follow-up**: "Could different real customers share a `card1` value, or
one customer have multiple?" **Follow-up answer**: plausibly yes on both
counts — this is a known limitation of using an anonymized, undisclosed
identifier as a proxy entity key; **not established by the project
evidence** exactly how noisy this mapping is, since ground truth isn't
available in the anonymized dataset.

### Q19. Is `card1` a customer ID?
**Short answer**: no — explicitly not. It's an anonymized, Vesta-internal
card-related identifier with no verified one-to-one mapping to a real
person or account.
**Deep answer**: this distinction matters both technically (the
"entity" behavioral features aggregate over may not be a clean concept)
and ethically/in interviews (overclaiming "customer profiling" would be
inaccurate).
**Project-specific**: every phase report and this README are careful to
use "pseudo-entity," never "customer."
**Follow-up**: "How would this change with a real customer ID?" **Follow-up
answer**: with a verified customer/account ID, behavioral aggregation
would be more semantically clean (one true entity per key), likely
reducing the noise `card1`'s ambiguity introduces — but that's a
reasonable inference, not something this project measured.

### Q20. What happens for an entity with no history?
**Short answer**: behavioral features are `NaN`/undefined (e.g.
`bhv_prev_txn_count = 0`, `bhv_hist_std_amt` undefined since you can't
compute a standard deviation from zero prior points) — a genuine
cold-start case, handled explicitly, not silently defaulted to zero.
**Deep answer**: LightGBM natively handles missing values by learning a
default split direction during training, so `NaN` behavioral features for
first-time entities are handled by the model itself, not by an ad hoc
imputation rule.
**Project-specific**: explicitly tested (`bhv_hist_std_amt` is NaN, not 0,
for a single-prior-transaction entity — verified by a dedicated Phase 4
edge-case test) and explicitly handled in Phase 8's Docker cold-start
design (`WARM_START_STATE=false` means every entity starts cold in the
container).
**Follow-up**: "Doesn't a cold-start entity get under-scrutinized, since
it has no behavioral red flags yet?" **Follow-up answer**: that's a real,
inherent limitation of behavioral features — a first-time fraudulent
transaction has no history to look anomalous against; the transaction-level
features (amount, product code, etc.) are the only signal available
for a true cold start, which is exactly why the model uses both
transaction-level AND behavioral features rather than behavioral alone.

### Q21. What is the difference between offline and online features?
**Short answer**: offline features are computed once, in a batch, over
the full historical dataset (for training/evaluation); online features
are computed incrementally, one transaction at a time, from live-updated
state.
**Deep answer**: offline computation can use fast vectorized pandas
operations across the whole dataset; online computation must maintain
sufficient statistics (count, running mean, min/max, recent timestamps)
per entity and update them incrementally after each transaction, without
ever re-scanning full history.
**Project-specific**: offline in `src/features/behavioral.py`
(vectorized cumulative operations), online in `src/engine/state.py`
(`BehavioralStateManager`, incremental updates) — verified consistent via
a dedicated offline/online parity test suite (100% decision parity,
0.999999996 score correlation on a 3,000-transaction simulation).
**Follow-up**: "Why not just recompute from full history on every
request?" — see Part 6, Q2.

### Q22. How were decision thresholds selected?
**Short answer**: on the VALIDATION partition only, comparing three
candidate policies (conservative/balanced/aggressive), then frozen and
evaluated exactly once on TEST.
**Deep answer**: threshold selection is itself a form of "training" —
tuning it against TEST results would be leakage by another name (implicit
overfitting to the held-out set through repeated evaluation).
**Project-specific**: the "balanced" policy was selected
(`approve_threshold=0.3206, block_threshold=0.6311`) based on validation
metrics balancing fraud recall, block-bucket precision, and legitimate-
customer friction (`reports/phase5_decision_policy_summary.md`).
**Follow-up**: "What if the validation-selected thresholds perform
noticeably worse on test?" **Follow-up answer**: that's an accepted risk
of the discipline — the test results (84.85% block precision, 0.20%
legitimate-blocked rate) are reported as the honest, one-time
confirmation, not adjusted after the fact even if they'd differed from
validation expectations.

### Q23. Why three decisions instead of binary classification?
**Short answer**: a binary approve/block forces every ambiguous case into
a hard automated call; REVIEW routes genuinely uncertain cases to a human
analyst instead.
**Deep answer**: the REVIEW bucket has a much higher fraud rate (46.12%)
than APPROVE (2.07%) but lower precision than BLOCK (84.85%) — exactly
the "genuinely ambiguous" middle ground a binary threshold would have to
arbitrarily assign to one side or the other.
**Project-specific**: verified test-set bucket statistics
(`data/interim/phase5_metrics.json`); only 0.82% of test traffic falls
into REVIEW, keeping analyst workload proportionate.
**Follow-up**: "Doesn't REVIEW just defer the hard problem to a human?"
**Follow-up answer**: yes, intentionally — that's the point; human review
capacity is a real, finite resource, and this project treats it as a
managed budget (this is exactly what Recall@K/review-budget framing
formalizes) rather than pretending an automated system should resolve
every ambiguous case itself.

### Q24. How would you handle changing fraud patterns?
**Short answer**: this project's answer is drift detection (Phase 10) —
passively flagging when incoming data/output distributions diverge from
the reference, as an investigation signal for a human to act on.
**Deep answer**: drift detection does NOT retrain or adapt the model
automatically; it surfaces PSI/KS-based severity signals so a human
decides whether investigation or retraining is warranted.
**Project-specific**: `GET /drift/summary` / `POST /drift/analyze`,
reference built from the VALIDATION partition, six monitored feature
categories.
**Follow-up**: "What's the actual retraining trigger/cadence?" **Follow-up
answer**: **Not established by the project evidence** — this project
explicitly does not implement automated retraining or a retraining
trigger policy (out of scope by design, per Phase 10's brief); that's
named directly as future work in Part 17.

### Q25. What would you try next to improve model performance?
**Short answer**: the honest next steps are hyperparameter tuning
(never swept in this project), broader feature engineering beyond the 12
behavioral features, and a genuine ablation of individual features'
contribution — none of which were performed here.
**Deep answer**: LightGBM was trained with fixed `DEFAULT_PARAMS`
(`random_state=42`, no grid/Bayesian search); a systematic hyperparameter
search on validation data is a natural, low-risk next step.
**Project-specific**: **not established by the project evidence** that
any tuning was attempted — this is a genuine, stated limitation, not
something to imply was already tried and didn't help.
**Follow-up**: "If you only had one more week, what's the highest-leverage
thing to try?" **Follow-up answer**: hyperparameter tuning on validation
data is the lowest-risk, likely-highest-return next step, since the
current model uses untuned defaults — genuine free performance may be
left on the table.

---

# PART 4 — MATHEMATICAL QUESTIONS

**Sigmoid** — `sigmoid(z) = 1 / (1 + e^(-z))`. Mathematical: maps any
real logit to (0,1). Computational: LightGBM's binary objective applies
this to raw leaf-sum outputs to produce a probability. Intuitive: "how
confident, as a probability, is the model that this is fraud." Relevance:
`model.predict_proba()`'s output — the `risk_score` this entire system is
built around — is exactly this sigmoid-transformed value.

**Logits** — the pre-sigmoid raw score (sum of leaf values across trees
for LightGBM). Mathematical: `z = logit(p) = ln(p/(1-p))`. Relevance:
LightGBM builds trees additively in logit space, not probability space —
useful to know when explaining why probabilities from boosted trees
aren't simply "read off" a single tree.

**Binary cross-entropy** — `L = -[y*ln(p) + (1-y)*ln(1-p)]`. Mathematical:
the negative log-likelihood of the true label under the predicted
probability. Computational: LightGBM's default `binary` objective
minimizes this (technically its gradient/Hessian) at each boosting
iteration. Intuitive: heavily penalizes confident wrong predictions.
Relevance: **not established by the project evidence** that this
project's LightGBM used anything other than the standard `binary`
objective — `DEFAULT_PARAMS` was not deeply audited for this report
beyond `random_state`.

**Class imbalance** — see Part 3, Q12.

**Precision** — `TP / (TP + FP)`. Intuitive: "of everything I flagged,
how much was actually fraud." Relevance: 84.85% precision among BLOCKed
transactions.

**Recall** — `TP / (TP + FN)`. Intuitive: "of all actual fraud, how much
did I catch." Relevance: 25.30%/40.90%/60.14% at 1%/2%/5% review budgets.

**F1** — `2 * (precision * recall) / (precision + recall)`. The harmonic
mean, balancing both. **Not a metric this project reports** — PR-AUC and
recall-at-budget were judged more directly relevant to the operational
review-capacity framing than a single-threshold F1 score.

**PR curve** — plots precision (y) vs. recall (x) as the decision
threshold sweeps from high to low. Intuitive: shows the real tradeoff
this project's decision policy navigates.

**ROC curve** — plots true positive rate (recall) vs. false positive rate
(`FP/(FP+TN)`) as threshold sweeps. Relevance: reported as a secondary
diagnostic (`reports/figures/phase2b_roc_curve_test.png`), not the
primary decision-making curve.

**AUC / PR-AUC** — see Part 3, Q5/Q8.

**Tree-based models / gradient boosting** — an ensemble of decision trees
built sequentially, each new tree fit to the negative gradient (residual
error) of the current ensemble's loss. Intuitive: each tree corrects the
previous ensemble's mistakes. Relevance: this is exactly LightGBM's
mechanism.

**LightGBM specifically** — a gradient boosting framework using
histogram-based split finding (bucketing continuous features into
discrete bins) and leaf-wise (rather than level-wise) tree growth for
speed, plus native categorical feature support without one-hot encoding.
Relevance: the categorical handling specifically mattered for this
project's mixed-type feature set (`ProductCD`, `card4`, etc.).

**Feature importance** — LightGBM reports split-count or gain-based
importance per feature. Relevance: Phase 2B's real, saved feature
importance table shows `V258`, `C1`, `C14`, `C13` as top features by gain
(`reports/phase2b_lightgbm_summary.md`, `reports/figures/phase2b_feature_importance.png`)
— this is standard built-in gain importance, **not SHAP** (SHAP was not
implemented anywhere in this project — see Part 14 for the honest answer
on why).

---

# PART 5 — BEHAVIORAL FEATURES DEEP DIVE

**`bhv_prev_txn_count`** — Definition: count of this entity's
transactions strictly before the current one. Formula: running count.
Captures: account/card maturity and activity level. Why it helps: brand-new
or rarely-used cards can carry different risk than established ones.
Online calculation: `EntityState.count`, incremented after each `update()`
call. Leakage risk: none if incremented strictly after scoring (verified).
Edge case: 0 for a true first-time entity.

**`bhv_hist_mean_amt`** — Definition: mean of this entity's prior
transaction amounts. Formula: `mean_amt` (Welford running mean, Phase 13).
Captures: this entity's "typical" spend. Why it helps: lets the model
compare the current amount to a personalized baseline instead of a global
one. Online calculation: incremental Welford update. Leakage risk: none
(current amount excluded until after scoring). Edge case: undefined
(`NaN`) with zero prior transactions.

**`bhv_hist_std_amt`** — Definition: standard deviation of prior amounts.
Formula: `sqrt(m2 / (count - 1))` for count≥2 (Welford's `m2`, Phase 13
fix — see Part 11). Captures: this entity's spend volatility. Why it
helps: a large deviation from a LOW-volatility entity's mean is more
suspicious than the same deviation for a naturally erratic spender.
Online calculation: incremental M2 update. Leakage risk: none. Edge case:
undefined with fewer than 2 prior transactions (explicitly `NaN`, not 0 —
verified by a dedicated test).

**`bhv_time_since_prev_txn`** — Definition: seconds since this entity's
last transaction. Formula: `current_time - state.last_time`. Captures:
recency/dormancy. Why it helps: a transaction immediately after a long
dormant period, or unusually rapid-fire, can both be risk signals.
Online: `EntityState.last_time`, a single scalar, overwritten each
update. Leakage risk: none. Edge case: undefined for a first transaction
(no prior time to subtract).

**`bhv_amt_to_hist_mean_ratio`** — Definition: current amount ÷ historical
mean amount. Captures: relative, scale-independent deviation. Why it
helps: a $500 transaction means something very different for a
$50-average card vs. a $5,000-average card; this feature normalizes for
that. Online: derived at scoring time from `mean_amt` and the current
transaction's own amount (the CURRENT amount is used only as an input to
this ratio calculation, never fed back into the historical statistics
until after the decision). Leakage risk: none — this is exactly the kind
of feature where the ordering discipline matters most, since it uses the
current amount directly. Edge case: undefined if historical mean is
undefined (no prior history) or zero.

**`bhv_amt_diff_from_hist_mean`** — Definition: current amount minus
historical mean (absolute, not ratio). Captures: the same relative signal
as the ratio feature, in absolute-dollar terms — useful when the model
benefits from both scale-normalized and raw-magnitude framings.

**`bhv_amt_zscore`** — Definition: `(current_amount - hist_mean) /
hist_std`. Formula: a standard z-score. Captures: how many standard
deviations from typical this transaction is, combining both the ratio and
volatility signals into one number. Why it helps: this is the single
most "supervised-anomaly-detection-like" feature in the set — a large
absolute z-score is a classic fraud tell. Edge case: undefined if
`hist_std` is 0 or undefined (fewer than 2 prior transactions, or
zero-variance history).

**`bhv_prior_count_1h`** — Definition: count of this entity's
transactions in the trailing 1 hour before the current one. Captures:
short-term velocity — a classic fraud signal (rapid-fire card testing).
Online calculation: maintained via a bounded deque of recent timestamps,
pruned to the last 24h (so 1h count is a sub-window query over that
deque), not a full history rescan. Edge case: 0 with no recent activity.

**`bhv_prior_count_24h`** — Definition: same, over a trailing 24-hour
window. Captures: medium-term velocity. Why both 1h and 24h: they capture
different fraud patterns (immediate card-testing bursts vs. sustained
unusual activity over a day).

---

# PART 6 — REAL-TIME STATEFUL INFERENCE

### 1. Why state?
Because behavioral features require history, and re-deriving that history
from a full transaction log on every request would not scale — state is
a compact, sufficient-statistics summary (count, running mean, M2,
min/max, recent timestamps), not raw transaction storage.

### 2. Why not recompute everything?
Recomputing from full history on every request means the per-request cost
grows with an entity's total lifetime transaction count; maintaining
running sufficient statistics keeps per-request cost constant regardless
of history length.

### 3. What does "stateful" mean here specifically?
The engine (`RiskDecisionEngine` + `BehavioralStateManager`) holds
in-process, per-entity data that persists ACROSS requests — unlike a pure
function that only sees what's in the current request.

### 4. How is state updated?
Incrementally, via Welford's algorithm for mean/variance and simple
scalar updates for count/min/max/last_time/recent-timestamps deque —
`BehavioralStateManager.update()`, called once per transaction, after
scoring.

### 5. Why update after prediction?
So the current transaction can never influence the very features used to
score it — see Part 2's full explanation.

### 6. How does the engine behave for first-time entities?
Behavioral features are `NaN`/zero as appropriate (count=0, std=NaN) —
handled natively by LightGBM's missing-value split logic, not by an
explicit imputation step.

### 7. How do offline and online features match?
Verified via a dedicated parity test suite: an independent offline batch
computation compared row-by-row against the online engine's output on the
same 3,000-transaction chronological slice — 2,997/3,000 exact feature
matches, 100% decision parity, 0.999999996 score correlation.

### 8. What happens if state becomes corrupted?
**Current project**: `POST /dev/reset-state` clears all in-memory state
(a blunt, whole-system reset) — there is no per-entity repair mechanism,
and state is never persisted to disk, so a process restart also clears
it. **Production-scale extension**: a real system would need
corruption detection (e.g. sanity bounds on count/variance) and possibly
per-entity state reconstruction from an authoritative transaction log —
**not implemented here**.

### 9. What happens under concurrent requests?
**Current project**: the state manager is a single in-process Python
object; FastAPI can run sync route handlers in a thread pool, so
concurrent requests for the SAME entity could race on state updates —
**this project has not implemented explicit per-entity locking**, and
concurrent-update correctness under high contention is **not established
by the project evidence** (no concurrency stress test was run).
**Production-scale extension**: a real system would need either explicit
per-entity locking, an atomic-update-capable external state store (e.g.
Redis with atomic increment operations), or a single-writer-per-entity
partitioning scheme.

### 10. How would you scale state management?
**Current project**: single-process, in-memory, non-persistent —
explicitly documented as such in every relevant phase report.
**Production-scale extension**: a distributed key-value store (Redis,
DynamoDB) keyed by entity, with atomic increment/update operations,
horizontal partitioning by entity-hash, and a durability/backup strategy
— none of this exists in the current implementation, and this report
does not claim otherwise.

---

# PART 7 — FASTAPI / DOCKER

1. **Why FastAPI?** Modern, typed, automatic request/response validation
   via Pydantic, automatic OpenAPI docs, good performance characteristics
   for a Python API layer — and it was the framework actually used
   throughout this project.
2. **Why API instead of directly calling the model?** A clean, documented
   contract decouples callers (dashboard, tests, any future client) from
   internal implementation details, and centralizes validation/error
   handling in one place.
3. **Why load the model once?** Loading a several-MB model bundle on
   every request would add unnecessary latency and I/O — it's loaded once
   at process startup (`lifespan` context) and reused for the life of the
   process.
4. **How does request validation work?** Pydantic schema models
   (`TransactionRequest`) enforce required fields, types, and value
   constraints (e.g. `TransactionAmt >= 0`) before a request ever reaches
   the engine; malformed requests get a 422 with a clear message, not a
   crash.
5. **How are errors handled?** Distinct exception handlers for
   engine-level validation errors (422), engine-not-initialized (503),
   and genuinely unexpected errors (500, generic message only — no
   traceback or internal path ever leaked to the client).
6. **Why Docker?** Reproducible builds and runs independent of the host
   machine's Python environment — a real, portable local deployment
   artifact, not a cloud deployment claim.
7. **What goes inside the Docker image?** `src/`, `config/`, `scripts/`,
   the trained model bundle, and the small drift-reference/governance-
   registry JSON artifacts.
8. **Why exclude the raw dataset?** It's ~1.3GB and not needed for
   inference — the trained model bundle already encodes what training
   needed from it; including it would bloat the image for no runtime
   benefit (confirmed: ~4.29MB build context without it).
9. **How would you deploy this to cloud?** **Current project**: not
   deployed anywhere — Docker image built and run locally only.
   **Production-scale extension**: push the image to a container
   registry (ECR/GCR/ACR), deploy behind a managed container service
   (ECS/Cloud Run/AKS) with a load balancer, secrets management, and TLS
   termination — none of this exists in the current repository.
10. **How would you scale inference horizontally?** **Current project**:
    single-process, single-container — the in-process behavioral state
    would NOT survive naively running multiple replicas (each would have
    its own independent, inconsistent state). **Production-scale
    extension**: externalize state to a shared store (see Part 6, Q10)
    before horizontal scaling would be correct, not just possible.

---

# PART 8 — MONITORING

1. **What metrics are monitored?** Request counts (total, by endpoint, by
   status class), prediction success/failure counts, per-endpoint average
   latency, decision distribution (APPROVE/REVIEW/BLOCK counts), risk-score
   statistics (count/mean/min/max/decile histogram), and error counts by
   category. *What* → observability into service health. *Why* → you
   can't operate a service you can't see into. *How* → a passive
   middleware (`MetricsMiddleware`) plus explicit recording calls in
   `/predict`. *Failure scenario*: if the metrics registry itself failed,
   `/predict` still works — verified by a dedicated test
   (`test_monitoring_absent_does_not_break_predict`) — monitoring is
   best-effort, never a hard dependency of inference.

2. **Request metrics.** *What* → count/latency/status per endpoint. *Why*
   → answers "is the service healthy, and where's it slow." *How* →
   middleware wraps every request, reading only status code and elapsed
   time. *Failure scenario*: a 5xx spike would show up in
   `by_status_class` immediately.

3. **Prediction metrics.** *What* → attempts vs. successes vs. failures,
   distinguishing schema-level rejection from engine-level rejection.
   *Why* → "is my model actually predicting, or mostly erroring." *How*
   → recorded explicitly in the `/predict` route, after the engine call
   returns. *Failure scenario*: a spike in `engine_validation_failures`
   without a spike in `request_validation_failures` would point at a data
   quality issue specifically, not a client bug.

4. **Latency.** *What* → simple running-mean per-endpoint latency. *Why*
   → basic performance visibility. *How* → measured in the middleware via
   `time.perf_counter()`. *Failure scenario*: this is explicitly NOT
   p50/p95/p99 — a mean can hide tail latency problems; **not implemented
   in this project**, documented as a known limitation.

5. **Errors.** *What* → four distinguished categories (request-validation,
   engine-validation, engine-not-initialized, internal). *Why* →
   different error types need different responses (client bug vs. data
   quality vs. service health). *How* → each has its own exception
   handler and counter. *Failure scenario*: `engine_not_initialized_errors`
   spiking would point at a startup/deployment problem, not a data issue.

6. **Decision distribution.** *What* → live APPROVE/REVIEW/BLOCK counts.
   *Why* → a sudden shift (e.g. BLOCK rate spiking) is an operational
   signal worth investigating even before drift analysis runs. *How* →
   recorded directly from the policy's own output.

7. **Risk-score distribution.** *What* → count/mean/min/max/decile
   histogram of observed scores. *Why* → a distributional summary of what
   the model is actually outputting live. *How* → recorded on every
   successful prediction.

8. **Data-quality issues.** *What* → transaction amount summary,
   validation failure counts. *Why* → early warning for malformed or
   unusual incoming data. *How* → `extract_transaction_amount()` pulls a
   single safe numeric signal per prediction — never the full payload.

9. **Prometheus-style metrics.** *What* → `GET /metrics` in Prometheus
   text exposition format. *Why* → a widely-understood, scrapeable
   format. *How* → hand-rendered with the standard library — **no actual
   Prometheus server or Grafana is deployed anywhere in this project**,
   only the compatible text format.

10. **Why monitoring matters generally.** A model that silently stops
    working, or a service that silently degrades, is worse than one that
    fails loudly — monitoring is what makes "silently" not an option.
    *Failure scenario without it*: the difference between finding out
    about a problem from a monitoring dashboard vs. from an angry
    customer/business stakeholder.

---

# PART 9 — DRIFT DETECTION

1. **What is data drift?** A change in the statistical distribution of
   input features between a reference period and the current period.
2. **What is concept drift?** A change in the underlying relationship
   between features and the target (i.e. the same feature values now mean
   something different for fraud likelihood) — distinct from data drift,
   which is about the inputs themselves, not the relationship.
   **This project detects data/output drift, not concept drift directly**
   — concept drift would require ground-truth labels over time to detect
   properly, which this project doesn't have access to at prediction
   time (see Q15).
3. **What is prediction drift?** A specific case of output drift — the
   model's own risk-score or decision distribution shifting, monitored
   here explicitly (`risk_score`/`decision` drift results).
4. **What is PSI?** Population Stability Index — a single number
   quantifying how much a distribution has shifted between a reference
   and a current period.
5. **PSI formula**: for each bin/category, `PSI = Σ (cur% - ref%) *
   ln(cur% / ref%)`, summed across all bins. Intuitive: each bin
   contributes more "surprise" the more its share of the population
   changed, weighted by the log-ratio of that change.
6. **PSI interpretation**: this project's thresholds —
   `PSI < 0.10 → NO_SIGNIFICANT_DRIFT`, `0.10-0.20 → LOW_DRIFT`,
   `0.20-0.30 → MODERATE_DRIFT`, `≥ 0.30 → HIGH_DRIFT`. **These are
   project monitoring conventions, based on a commonly-cited industry
   heuristic, not universal laws or a claim of statistical
   optimality for this specific application.**
7. **Why use PSI?** Standard, interpretable, single-number severity-banded
   summary for exactly this reference-vs-current comparison use case.
8. **What is KS?** Kolmogorov-Smirnov statistic — the maximum absolute
   difference between two cumulative distribution functions.
9. **Why PSI + KS together?** PSI is coarse (this project uses 10 bins)
   and can miss shape differences within a bin; KS is sensitive to any
   distributional difference, including shape changes PSI's binning might
   average away — genuinely complementary, not redundant.
10. **How are categorical variables handled?** Same PSI mechanic, treating
    categories as bins, with an added rare-category grouping step (see
    Part 10) to prevent long-tail sampling noise from inflating PSI.
11. **What happens to unseen categories?** Explicitly surfaced by name in
    the drift report (`new_categories`), never silently dropped.
12. **What are rare categories?** Categories below a 1% reference
    frequency threshold — grouped into a single `__OTHER__` bucket before
    computing PSI, a standard technique to avoid instability (see Part
    10's full story).
13. **Why can behavioral drift be natural?** Several behavioral features
    (`bhv_prev_txn_count`, `bhv_hist_mean_amt`) are cumulative counters
    that structurally grow over the dataset's timeline — an entity seen
    later has, by construction, more accumulated history than one seen
    earlier at its own point in time. A batch from a later period can
    show real PSI movement in these features that reflects this design,
    not a broken pipeline.
14. **How do you distinguish natural behavioral evolution from pipeline
    failure?** This project found a concrete example: comparing the
    VALIDATION reference against a real TEST-partition sample showed
    HIGH_DRIFT in behavioral features — investigated and traced to the
    structural cumulative-counter effect, not a bug, and documented as
    such rather than either silently dismissed or falsely alarmed on.
    **A general, automated way to make this distinction is not
    implemented** — it currently requires human interpretation of WHICH
    features are drifting and why.
15. **What would you do after detecting high drift?** **Current project**:
    surface it via `/drift/summary`/`/drift/analyze` as an investigation
    signal — nothing automated happens. **Production-scale extension**:
    a defined human investigation workflow, potentially triggering
    governance evaluation of a retrained candidate model — explicitly
    NOT implemented (drift and governance are structurally
    isolated by design, verified by a test that the drift module cannot
    even import the governance module).

---

# PART 10 — PSI BUG STORY (STAR)

**Situation**: while building the Phase 10 drift-monitoring demonstration
(comparing a "stable" batch against the reference), the categorical
feature `R_emaildomain` reported PSI ≈ 0.78 — flagged as HIGH_DRIFT — even
though the comparison batch was meant to represent genuinely stable data.

**Task**: determine whether this was a real drift signal or a bug in the
drift-detection methodology itself, before shipping a detector that could
cry wolf on stable data.

**Action**: investigated the reference profile directly and found
`R_emaildomain` has 55 distinct reference categories, many with less than
1% frequency (a long tail — `gmail.com` at ~43%, `hotmail.com` at ~22%,
down to dozens of email domains each under 1%). A moderate-sized current
batch will, purely from sampling variance, show zero occurrences for many
of these rare categories — and PSI's log-ratio formula treats each such
"surprise" as a real signal, summing dozens of small artifacts into one
large, misleading number.

**Result**: implemented the standard fix — group reference categories
below a 1% frequency threshold into a single `__OTHER__` bucket before
computing PSI (with the current batch grouped identically), while
separately preserving the FULL original category list so genuinely new
categories are still surfaced by name, never masked by the grouping. This
brought the same comparison down to a representative 0.19-0.31 depending
on the batch — still showing some residual movement, but no longer a
false HIGH_DRIFT alarm on stable data.

**Why this matters**: it's a concrete example of not trusting a metric's
output just because the code runs — investigating a suspicious result,
tracing it to a real statistical mechanism (long-tail sampling noise),
and fixing the methodology rather than either silently accepting a
misleading number or dismissing it without explanation.

---

# PART 11 — NUMERICAL STABILITY STORY (STAR)

**Situation**: the real-time engine's online variance computation used
`variance = (sum_sq_amt - count * mean²) / (count - 1)` — the same
formula as the offline feature computation. During Phase 6's real-data
validation, one specific real transaction showed a 0.113 absolute
difference between the offline and online computation of
`bhv_hist_std_amt` — small, but a genuine mismatch that shouldn't have
existed given both paths compute the same quantity.

**Task**: determine the root cause and decide whether/how to fix it,
during Phase 13's engineering audit.

**Action**: recognized this as a textbook catastrophic-cancellation
failure mode — for an entity with a high transaction count and low true
variance, `sum_sq_amt` and `count * mean²` become two very large, very
close floating-point numbers, and subtracting them loses precision to
rounding error. Constructed a synthetic adversarial test (20,000
transactions clustered tightly around a $1,000,000 base amount, true
std=$0.001) to quantify this directly: the naive formula produced
`naive_var = -0.0002` — an actual **negative variance**, which becomes
`NaN` after the square root. Replaced the formula with Welford's online
algorithm (`mean` and `M2`, the running sum of squared deviations from
the running mean, updated incrementally as `delta = x - mean; mean +=
delta/count; M2 += delta*(x - mean)`) in the real-time engine
(`bulk_initialize()` uses a stable two-pass computation per historical
block, merged via Chan et al.'s parallel-variance formula for
consistency with the incremental path).

**Result**: on the same adversarial input, the Welford-based
implementation achieves an error of ≈4.9e-12 against `numpy`'s own
independent ground-truth variance calculation — versus the old formula's
outright failure (`NaN`). Verified by a dedicated regression test
comparing both formulas against `numpy.std` directly. The offline batch
computation (`src/features/behavioral.py`) was deliberately left
unchanged — it's fully vectorized (not a natural fit for an inherently
incremental algorithm without a much larger rewrite) and its real-world
impact was already shown negligible (p99 absolute difference 7.4e-6
across 35,845 real comparisons) — a proportionate, scoped fix, not a
blanket rewrite.

**Why this matters**: it demonstrates diagnosing a subtle numerical bug
from first principles (recognizing the catastrophic-cancellation pattern),
constructing a targeted adversarial test to prove the failure mode and
the fix, and making a proportionate scoping decision (fix the online
path, document why the offline path was left alone) rather than either
ignoring the issue or over-engineering the fix.

**Welford's algorithm, conceptually**:
```text
Initialize: count = 0, mean = 0, M2 = 0
For each new value x:
    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean          (using the UPDATED mean)
    M2 += delta * delta2
Variance (sample) = M2 / (count - 1)
```
It never computes a large sum-of-squares or subtracts two large near-equal
numbers — every update is a small, local adjustment relative to the
current running mean, which is what makes it numerically stable.

---

# PART 12 — MODEL GOVERNANCE

1. **What is the champion model?** The currently approved, production-
   representative model — `fraud-risk-lightgbm-v1` in this project's
   registry, status `CHAMPION`.
2. **What is a candidate model?** A proposed model registered with status
   `CANDIDATE`, evaluated against the champion, but never automatically
   promoted.
3. **What is the model registry?** A local JSON file
   (`artifacts/models/registry.json`) tracking every registered model's
   metadata and lifecycle status.
4. **What is the artifact hash for?** A real SHA-256 hash of the actual
   trained model bundle file, computed and stored at registration time —
   lets you verify the deployed artifact hasn't silently changed.
5. **Why SHA-256 specifically?** A standard, collision-resistant
   cryptographic hash — the specific algorithm choice isn't unique to
   this use case, just a reliable standard one.
6. **What is schema compatibility?** One of the six promotion gates —
   confirms the candidate declares the same `feature_schema_version` as
   the champion, so a candidate expecting a different feature contract
   can't be silently promoted.
7. **What are the metric gates?** Two gates: required-metric-availability
   (both models have valid, non-NaN PR-AUC/ROC-AUC) and candidate-quality
   threshold (candidate PR-AUC must be ≥0.30, a project-configured
   minimum).
8. **What is the degradation gate?** Checks the candidate isn't more than
   0.02 PR-AUC worse than the champion; within 0.005 of that boundary,
   it returns REQUIRES_REVIEW rather than an outright PASS/FAIL.
9. **What is operational compatibility?** Checks the candidate doesn't
   increase the legitimate-customer-blocked rate by more than 0.5
   percentage points versus the champion.
10. **What is decision-policy compatibility?** Confirms both models
    reference the same frozen decision-policy version.
11. **How does promotion work?** `ModelRegistry.promote(candidate_id,
    approved_by, reason)` — requires explicit, human-supplied arguments;
    retires the previous champion (status → `RETIRED`, never deleted,
    full history preserved) and promotes the candidate.
12. **How does rollback work?** `ModelRegistry.rollback(target_id,
    approved_by, reason)` — restores a previously-RETIRED model to
    CHAMPION, retiring the current one in turn; only possible because
    `promote()` never deletes history.
13. **Why require approval/reason arguments?** To make every promotion/
    rollback explicit and auditable — no code path (drift detection
    included) can supply these on its own, so promotion can never happen
    silently or automatically.
14. **Why is governance needed at all?** Without it, a new model could
    replace the champion with no comparison, no audit trail, and no way
    to undo the change — governance is what makes model changes
    deliberate and reversible.
15. **What's the difference between REJECT and REQUIRES_REVIEW?** REJECT
    means at least one gate returned FAIL (a hard block); REQUIRES_REVIEW
    means no gate failed but at least one needs human judgment (e.g. a
    borderline degradation) — PROMOTE means every gate passed cleanly.

**On the synthetic candidate**: if asked to walk through a real
governance example, the honest answer is: *"That was a synthetic
governance demonstration used only to test the promotion mechanism — the
candidate's scores were a constructed blend of the real champion's own
scores plus label information, built specifically to exercise the gate
logic, never a second trained model. It's not a claim about any real
model's performance."* The real champion's real PR-AUC (0.5483) is the
only number that should ever be cited as model performance.

---

# PART 13 — SYSTEM DESIGN QUESTIONS

1. **Scaling — thousands to millions of transactions?** **Current
   project**: single-process FastAPI, in-memory state, untested beyond
   demo-scale request volumes. **Production-scale extension**: horizontal
   scaling behind a load balancer, externalized shared state (Redis/
   DynamoDB), async request handling, and a real load test to establish
   actual throughput limits — none of this has been measured here.

2. **State — how would you distribute it?** **Current**: single
   in-process object. **Extension**: partition by entity-hash across a
   distributed key-value store with atomic per-entity update operations,
   as discussed in Part 6/Q10.

3. **Reliability — what if the model service crashes?** **Current**:
   Docker's `HEALTHCHECK` would report unhealthy; there's no automatic
   restart/orchestration configured. **Extension**: run under an
   orchestrator (Kubernetes, ECS) with liveness/readiness probes and
   automatic restart, plus multiple replicas for redundancy (which
   requires solving the state-distribution problem first).

4. **Latency — how would you reduce it?** **Current**: one real measured
   example was ~155ms for a single request (not a load-tested figure).
   **Extension**: profile the actual bottleneck (feature computation vs.
   model inference vs. serialization) before optimizing; likely
   candidates include batching, a faster serialization format, or
   moving preprocessing into a compiled path — none of this profiling has
   been done here.

5. **Consistency — exactly-once state updates?** **Current**: not
   guaranteed — a client retry after a timeout could plausibly cause a
   duplicate state update (this project has not implemented
   idempotency keys or duplicate-detection). **Extension**: idempotency
   keys per transaction ID, with the engine checking "has this
   transaction ID already updated state" before applying an update.

6. **Deployment — AWS/Azure/GCP?** **Current**: local Docker only.
   **Extension**: push to a container registry, deploy via a managed
   container service, front with an API gateway/load balancer, manage
   secrets via a cloud secrets manager — genuinely straightforward given
   the existing Docker image, but not done here.

7. **Observability at production scale?** **Current**: Phase 9's
   Prometheus-text `/metrics` is real and scrape-compatible.
   **Extension**: an actual Prometheus server + Grafana dashboards +
   alerting rules — the format is ready for this, the infrastructure
   itself is not deployed.

8. **What happens when drift is detected?** **Current**: nothing
   automated — a human reads `/drift/summary` and decides. **Extension**:
   a defined escalation workflow (who gets paged, what threshold triggers
   what action) — not implemented, and deliberately not automated to
   governance (structurally isolated, see Part 9/Q15).

9. **How would you safely deploy a new model?** **Current**: exactly
   Phase 11's mechanism — register as candidate, evaluate, pass gates,
   explicit human promotion. **Extension**: a canary/staged rollout
   (route a small % of traffic to the new model, monitor, then ramp) —
   this project's promotion is immediate and total once approved, with no
   staged-traffic mechanism.

10. **How would you roll back instantly?** **Current**: `rollback()` is
    already instant in the sense that it's a single explicit registry
    call — but it does NOT automatically re-deploy a different running
    model artifact; wiring an approved rollback into the live serving
    engine (making `RiskDecisionEngine` actually swap models) is **not
    implemented** — governance and live serving are currently separate
    concerns.

---

# PART 14 — ADVERSARIAL INTERVIEWER

> **"Your PR-AUC is only 0.5483. Is that actually good?"**
It's roughly 15-16x the no-skill baseline for this fraud rate (0.0348),
and it's the number that drove every downstream decision in this
project honestly. It's not a claim of state-of-the-art performance
against external benchmarks — no such comparison was made — but it's a
real, meaningful, honestly-reported result on a genuinely hard dataset.

> **"Why should I trust Recall@2%?"**
Because it's computed directly from the held-out, untouched test set
using the frozen model and policy — you can re-derive it yourself from
`data/interim/phase4_metrics.json`. It's an honest number, not a
cherry-picked one; the report also shows Recall@1% and @5% so you can
see the full tradeoff curve, not just the flattering point.

> **"Why didn't you use XGBoost?"**
LightGBM was chosen upfront based on its established strengths for this
data profile (histogram-based splitting, native categorical support,
speed); a head-to-head LightGBM-vs-XGBoost comparison was **not run** —
that's a legitimate next experiment, not something this project claims
to have already ruled out empirically.

> **"Why not deep learning?"**
Tabular data with this many transactions and this feature mix is a
well-known strong domain for gradient-boosted trees; deep learning
typically needs either much larger data or careful architecture work
(e.g. entity embeddings) to match tree-based performance on tabular data,
and wasn't attempted here — an honest scope decision, not a claim that
it would necessarily be worse.

> **"Why not SHAP?"**
Not implemented in this project. The "Decision Signals" the dashboard
shows are the raw behavioral feature values that fed into a decision, NOT
a causal/SHAP-based attribution — deliberately labeled that way rather
than overclaiming an explainability method that isn't there.

> **"Why is `card1` acceptable?"**
It's the best available proxy in an anonymized dataset with no verified
customer ID — used with explicit, consistent "pseudo-entity" framing
throughout, never presented as a real identity. It's a real limitation,
acknowledged directly, not hidden.

> **"Isn't your behavioral feature just leakage?"**
No — and this is worth being precise about: leakage would mean using
information from the transaction being scored, or from the future. These
features use only transactions strictly BEFORE the current one, enforced
structurally in both the offline and online code paths, and verified by
dedicated tests. It's historical context, not leakage.

> **"Your behavioral improvement is only +0.0055. Why does it matter?"**
It's real, reproducible, and validation-confirmed — correctly described
as modest, not transformative. It matters because it's evidence the
feature engineering had a genuine, measured effect rather than being
assumed to help and shipped regardless.

> **"Isolation Forest performed poorly. Why include it at all?"**
Because a negative result, honestly measured and explained, is real
engineering evidence — it demonstrates the model-selection decision was
evidence-based, not assumption-based. It's presented as a rejected
experiment, not part of the shipped architecture.

> **"Your dataset isn't live. Why call this real-time?"**
"Real-time" describes the SERVING architecture — the engine genuinely
processes one transaction at a time, maintains and updates state
correctly, and returns a decision in real time when called — verified via
parity testing. It has never received live production traffic, and this
project doesn't claim otherwise; "real-time-capable" is the accurate
framing.

> **"Your monitoring isn't production-scale. Isn't this just a demo?"**
The monitoring CODE and its correctness (non-interference, accurate
counters) are real and tested. The TRAFFIC it's been exercised against is
demo/test-scale — that distinction is stated plainly, not obscured.

> **"What happens if fraud patterns change tomorrow?"**
Drift detection would surface a distributional shift as an investigation
signal (PSI/KS against the reference) — but nothing retrains or adapts
automatically. A human would need to investigate and, if warranted, train
and govern-in a new model through the Phase 11 process. This project
does not claim automated adaptation.

> **"What happens if the model becomes unavailable?"**
`GET /health` would report the failure; `/predict` requests would return
a 503 via the `engine_not_initialized` handler rather than crashing or
hanging. There's no automatic failover to a backup model — **not
implemented**.

> **"What is the biggest weakness of this project?"**
See Part 16 — answered directly, not deflected. The honest top answer:
58.1% of test-set fraud is still approved under the current policy, and
this system has never been validated against real, live traffic.

> **"You never load-tested this. How do you know it 'works'?"**
Correctness (does it produce the right answer, does it maintain state
correctly, does monitoring not interfere) was rigorously tested — 316
automated tests plus real Docker/API validation. Performance UNDER LOAD
was never measured, and I wouldn't claim it was.

> **"Why should I believe your offline/online parity numbers?"**
Because they came from an independent, separately-computed offline batch
calculation compared row-by-row against the live engine's output on the
same 3,000-transaction slice — not self-reported by the same code path
twice. That said, that specific run predates a later numerical-stability
fix (see Part 11) and hasn't been re-measured since — a limitation I'd
state upfront, not wait to be asked about.

> **"Doesn't Recall@5% = 60.14% mean you're still missing 40% of fraud
> even with a generous review budget?"**
Yes, exactly — and that's stated directly in this project's own
documentation, not glossed over. It's a real limitation of the current
model/policy combination, and a legitimate target for future improvement
(better features, tuned hyperparameters, or a larger review budget).

> **"Your governance system has never actually governed a real model
> change. Does it even work?"**
The MECHANISM was exercised end-to-end with real code paths — real
registration, real gate evaluation, a real REJECT and a real PROMOTE-then-
ROLLBACK cycle — using constructed candidate scores specifically to test
that mechanism. It has never governed a second REAL trained model, which
is an honest limitation, not a claim otherwise.

> **"Isn't a JSON file a toy model registry?"**
For this project's actual scale, yes, intentionally — it's explicitly
described as "lightweight local," not an enterprise MLOps platform. The
important part is the DISCIPLINE it enforces (explainable gates, no
silent promotion, hash-verified artifacts) — that discipline would carry
over to a more scalable backing store without changing the underlying
governance logic.

> **"Why is your test suite 316 tests impressive? Couldn't that just be
> a lot of trivial assertions?"**
Fair challenge — the count alone isn't the point. What matters is what
they cover: leakage guards with hand-computed expected values, offline/
online parity, non-interference verification for monitoring/drift/
governance, and structural checks (like the AST-based "drift can't
import governance" test) that go beyond "does it run."

---

# PART 15 — "WHY DID YOU CHOOSE X?" RAPID-FIRE

| Question | Strong Answer |
|---|---|
| Why LightGBM? | Strong tabular performance, native categorical support, fast iteration. |
| Why PR-AUC? | Positive class is rare (3.5%); PR-AUC is the metric that actually reflects that. |
| Why chronological split? | Mirrors real deployment — predicting the future from the past, never the reverse. |
| Why behavioral features? | A single transaction is less informative than its entity's recent pattern. |
| Why stateful inference? | Behavioral features need history without rescanning full logs per request. |
| Why FastAPI? | Typed validation, automatic docs, clean separation from the model internals. |
| Why Docker? | Reproducible, portable local deployment, independent of host environment. |
| Why PSI? | Standard, interpretable, severity-banded reference-vs-current comparison. |
| Why KS? | Complements PSI's coarse binning with shape-sensitive distributional comparison. |
| Why Welford? | Numerically stable online variance — avoids catastrophic cancellation. |
| Why governance gates? | Auditable, explainable promotion decisions — never a silent model swap. |
| Why three-way decisions? | Routes genuinely ambiguous cases to human review instead of a forced binary call. |

---

# PART 16 — PROJECT WEAKNESSES

**1. 58.1% of test-set fraud is still approved.** *Why it exists*: the
policy was tuned to balance BLOCK precision (84.85%) against legitimate-
customer friction (0.20% blocked); catching more fraud would mean
blocking/reviewing more legitimate traffic too. *How to explain it*: an
honest, quantified tradeoff, not a hidden flaw — state the number
directly. *How I'd improve it in production*: tune the policy against a
real, business-supplied cost ratio (cost of a missed fraud vs. cost of a
false block) rather than the illustrative validation-based selection used
here, and explore whether additional features would separate the classes
better in the first place.

**2. No ground-truth-based online accuracy monitoring.** *Why it exists*:
true fraud outcomes aren't available at prediction time (and in real
fraud systems, often arrive with significant delay via chargebacks/
disputes) — this project has no delayed-label ingestion mechanism. *How
to explain it*: drift detection compares distributions, not outcomes; it
cannot by itself confirm the model is "still accurate." *How I'd improve
it*: build a delayed-label ingestion pipeline to measure real online
precision/recall once outcomes become known, closing the loop between
drift signals and actual performance degradation.

**3. No horizontal scalability — state doesn't survive multiple
replicas correctly.** *Why it exists*: state is a single in-process
Python object; running multiple replicas would give each an independent,
inconsistent view. *How to explain it*: this is a real, structural
limitation of the current design, not an oversight — a conscious
first-version scoping choice. *How I'd improve it*: externalize state to
a distributed store with atomic per-entity operations before any
horizontal scaling.

**4. No authentication anywhere in the service.** *Why it exists*: this
project's scope never included a deployed, publicly-reachable service —
every endpoint, including the dev-only reset endpoint, is reachable by
anyone who can reach the process. *How to explain it*: acceptable for a
local portfolio system, a real gap for any public deployment. *How I'd
improve it*: add an authentication/authorization layer (API keys or
OAuth2) before any real external exposure.

**5. Never load-tested — no real latency/throughput numbers.** *Why it
exists*: this project's validation focused on correctness (does it
compute the right answer, does state update correctly, does monitoring
interfere) rather than performance benchmarking. *How to explain it*: one
real single-request latency measurement (~155ms) exists; no systematic
benchmark does. *How I'd improve it*: run a real load test (e.g. Locust
or k6) to establish actual throughput/latency-under-load numbers before
making any capacity claims.

---

# PART 17 — FUTURE IMPROVEMENTS

**Immediate ML improvements**: hyperparameter tuning (never swept —
`DEFAULT_PARAMS` used as-is); a genuine ablation of individual behavioral
features' contribution; broader feature engineering beyond the current 12
behavioral features.

**Feature improvements**: additional velocity windows beyond 1h/24h;
cross-entity features (e.g. shared device/IP signals, if such fields
existed reliably in the anonymized data); interaction features between
transaction-level and behavioral signals.

**Infrastructure improvements**: externalized, distributed behavioral
state (Redis/DynamoDB); horizontal API scaling; cloud deployment with a
real container registry and orchestrator; idempotency keys for
exactly-once state updates.

**Monitoring improvements**: real percentile latency tracking (not just
mean); an actual deployed Prometheus + Grafana stack (the text format is
ready, the infrastructure isn't); correlation IDs threaded through
structured logs for easier request tracing.

**Governance improvements**: a canary/staged-rollout mechanism instead of
immediate-and-total promotion; wiring an approved promotion/rollback into
the actual live serving engine (currently a separate concern from
`RiskDecisionEngine`); a formal model registry backing store beyond a
local JSON file at larger scale.

**Production-scale improvements**: delayed-label ingestion for real
online accuracy measurement; an authentication/authorization layer; a
defined drift-to-investigation escalation workflow; real load testing to
establish actual capacity limits before any scale claim.

None of the above are implemented — they are explicitly framed as future
work, not partially-built features.

---

# PART 18 — FINAL CHEAT SHEET

**Project**: An end-to-end fraud risk platform — leakage-safe modeling,
behavioral intelligence, a real-time stateful engine, and a full MLOps
layer (monitoring, drift detection, model governance) — built and tested
on the IEEE-CIS Fraud Detection dataset.

**Dataset**: IEEE-CIS Fraud Detection, 590,540 anonymized transactions,
394 columns, 3.499% fraud rate.

**Model**: LightGBM trained on transaction-level + historical behavioral
features, chronologically evaluated.

**Best metric**: PR-AUC — the metric that actually reflects performance
under 3.5% class imbalance.

**Strongest quantitative result**: Test PR-AUC 0.5483, Test ROC-AUC
0.9058, on the full 88,581-row held-out chronological test set.

**Behavioral features**: 12 `bhv_*` features per `card1` pseudo-entity,
strictly pre-current-transaction, giving a modest but real +0.0055
validation PR-AUC improvement.

**Decision policy**: a frozen three-way APPROVE/REVIEW/BLOCK policy,
84.85% precision on BLOCKed transactions, thresholds selected on
validation data only.

**Real-time architecture**: a stateful engine that scores a transaction
BEFORE updating that entity's history, verified via 100% offline/online
decision parity on a 3,000-transaction simulation.

**Monitoring**: a passive Prometheus-style observability layer, verified
to never interfere with predictions.

**Drift**: PSI + KS batch drift detection against a validation-partition
reference, including a real fix for a categorical long-tail sampling-noise
bug.

**Governance**: a local model registry with six explainable promotion
gates, real SHA-256 artifact hashing, and human-approved-only promotion/
rollback.

**Strongest engineering story**: diagnosing and fixing a real numerical-
stability bug (negative variance → NaN) using Welford's algorithm, proven
via a quantified adversarial test (error ≈4.9e-12 vs. NumPy ground
truth).

**Biggest limitation**: 58.1% of test-set fraud is still approved under
the current policy, and the system has never processed real, live
transaction traffic.

**Four numbers to memorize**:
1. **0.5483** — Test PR-AUC (the headline model result)
2. **0.9058** — Test ROC-AUC (paired context for the above)
3. **84.85%** — precision among BLOCKed transactions (the decision-policy result)
4. **316** — automated tests passing (the engineering-rigor number)
