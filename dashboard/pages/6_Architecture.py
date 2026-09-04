"""
Phase 12: Architecture / System Flow — a static, presentation-only page.
No logic, no data calls; just a clear picture of how the real system
(Phases 1-11) fits together.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.utils import ui

st.set_page_config(page_title="Architecture — Fraud Risk Dashboard", page_icon="🧭", layout="wide")

ui.page_header("System Architecture", "How a transaction actually flows through the approved system.")

st.markdown("### Real-Time Inference Path")
st.code(
    """
Transaction
    |
    v
FastAPI  (POST /predict -- Phase 7)
    |
    v
RiskDecisionEngine.process_transaction()  (Phase 6)
    |
    +- STEP 1-2: Receive + validate
    |
    +- STEP 3-4: Read EXISTING behavioral state, generate bhv_* features
    |            (Phase 4 formulas -- current transaction NEVER
    |             contributes to its own features)
    |
    +- STEP 5-6: Combine transaction + behavioral features,
    |            apply Phase 1 FeaturePipeline + Phase 4 LightGBMPreprocessor
    |
    +- STEP 7: LightGBM.predict_proba() -> fraud risk score
    |
    +- STEP 8: Apply frozen Phase 5 DecisionPolicy -> APPROVE/REVIEW/BLOCK
    |
    +- STEP 9: Return result
    |
    +- STEP 10: Update behavioral state -- ONLY NOW, AFTER the
                 prediction/decision are already finalized
    """,
    language="text",
)

st.warning(
    "**State update happens LAST — after prediction and decision are already computed.** "
    "This is the concrete mechanism that prevents a transaction from ever influencing its "
    "own behavioral features (Phase 6's core leakage-safety guarantee)."
)

st.divider()

st.markdown("### Cross-Cutting Concerns (Observation, Not Interference)")
st.code(
    """
                    +-----------------------------+
                    |   RiskDecisionEngine         |
                    |   (real-time inference path) |
                    +---------------+---------------+
                                    |  (read-only observation, AFTER the fact)
        +---------------+-----------+-----------+---------------+
        v               v           v           v               v
   Observability     Structured   Prometheus   Drift          Governance
   (Phase 9)         Logging      /metrics     Detection      Registry
        |                                      (Phase 10)     (Phase 11)
        v                                          |               |
   MetricsRegistry                            DriftMonitor    ModelRegistry
   (in-memory, resets                         (batch-only,    (JSON file,
    on restart)                                separate from   explicit
                                                /predict)       promote()/
                                                                rollback() only)
    """,
    language="text",
)

st.markdown(
    """
    **None of Observability, Drift Detection, or Governance can change a risk score, a
    decision, or behavioral state.** Each was validated with a dedicated
    "non-interference" test comparing direct-engine output against fully-monitored
    API output (Phases 9, 10, 11 reports) — scores matched to `<1e-9`, decisions
    were identical.
    """
)

st.divider()

st.markdown("### Full Phase Map")
phases = [
    ("0", "Leakage-aware EDA"),
    ("1", "Chronological train/validation/test split, leakage-safe pipeline"),
    ("2A/2B", "Logistic Regression baseline, LightGBM champion"),
    ("3", "Isolation Forest complementarity study (rejected)"),
    ("4", "Leakage-safe behavioral features (card1 pseudo-entity)"),
    ("5", "APPROVE / REVIEW / BLOCK decision policy (frozen)"),
    ("6", "Stateful real-time RiskDecisionEngine + offline/online parity"),
    ("7", "FastAPI inference service"),
    ("8", "Docker containerization"),
    ("9", "Monitoring & observability (Prometheus-style /metrics)"),
    ("10", "PSI + KS drift detection"),
    ("11", "Model lifecycle governance (registry, promotion gates, rollback)"),
    ("12", "This dashboard — presentation layer only"),
]
for phase, desc in phases:
    st.markdown(f"- **Phase {phase}**: {desc}")

st.caption("See `reports/phase*_summary.md` for each phase's full write-up and real, executed validation results.")
