"""
Phase 12: Model Governance.

Shows the REAL current champion (live `/model-governance/summary` or the
real `artifacts/models/registry.json`) clearly separated from the
Phase 11 MOCK candidate governance demonstration
(`reports/model_governance/*.json`) — the mock candidate's PR-AUC of
0.8700 is a synthetic label-informed blend used ONLY to exercise the
promotion-gate mechanics, never a real model performance result, and is
labeled as such everywhere it appears on this page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.utils import api_client, artifacts, ui
from dashboard.utils.session import get_api_base_url, check_api

st.set_page_config(page_title="Governance — Fraud Risk Dashboard", page_icon="🏛️", layout="wide")

ui.page_header(
    "Model Governance",
    "Phase 11's lightweight local model lifecycle registry — champion identity, candidates, and promotion gates.",
)

base_url = get_api_base_url()
api_available = check_api(base_url)
st.markdown(ui.api_status_badge(api_available))

# ---------------------------------------------------------------------------
# Current champion — REAL
# ---------------------------------------------------------------------------
st.markdown("### Current Champion (Real)")

gov_summary = None
if api_available:
    ok, gov = api_client.get_governance_summary(base_url)
    if ok and gov.get("available"):
        gov_summary = gov

registry = artifacts.load_model_registry()
champion = None
if registry and registry.get("champion_model_id"):
    champion = registry["models"][registry["champion_model_id"]]

if gov_summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion model ID", gov_summary["champion_model_id"])
    c2.metric("Version", gov_summary["champion_version"])
    c3.metric("Status", gov_summary["champion_status"])
    c4.metric("Candidates registered", gov_summary["n_candidates"])
    ui.source_caption("Live API: GET /model-governance/summary")
elif champion:
    c1, c2, c3 = st.columns(3)
    c1.metric("Champion model ID", champion["model_id"])
    c2.metric("Version", champion["model_version"])
    c3.metric("Status", champion["status"])
    ui.source_caption("artifacts/models/registry.json (offline — API not reachable)")
else:
    st.warning("No governance registry found — run `python -m src.run_phase11_register_champion`.")
    st.stop()

if champion:
    with st.expander("Full champion metadata (real)"):
        st.json(champion)
    st.caption(
        f"Artifact SHA-256: `{champion.get('artifact_sha256', 'n/a')}` — "
        f"a real hash of the real trained model bundle, computed at registration time."
    )

st.divider()

# ---------------------------------------------------------------------------
# Governance demonstration — MOCK, clearly labeled
# ---------------------------------------------------------------------------
st.markdown("### Governance Mechanism Demonstration")
st.warning(
    "⚠️ **Everything below this line is from Phase 11's demonstration run "
    "(`scripts/demo_model_governance.py`), using MOCK/SYNTHETIC candidates — "
    "NOT real trained fraud models and NOT the current live registry state.** "
    "The 'better' mock candidate's PR-AUC (0.8700) is a synthetic, label-informed "
    "blend constructed ONLY to demonstrate the promotion-gate mechanics passing. "
    "It is never a claim that a real model achieving that performance exists."
)

reports_dir = REPO_ROOT / "reports" / "model_governance"
worse_report_path = reports_dir / "champion_vs_mock_worse_candidate_report.json"
better_report_path = reports_dir / "champion_vs_mock_better_candidate_report.json"

for label, path in [("Mock 'worse' candidate (expected REJECT)", worse_report_path),
                     ("Mock 'better' candidate (expected PROMOTE)", better_report_path)]:
    if not path.is_file():
        continue
    with open(path) as f:
        report = json.load(f)

    st.markdown(f"#### {label}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Recommendation**\n\n{ui.recommendation_badge(report['recommendation'])}")
    c2.metric("Champion PR-AUC (real)", f"{report['metrics_comparison']['ranking_metrics']['champion']['pr_auc']:.4f}")
    c3.metric("Mock candidate PR-AUC (synthetic)", f"{report['metrics_comparison']['ranking_metrics']['candidate']['pr_auc']:.4f}")
    st.write(report["recommendation_explanation"])

    st.markdown("**Promotion gates**")
    for g in report["promotion_gates"]:
        st.markdown(f"{ui.gate_result_badge(g['result'])} — `{g['gate']}`: {g['explanation']}")

    with st.expander("Full report JSON"):
        st.json(report)
    st.divider()

if not worse_report_path.is_file() and not better_report_path.is_file():
    st.info(
        "No governance demonstration reports found yet. Run "
        "`python scripts/demo_model_governance.py` to generate them "
        "(uses a throwaway demo registry — does not affect the real one)."
    )

st.markdown("### Promotion & Rollback — How They Actually Work")
st.markdown(
    """
    - Promotion and rollback are **explicit, human-invoked actions only** —
      `ModelRegistry.promote(candidate_id, approved_by, reason)` and
      `ModelRegistry.rollback(target_id, approved_by, reason)`, both requiring
      a real `approved_by`/`reason` string.
    - **No API route, and no part of this dashboard, can call either method.**
      This page is read-only, exactly like `GET /model-governance/summary`.
    - Phase 10's drift detector cannot trigger promotion or rollback either —
      verified structurally (a test parses every file in `src/drift/` and
      confirms none of them import anything from `src/governance/`).
    """
)
