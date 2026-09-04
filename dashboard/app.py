"""
Phase 12: Fraud Risk Intelligence Dashboard — entry point / Executive Overview.

Run with:
    streamlit run dashboard/app.py

This page (and every page under dashboard/pages/) is a presentation layer
only. It reuses the real FastAPI service (dashboard/utils/api_client.py)
when reachable, falls back to the real RiskDecisionEngine directly
(dashboard/utils/engine_fallback.py) otherwise, and reads real,
already-computed evaluation artifacts (dashboard/utils/artifacts.py) for
offline metrics. Nothing here retrains a model, changes a threshold, or
recomputes an evaluation metric differently from the established
protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.utils import api_client, artifacts, ui
from dashboard.utils.session import render_api_base_url_control, check_api

st.set_page_config(
    page_title="Fraud Risk Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("# 🛡️ Fraud Risk\n## Intelligence Dashboard")
    st.caption("Real-Time Risk Decision & Fraud Intelligence Engine — Phase 12")

base_url = render_api_base_url_control()
api_available = check_api(base_url)

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**API status**: {ui.api_status_badge(api_available)}")
    st.caption(
        "Pages prefer the live FastAPI service (`/predict`, `/monitoring/summary`, "
        "`/drift/summary`, `/model-governance/summary`) and fall back to reading the "
        "real local engine/artifacts directly when it isn't reachable."
    )
    st.markdown("---")
    st.caption("Use the **Pages** menu above for Live Scoring, Analytics, Monitoring, Drift, Governance, and Architecture.")

ui.page_header(
    "Executive Overview",
    "High-level system status — offline evaluation metrics, live policy configuration, and current operational state.",
)

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
st.markdown("### Model & Policy Identity")
col1, col2, col3 = st.columns(3)

registry = artifacts.load_model_registry()
champion = None
if registry and registry.get("champion_model_id"):
    champion = registry["models"][registry["champion_model_id"]]

policy_config = artifacts.load_decision_policy()

with col1:
    st.metric("Champion model", champion["model_id"] if champion else "not registered")
    if champion:
        ui.source_caption("artifacts/models/registry.json (Phase 11)")
with col2:
    st.metric("Model status", champion["status"] if champion else "—")
with col3:
    st.metric("Decision policy", policy_config["policy_name"] if policy_config else "—")
    if policy_config:
        ui.source_caption("config/decision_policy.yaml (Phase 5, frozen)")

st.divider()

# ---------------------------------------------------------------------------
# Offline evaluation KPIs (real Phase 4/5 test-set numbers)
# ---------------------------------------------------------------------------
st.markdown("### Offline Evaluation Metrics — Final Test Set (Phase 4/5)")
st.caption(
    "These are the project's one-time, final test-set numbers, read directly from "
    "`data/interim/phase4_metrics.json` and `phase5_metrics.json` — never recomputed here, "
    "and never re-used as a tuning signal (see reports/phase4_behavioral_features_summary.md)."
)

champ_metrics = artifacts.get_champion_test_metrics()
policy_test_eval = artifacts.get_champion_policy_test_eval()

if champ_metrics is None:
    st.warning("Phase 4 evaluation artifact not found — run Phase 4's pipeline to populate `data/interim/phase4_metrics.json`.")
else:
    ranking = champ_metrics["test_ranking"]
    budget_table = {row["budget"]: row for row in champ_metrics["test_budget_table"]}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test PR-AUC", f"{ranking['pr_auc']:.4f}")
    c2.metric("Test ROC-AUC", f"{ranking['roc_auc']:.4f}")
    c3.metric("Recall@1%", f"{budget_table.get(0.01, {}).get('recall_at_k', float('nan')):.2%}" if 0.01 in budget_table else "n/a")
    c4.metric("Recall@2%", f"{budget_table.get(0.02, {}).get('recall_at_k', float('nan')):.2%}" if 0.02 in budget_table else "n/a")
    c5.metric("Recall@5%", f"{budget_table.get(0.05, {}).get('recall_at_k', float('nan')):.2%}" if 0.05 in budget_table else "n/a")
    ui.source_caption("data/interim/phase4_metrics.json -> model_b_behavioral.test_ranking / test_budget_table")

st.divider()

# ---------------------------------------------------------------------------
# Decision policy configuration (live/frozen)
# ---------------------------------------------------------------------------
st.markdown("### Frozen Decision Policy Configuration")
if policy_config is None:
    st.warning("No decision policy configuration found.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Approve threshold", f"{policy_config['approve_threshold']:.4f}")
    c2.metric("Block threshold", f"{policy_config['block_threshold']:.4f}")
    c3.metric("Policy name", policy_config["policy_name"])
    with st.expander("Full policy configuration"):
        st.json(policy_config)
    ui.source_caption("config/decision_policy.yaml (Phase 5, selected on VALIDATION data only)")

if policy_test_eval:
    st.markdown("**Final test-set decision distribution (real, one-time evaluation):**")
    buckets = policy_test_eval["buckets"]
    c1, c2, c3 = st.columns(3)
    c1.metric("APPROVE", f"{buckets['APPROVE']['pct_of_total']:.2%}", f"{buckets['APPROVE']['count']:,} txns")
    c2.metric("REVIEW", f"{buckets['REVIEW']['pct_of_total']:.2%}", f"{buckets['REVIEW']['count']:,} txns")
    c3.metric("BLOCK", f"{buckets['BLOCK']['pct_of_total']:.2%}", f"{buckets['BLOCK']['count']:,} txns")
    ui.source_caption("data/interim/phase5_metrics.json -> test_eval")

st.divider()

# ---------------------------------------------------------------------------
# Live system status
# ---------------------------------------------------------------------------
st.markdown("### Live System Status")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**API / Engine health**")
    if api_available:
        ok, health = api_client.get_health(base_url)
        if ok:
            st.success(f"API: {health['status']} (model_loaded={health['model_loaded']}, policy_loaded={health['policy_loaded']})")
        else:
            st.error("API reachable but /health returned an error.")
    else:
        st.info("API not reachable — local engine fallback available on the Live Scoring page.")

with c2:
    st.markdown("**Drift status**")
    ok, drift = (api_client.get_drift_summary(base_url) if api_available else (False, None))
    if ok and drift.get("available"):
        latest = drift.get("latest_status")
        if latest:
            st.markdown(ui.severity_badge(latest))
        else:
            st.info("No batches analyzed yet this session.")
        st.caption(f"Batches analyzed: {drift.get('batches_analyzed', 0)}")
    else:
        st.info("Drift monitor not reachable via live API — see the Drift page for offline reference-profile details.")

with c3:
    st.markdown("**Governance status**")
    ok, gov = (api_client.get_governance_summary(base_url) if api_available else (False, None))
    if ok and gov.get("available"):
        st.markdown(f"Champion: `{gov['champion_model_id']}`")
        st.caption(f"Candidates registered: {gov.get('n_candidates', 0)}")
    else:
        reg = artifacts.load_model_registry()
        if reg:
            st.markdown(f"Champion (from registry file): `{reg['champion_model_id']}`")
        else:
            st.info("No governance registry found.")

st.divider()

# ---------------------------------------------------------------------------
# Cached test suite status
# ---------------------------------------------------------------------------
st.markdown("### Automated Test Suite (cached, not live)")
test_summary = artifacts.read_cached_test_summary()
if test_summary:
    c1, c2 = st.columns(2)
    c1.metric("Tests passed (last recorded run)", test_summary.get("n_passed", "n/a"))
    c2.metric("All passed?", "✅ Yes" if test_summary.get("all_passed") else "❌ No")
    st.caption(
        f"Recorded at {test_summary.get('recorded_at')} — via "
        f"`python scripts/refresh_dashboard_test_summary.py`. The dashboard never runs "
        f"pytest live from inside a page load."
    )
else:
    st.info(
        "No cached test summary yet. Run `python scripts/refresh_dashboard_test_summary.py` "
        "to populate this section from a real `pytest -q` run."
    )
