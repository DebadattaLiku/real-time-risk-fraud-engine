"""
Phase 12: Drift Monitoring.

Exposes the real Phase 10 drift detector via `/drift/summary` (live
status) and the real reference profile artifact (offline: which features
are monitored, and what the reference distribution looks like). Does not
reimplement PSI/KS — everything numeric here comes from either the live
API or the saved reference-profile artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.utils import api_client, artifacts, ui
from dashboard.utils.session import get_api_base_url, check_api

st.set_page_config(page_title="Drift Monitoring — Fraud Risk Dashboard", page_icon="📈", layout="wide")

ui.page_header(
    "Drift Monitoring",
    "Phase 10's passive, batch-only comparison against the approved VALIDATION reference distribution.",
)

st.info(
    "**Important, carried over from the Phase 10 report:** some behavioral features "
    "(e.g. `bhv_prev_txn_count`, `bhv_hist_mean_amt`) are CUMULATIVE counters that "
    "naturally grow over the dataset's timeline. A batch drawn from a later chronological "
    "period than the reference can show real, measurable PSI movement in these features "
    "that reflects that structural design, not a data-quality problem. Not every behavioral "
    "PSI increase is a failure — see `reports/phase10_drift_detection_summary.md` for the "
    "concrete example this project found and investigated."
)

base_url = get_api_base_url()
api_available = check_api(base_url)
st.markdown(ui.api_status_badge(api_available))

# ---------------------------------------------------------------------------
# Live drift status
# ---------------------------------------------------------------------------
st.markdown("### Live Drift Status")
if api_available:
    ok, drift = api_client.get_drift_summary(base_url)
    if ok and drift.get("available"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Batches analyzed (this session)", drift["batches_analyzed"])
        c2.markdown(f"**Latest status**\n\n{ui.severity_badge(drift['latest_status']) if drift['latest_status'] else '_none yet_'}")
        c3.metric("Last analyzed", drift.get("latest_timestamp") or "n/a")
        if drift.get("latest_explanation"):
            st.write(f"**Latest explanation:** {drift['latest_explanation']}")
        if drift["severity_counts"]:
            st.markdown("**Severity history this session**")
            st.bar_chart(pd.Series(drift["severity_counts"], name="count"))
    elif ok:
        st.warning(
            "Drift monitor not available via the live API — no reference profile is loaded "
            "(run `python -m src.run_phase10_build_reference`)."
        )
    else:
        st.error(f"Could not retrieve /drift/summary: {drift}")
else:
    st.info("API not reachable — showing the offline reference profile below instead of live status.")

st.divider()

# ---------------------------------------------------------------------------
# Reference profile (offline, real artifact)
# ---------------------------------------------------------------------------
st.markdown("### Reference Profile (Offline Artifact)")
profile = artifacts.load_reference_profile()
if profile is None:
    st.warning("No reference profile found — run `python -m src.run_phase10_build_reference`.")
    st.stop()

meta = profile["metadata"]
c1, c2, c3 = st.columns(3)
c1.metric("Reference partition", meta["reference_partition"])
c2.metric("Reference transactions", f"{meta['n_reference_transactions']:,}")
c3.metric("Reference version", meta["reference_version"])
ui.source_caption("artifacts/drift/reference_profile.json (built once from real VALIDATION data, Phase 10)")

st.markdown("#### Monitored Signals")
tab1, tab2, tab3 = st.tabs(["Numeric", "Categorical", "Behavioral"])

with tab1:
    rows = []
    for name, entry in profile["numeric"].items():
        rows.append({"Feature": name, "Reference Mean": entry["mean"], "Reference Std": entry["std"],
                      "Min": entry["min"], "Max": entry["max"], "Missing %": entry.get("missing_pct")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Phase 2B's own top-10 LightGBM features by gain, plus TransactionAmt — not an arbitrary column list.")

with tab2:
    rows = []
    for name, entry in profile["categorical"].items():
        top_cats = sorted(entry["category_frequencies"].items(), key=lambda x: -x[1])[:5]
        rows.append({"Feature": name, "N Categories (reference)": entry["n_categories"],
                      "Top categories": ", ".join(f"{c}={f:.1%}" for c, f in top_cats)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    rows = []
    for name, entry in profile["behavioral"].items():
        rows.append({"Feature": name, "Reference Mean": entry["mean"], "Reference Std": entry["std"],
                      "Min": entry["min"], "Max": entry["max"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Exact Phase 4 behavioral feature names — see the caveat at the top of this page before reading PSI on these as pure data-quality signals.")

st.divider()

# ---------------------------------------------------------------------------
# Run a live drift batch (optional, interactive)
# ---------------------------------------------------------------------------
st.markdown("### Run a Drift Batch (Live API Only)")
st.caption(
    "Analyzes a small synthetic batch to demonstrate the `/drift/analyze` endpoint — "
    "NOT a claim that this batch represents real observed traffic."
)
if api_available:
    if st.button("Run demo batch (synthetic, clearly labeled)"):
        import numpy as np
        rng = np.random.default_rng(0)
        records = [
            {
                "TransactionAmt": float(rng.gamma(2.0, 50.0)),
                "ProductCD": rng.choice(["W", "C", "R"]),
                "risk_score": float(rng.beta(1, 20)),
                "decision": "APPROVE",
            }
            for _ in range(40)
        ]
        ok, result = api_client.analyze_drift_batch(records, base_url)
        if ok:
            st.success(f"Overall: {ui.severity_badge(result['overall_severity'])}")
            st.write(result["overall_explanation"])
            with st.expander("Full batch result"):
                st.json(result)
        else:
            st.error(f"Drift analysis failed: {result}")
else:
    st.info("Requires the live API.")
