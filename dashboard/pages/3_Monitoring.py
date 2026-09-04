"""
Phase 12: Monitoring Dashboard.

Consumes the real Phase 9 `/monitoring/summary` endpoint (and `/metrics`
for the raw Prometheus text) directly — no competing monitoring system.
Requires the live API; there is no meaningful "offline" monitoring state
to fall back to (in-memory counters only exist in a running process).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.utils import api_client, ui
from dashboard.utils.session import get_api_base_url, check_api

st.set_page_config(page_title="Monitoring — Fraud Risk Dashboard", page_icon="📡", layout="wide")

ui.page_header(
    "Monitoring Dashboard",
    "Live service observability — Phase 9's request/prediction/decision/error metrics.",
)

base_url = get_api_base_url()
api_available = check_api(base_url)
st.markdown(ui.api_status_badge(api_available))

if not api_available:
    st.warning(
        "This page requires the live FastAPI service — in-memory monitoring counters only "
        "exist inside a running process, so there is no meaningful offline fallback. "
        "Start the API (`uvicorn src.api.main:app`) and refresh this page."
    )
    st.stop()

ok, summary = api_client.get_monitoring_summary(base_url)
if not ok:
    st.error(f"Could not retrieve /monitoring/summary: {summary}")
    st.stop()

st.caption(f"Uptime: {summary['uptime_seconds']:.1f}s — resets to zero on process restart (in-memory only, Phase 9 design).")

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
st.markdown("### Request Activity")
req = summary["requests"]
c1, c2, c3 = st.columns(3)
c1.metric("Total requests", req["total"])
c2.metric("Avg latency (all endpoints)", f"{req['avg_latency_ms']:.1f} ms" if req["avg_latency_ms"] is not None else "n/a")
c3.metric("Distinct endpoints hit", len(req["by_endpoint"]))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Requests by endpoint**")
    if req["by_endpoint"]:
        st.bar_chart(pd.Series(req["by_endpoint"], name="count"))
    else:
        st.info("No requests recorded yet.")
with c2:
    st.markdown("**Requests by HTTP status class**")
    if req["by_status_class"]:
        st.bar_chart(pd.Series(req["by_status_class"], name="count"))
    else:
        st.info("No requests recorded yet.")

st.divider()

# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------
st.markdown("### Prediction Activity")
pred = summary["predictions"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prediction attempts", pred["attempts"])
c2.metric("Successful predictions", pred["successes"])
c3.metric("Failed predictions", pred["failures"])
c4.metric("Avg engine latency", f"{pred['avg_engine_latency_ms']:.1f} ms" if pred["avg_engine_latency_ms"] is not None else "n/a")

st.divider()

# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------
st.markdown("### Decision Counts (this process session)")
dec = summary["decisions"]
c1, c2 = st.columns([1, 2])
with c1:
    for d, count in dec["counts"].items():
        st.markdown(f"{ui.decision_badge(d)}: **{count}** ({dec['distribution'].get(d, 0):.1%})" if dec["total"] else f"{ui.decision_badge(d)}: **{count}**")
with c2:
    if dec["total"] > 0:
        st.bar_chart(pd.Series(dec["counts"], name="count"))
    else:
        st.info("No decisions recorded yet this session — score a transaction on the Live Scoring page.")

st.divider()

# ---------------------------------------------------------------------------
# Risk score statistics
# ---------------------------------------------------------------------------
st.markdown("### Risk-Score Statistics (this process session)")
rs = summary["risk_scores"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scores observed", rs["count"])
c2.metric("Mean", f"{rs['mean']:.4f}" if rs["mean"] is not None else "n/a")
c3.metric("Min", f"{rs['min']:.4f}" if rs["min"] is not None else "n/a")
c4.metric("Max", f"{rs['max']:.4f}" if rs["max"] is not None else "n/a")

if rs["bucket_counts"]:
    st.markdown("**Risk-score distribution (decile buckets)**")
    bucket_df = pd.Series(rs["bucket_counts"], name="count").reindex(
        [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(rs["bucket_edges"][:-1], rs["bucket_edges"][1:])], fill_value=0
    )
    st.bar_chart(bucket_df)
else:
    st.info("No risk scores recorded yet this session.")

st.divider()

# ---------------------------------------------------------------------------
# Data quality / error metrics
# ---------------------------------------------------------------------------
st.markdown("### Data Quality & Errors")
err = summary["errors"]
dq = summary["data_quality"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Request validation failures", err["request_validation_failures"], help="Rejected by request-schema validation — never reached the engine.")
c2.metric("Engine validation failures", err["engine_validation_failures"], help="Rejected by the engine's own validation (e.g. isFraud present).")
c3.metric("Engine-not-initialized errors", err["engine_not_initialized_errors"])
c4.metric("Internal errors", err["internal_errors"])

amt = dq["transaction_amount"]
if amt["count"] > 0:
    st.markdown("**Observed transaction amount summary (successful predictions only)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Count", amt["count"])
    c2.metric("Mean", f"${amt['mean']:.2f}")
    c3.metric("Min", f"${amt['min']:.2f}")
    c4.metric("Max", f"${amt['max']:.2f}")

st.divider()

with st.expander("Raw Prometheus-format /metrics (excerpt)"):
    ok, text = api_client.get_metrics_text(base_url)
    if ok:
        st.code(text, language="text")
    else:
        st.info("Could not retrieve /metrics.")

with st.expander("Full raw /monitoring/summary JSON"):
    st.json(summary)
