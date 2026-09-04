"""
Phase 12: Risk & Decision Analytics.

Visualizations built entirely from real, already-computed evaluation
artifacts (Phase 4/5) — no metric here is recomputed differently from the
established chronological-split protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.utils import artifacts, ui

st.set_page_config(page_title="Analytics — Fraud Risk Dashboard", page_icon="📊", layout="wide")

ui.page_header(
    "Risk & Decision Analytics",
    "Built from the project's real, one-time final test-set evaluation artifacts (Phase 4/5).",
)

champ_metrics = artifacts.get_champion_test_metrics()
policy_test_eval = artifacts.get_champion_policy_test_eval()

if champ_metrics is None or policy_test_eval is None:
    st.warning("Evaluation artifacts not found — run Phases 4/5's pipelines first.")
    st.stop()

# ---------------------------------------------------------------------------
# APPROVE / REVIEW / BLOCK distribution
# ---------------------------------------------------------------------------
st.markdown("### Decision Distribution — Final Test Set")
buckets = policy_test_eval["buckets"]
dist_df = pd.DataFrame({
    "Decision": ["APPROVE", "REVIEW", "BLOCK"],
    "Count": [buckets["APPROVE"]["count"], buckets["REVIEW"]["count"], buckets["BLOCK"]["count"]],
})
c1, c2 = st.columns([2, 1])
with c1:
    st.bar_chart(dist_df.set_index("Decision"), color="#4C72B0")
with c2:
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
ui.source_caption("data/interim/phase5_metrics.json -> test_eval.buckets")

st.divider()

# ---------------------------------------------------------------------------
# Fraud vs legitimate, per decision bucket
# ---------------------------------------------------------------------------
st.markdown("### Fraud vs. Legitimate Transactions per Decision Bucket")
st.caption("Real labels available for this final test-set evaluation only.")
fraud_legit_df = pd.DataFrame({
    "Decision": ["APPROVE", "REVIEW", "BLOCK"],
    "Fraud": [buckets["APPROVE"]["fraud_count"], buckets["REVIEW"]["fraud_count"], buckets["BLOCK"]["fraud_count"]],
    "Legitimate": [buckets["APPROVE"]["legitimate_count"], buckets["REVIEW"]["legitimate_count"], buckets["BLOCK"]["legitimate_count"]],
}).set_index("Decision")
st.bar_chart(fraud_legit_df)
c1, c2, c3 = st.columns(3)
c1.metric("Total fraud (test set)", f"{policy_test_eval['total_fraud_cases']:,}")
c2.metric("Fraud captured (REVIEW+BLOCK)", f"{policy_test_eval['fraud_captured_review_plus_block']:,}")
c3.metric("Fraud missed (APPROVE)", f"{policy_test_eval['fraud_missed_in_approve']:,}")
ui.source_caption("data/interim/phase5_metrics.json -> test_eval")

st.divider()

# ---------------------------------------------------------------------------
# Recall / precision at review budgets
# ---------------------------------------------------------------------------
st.markdown("### Recall & Precision at Fixed Review Budgets")
budget_table = champ_metrics["test_budget_table"]
budget_df = pd.DataFrame(budget_table)
budget_df["budget_pct"] = (budget_df["budget"] * 100).astype(str) + "%"

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Recall@K**")
    st.bar_chart(budget_df.set_index("budget_pct")["recall_at_k"])
with c2:
    st.markdown("**Precision@K**")
    st.bar_chart(budget_df.set_index("budget_pct")["precision_at_k"])
st.dataframe(
    budget_df[["budget_pct", "recall_at_k", "precision_at_k"]].rename(
        columns={"budget_pct": "Review Budget", "recall_at_k": "Recall", "precision_at_k": "Precision"}
    ),
    use_container_width=True, hide_index=True,
)
ui.source_caption("data/interim/phase4_metrics.json -> model_b_behavioral.test_budget_table")

st.divider()

# ---------------------------------------------------------------------------
# Review/block operational workload
# ---------------------------------------------------------------------------
st.markdown("### Operational Workload & Customer Friction")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Review queue size", f"{buckets['REVIEW']['count']:,}", f"{buckets['REVIEW']['pct_of_total']:.2%} of traffic")
c2.metric("Blocked", f"{buckets['BLOCK']['count']:,}", f"{buckets['BLOCK']['pct_of_total']:.2%} of traffic")
c3.metric("Precision among blocked", f"{policy_test_eval['precision_among_blocked']:.2%}")
c4.metric("Legitimate customers blocked", f"{policy_test_eval['legitimate_transactions_blocked']:,}",
          f"{policy_test_eval['pct_legitimate_blocked']:.4%} of all legitimate traffic")
ui.source_caption("data/interim/phase5_metrics.json -> test_eval (see reports/phase5_decision_policy_summary.md for full discussion)")
