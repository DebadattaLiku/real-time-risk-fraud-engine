"""
Phase 12: Live Transaction Risk Scoring + Decision Explanation.

Prefers the real FastAPI `/predict` endpoint (dashboard/utils/api_client.py);
falls back to the real RiskDecisionEngine directly
(dashboard/utils/engine_fallback.py) when the API isn't reachable. No
second inference implementation exists anywhere in this dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.utils import api_client, ui
from dashboard.utils.session import get_api_base_url, get_local_engine, check_api

st.set_page_config(page_title="Live Scoring — Fraud Risk Dashboard", page_icon="🎯", layout="wide")

ui.page_header(
    "Live Transaction Risk Scoring",
    "Submit a transaction and get a real fraud risk score + APPROVE/REVIEW/BLOCK decision.",
)

base_url = get_api_base_url()
api_available = check_api(base_url)
st.markdown(ui.api_status_badge(api_available))

EXAMPLE_TRANSACTIONS = {
    "Low-value, low-risk example": {
        "TransactionID": 9100001, "TransactionDT": 5000000, "TransactionAmt": 25.0,
        "ProductCD": "W", "card1": 9500, "card4": "visa", "card6": "debit", "addr1": 204.0,
    },
    "Higher-value example": {
        "TransactionID": 9100002, "TransactionDT": 5000000, "TransactionAmt": 850.0,
        "ProductCD": "C", "card1": 14200, "card4": "mastercard", "card6": "credit", "addr1": 330.0,
    },
}

with st.form("transaction_form"):
    st.markdown("### Transaction Input")
    preset = st.selectbox("Load an example (optional)", ["— manual entry —"] + list(EXAMPLE_TRANSACTIONS.keys()))
    defaults = EXAMPLE_TRANSACTIONS.get(preset, {})

    c1, c2, c3 = st.columns(3)
    with c1:
        transaction_id = st.number_input("TransactionID", min_value=1, value=int(defaults.get("TransactionID", 9000000)), step=1)
        transaction_dt = st.number_input("TransactionDT (seconds, dataset-relative)", min_value=0, value=int(defaults.get("TransactionDT", 5000000)), step=1)
        transaction_amt = st.number_input("TransactionAmt ($)", min_value=0.0, value=float(defaults.get("TransactionAmt", 100.0)), step=1.0)
    with c2:
        product_cd = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"], index=["W", "C", "R", "H", "S"].index(defaults.get("ProductCD", "W")))
        card1 = st.number_input("card1 (pseudo-entity id — NOT a verified customer ID)", min_value=1, value=int(defaults.get("card1", 10000)), step=1)
        card4 = st.selectbox("card4 (network)", ["visa", "mastercard", "american express", "discover"],
                              index=["visa", "mastercard", "american express", "discover"].index(defaults.get("card4", "visa")))
    with c3:
        card6 = st.selectbox("card6 (type)", ["debit", "credit"], index=["debit", "credit"].index(defaults.get("card6", "debit")))
        addr1 = st.number_input("addr1 (optional, leave 0 for unknown)", min_value=0.0, value=float(defaults.get("addr1", 0.0)), step=1.0)

    submitted = st.form_submit_button("Score Transaction", type="primary")

if submitted:
    transaction = {
        "TransactionID": int(transaction_id), "TransactionDT": float(transaction_dt),
        "TransactionAmt": float(transaction_amt), "ProductCD": product_cd,
        "card1": int(card1), "card4": card4, "card6": card6,
    }
    if addr1 > 0:
        transaction["addr1"] = float(addr1)

    result = None
    result_source = None
    error = None

    if api_available:
        ok, response = api_client.predict(transaction, base_url)
        if ok:
            result = response
            result_source = "live API (POST /predict)"
        else:
            error = response
    if result is None:
        try:
            engine = get_local_engine()
            from dashboard.utils.engine_fallback import score_transaction_locally
            ok, response = score_transaction_locally(engine, transaction)
            if ok:
                result = response
                result_source = "local RiskDecisionEngine fallback (API unavailable or rejected the request)"
            else:
                error = response
        except Exception as e:
            error = {"error": "local_engine_error", "detail": str(e)}

    st.divider()
    if result is None:
        st.error(f"Scoring failed: {error}")
    else:
        st.success(f"Scored via: {result_source}")
        risk_score = result["risk_score"]
        decision = result["decision"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{risk_score:.4f}", help="Fraud probability in [0, 1]. HIGHER = higher predicted fraud risk.")
        c2.markdown(f"### Decision\n{ui.decision_badge(decision)}")
        c3.metric("Transaction ID", result.get("transaction_id", transaction["TransactionID"]))

        st.markdown("### Decision Signals")
        st.caption(
            "This project has NOT implemented SHAP or another causal feature-attribution "
            "method — the values below are the raw signals that fed into the decision, "
            "not a claim about which one 'caused' it."
        )

        policy_metadata = result.get("processing_metadata", {})
        approve_thr = policy_metadata.get("approve_threshold")
        block_thr = policy_metadata.get("block_threshold")
        if approve_thr is not None and block_thr is not None:
            st.write(f"**Policy thresholds**: approve < `{approve_thr:.4f}` | review between | block >= `{block_thr:.4f}`")

        if "behavioral_features" in result:
            st.markdown("**Behavioral signals** (from `card1` pseudo-entity history — internal to the engine, "
                         "available here because this used the local engine directly):")
            bhv = result["behavioral_features"]
            bhv_display = {k: (f"{v:.4f}" if isinstance(v, float) and v == v else v) for k, v in bhv.items()}
            st.dataframe(bhv_display, use_container_width=True)
        else:
            st.info(
                "Behavioral feature values are internal to the engine and NOT exposed via the "
                "public `/predict` API response (a deliberate Phase 7 design choice — see "
                "`src/api/schemas.py`). They would be visible here if this request had used the "
                "local engine fallback instead."
            )

        with st.expander("Full raw result"):
            st.json(result)

        with st.expander("Transaction submitted"):
            st.json(transaction)
