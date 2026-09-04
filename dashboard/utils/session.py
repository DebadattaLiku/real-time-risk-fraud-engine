"""
Phase 12: Shared session-state helpers.

`get_api_base_url()` reads a single sidebar-configurable value shared by
every page. `get_local_engine()` builds (once per Streamlit process, via
`st.cache_resource`) the local `RiskDecisionEngine` fallback — reusing
`dashboard/utils/engine_fallback.py`, itself a thin wrapper around the
real `src.api.dependencies.build_engine`.
"""

from __future__ import annotations

import streamlit as st

from dashboard.utils.api_client import DEFAULT_BASE_URL, is_api_available
from dashboard.utils.engine_fallback import build_engine_for_dashboard

API_BASE_URL_KEY = "fraud_dashboard_api_base_url"


def get_api_base_url() -> str:
    if API_BASE_URL_KEY not in st.session_state:
        st.session_state[API_BASE_URL_KEY] = DEFAULT_BASE_URL
    return st.session_state[API_BASE_URL_KEY]


def render_api_base_url_control() -> str:
    with st.sidebar:
        st.text_input(
            "API base URL", value=get_api_base_url(), key=API_BASE_URL_KEY,
            help="The dashboard prefers this live FastAPI service. If it's unreachable, "
                 "pages fall back to a local RiskDecisionEngine instance automatically.",
        )
    return get_api_base_url()


@st.cache_resource(show_spinner="Loading local RiskDecisionEngine (cold start, no warm-up)...")
def get_local_engine():
    """Cached for the lifetime of the Streamlit process — the model bundle
    and policy are loaded once, not on every page interaction."""
    return build_engine_for_dashboard(warm_start=False)


def check_api(base_url: str) -> bool:
    return is_api_available(base_url)
