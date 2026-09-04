"""
Phase 12: Shared UI helpers — small, presentation-only functions reused
across dashboard pages. No business logic lives here.
"""

from __future__ import annotations

import streamlit as st

SEVERITY_COLORS = {
    "NO_SIGNIFICANT_DRIFT": "🟢",
    "LOW_DRIFT": "🟡",
    "MODERATE_DRIFT": "🟠",
    "HIGH_DRIFT": "🔴",
}

DECISION_COLORS = {
    "APPROVE": "🟢",
    "REVIEW": "🟡",
    "BLOCK": "🔴",
}

RECOMMENDATION_COLORS = {
    "PROMOTE": "🟢",
    "REQUIRES_REVIEW": "🟡",
    "REJECT": "🔴",
}

GATE_RESULT_COLORS = {
    "PASS": "🟢",
    "REQUIRES_REVIEW": "🟡",
    "FAIL": "🔴",
}


def severity_badge(severity: str) -> str:
    return f"{SEVERITY_COLORS.get(severity, '⚪')} {severity.replace('_', ' ').title()}"


def decision_badge(decision: str) -> str:
    return f"{DECISION_COLORS.get(decision, '⚪')} {decision}"


def recommendation_badge(recommendation: str) -> str:
    return f"{RECOMMENDATION_COLORS.get(recommendation, '⚪')} {recommendation}"


def gate_result_badge(result: str) -> str:
    return f"{GATE_RESULT_COLORS.get(result, '⚪')} {result}"


def api_status_badge(available: bool) -> str:
    return "🟢 Live API connected" if available else "⚪ Local engine fallback (API not reachable)"


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def source_caption(text: str) -> None:
    """Small caption clarifying exactly where a number/chart came from —
    used throughout to distinguish offline evaluation artifacts, live API
    monitoring, and drift/governance state."""
    st.caption(f"📎 Source: {text}")
