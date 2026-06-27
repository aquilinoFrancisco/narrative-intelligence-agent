"""
results.py
Streamlit component — pipeline results display.

Renders the full NarrativeState produced by graph.invoke(), including:
  - Threat level banner
  - Key metrics row
  - Top narratives table
  - Coordination alerts
  - Executive briefing
  - Report path
  - Error log (if any)

No business logic.  All data comes from the NarrativeState dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


# Risk level → (emoji, Streamlit colour keyword)
_RISK_COLOUR: dict[str, tuple[str, str]] = {
    "CRITICAL": ("🚨", "red"),
    "HIGH":     ("🔴", "red"),
    "MEDIUM":   ("🟡", "orange"),
    "LOW":      ("🟢", "green"),
}


def render_results(state: dict[str, Any]) -> None:
    """
    Render all pipeline results from *state*.

    Args:
        state: The NarrativeState dict returned by graph.invoke().
    """
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    errors = state.get("errors") or []
    if errors:
        _render_errors(errors)
        # Still try to show whatever partial results are available
        if not state.get("threat_assessment"):
            return

    threat  = state.get("threat_assessment", {})
    briefing = state.get("briefing", {})

    _render_threat_banner(threat, state.get("briefing_type", ""))
    _render_key_metrics(state, threat, briefing)
    _render_top_narratives(briefing)
    _render_coordination_alerts(briefing)
    _render_executive_briefing(briefing)
    _render_recommended_actions(briefing)
    _render_report_path(state)


# ---------------------------------------------------------------------------
# Sub-renderers (each receives only the slice of data it needs)
# ---------------------------------------------------------------------------


def _render_threat_banner(threat: dict, briefing_type: str) -> None:
    """Full-width threat level banner."""
    level = threat.get("overall_risk_level", "UNKNOWN")
    score = threat.get("overall_risk_score", 0.0)
    source = threat.get("source", "")
    emoji, colour = _RISK_COLOUR.get(level, ("⚪", "gray"))

    badge = "🚨 ALERT BRIEFING" if briefing_type == "alert" else "📋 STANDARD BRIEFING"

    if level == "CRITICAL":
        st.error(f"{emoji} **Threat Level: {level}** — Score: {score:.2f}  |  {badge}")
    elif level == "HIGH":
        st.error(f"{emoji} **Threat Level: {level}** — Score: {score:.2f}  |  {badge}")
    elif level == "MEDIUM":
        st.warning(f"{emoji} **Threat Level: {level}** — Score: {score:.2f}  |  {badge}")
    else:
        st.success(f"{emoji} **Threat Level: {level}** — Score: {score:.2f}  |  {badge}")

    if source:
        st.caption(f"Threat level source: `{source}`")


def _render_key_metrics(
    state: dict, threat: dict, briefing: dict
) -> None:
    """Four key metrics in a single row."""
    col1, col2, col3, col4 = st.columns(4)

    n_loaded   = len(state.get("narratives_loaded") or [])
    top_count  = len(briefing.get("top_narratives") or [])
    alert_count = len(briefing.get("coordination_alerts") or [])
    emerging   = len(briefing.get("emerging_narratives") or [])

    with col1:
        st.metric("Narratives Loaded",     n_loaded)
    with col2:
        st.metric("Top-Risk Narratives",   top_count)
    with col3:
        st.metric("Coordination Alerts",   alert_count)
    with col4:
        st.metric("Emerging Narratives",   emerging)


def _render_top_narratives(briefing: dict) -> None:
    """Top narratives ranked by risk."""
    top = briefing.get("top_narratives") or []
    if not top:
        return

    st.markdown("### 🏆 Top-Risk Narratives")
    for i, nid in enumerate(top, 1):
        st.markdown(f"{i}. `{nid}`")


def _render_coordination_alerts(briefing: dict) -> None:
    """Coordination alert list."""
    alerts = briefing.get("coordination_alerts") or []
    if not alerts:
        return

    st.markdown("### ⚠️ Coordination Alerts")
    st.caption("Narratives flagged for suspected inorganic or coordinated amplification.")
    for nid in alerts:
        st.markdown(f"- `{nid}`")


def _render_executive_briefing(briefing: dict) -> None:
    """Executive summary block."""
    summary = briefing.get("executive_summary", "").strip()
    if not summary:
        return

    st.markdown("### 📝 Executive Summary")
    st.info(summary)

    # Show raw briefing JSON in a collapsed expander for transparency
    with st.expander("🔍 Full Briefing JSON", expanded=False):
        st.json(briefing)


def _render_recommended_actions(briefing: dict) -> None:
    """Recommended actions list."""
    actions = briefing.get("recommended_actions") or []
    priorities = briefing.get("monitoring_priorities") or []
    if not actions and not priorities:
        return

    col1, col2 = st.columns(2)
    with col1:
        if actions:
            st.markdown("### ✅ Recommended Actions")
            for i, action in enumerate(actions, 1):
                st.markdown(f"{i}. {action}")

    with col2:
        if priorities:
            st.markdown("### 👁️ Monitoring Priorities")
            for nid in priorities:
                st.markdown(f"- `{nid}`")


def _render_report_path(state: dict) -> None:
    """Show where the briefing was saved."""
    path = state.get("report_path") or (
        state.get("analysis_results", {}).get("saved_path")
    )
    if not path:
        return

    st.markdown("### 💾 Saved Report")
    st.markdown(f"**File:** `{Path(path).name}`")
    st.markdown(f"**Path:** `{path}`")

    # Offer a raw view of the saved file
    try:
        import json
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        with st.expander("📄 View saved report file", expanded=False):
            st.json(content)
    except Exception:
        pass


def _render_errors(errors: list[str]) -> None:
    """Error log shown at the top of results if any errors occurred."""
    st.markdown("### ❌ Pipeline Errors")
    for err in errors:
        st.error(err)
    st.markdown(
        "The pipeline encountered errors above. "
        "This is expected if `OPENAI_API_KEY` is not configured — "
        "the CrewAI agents require a valid key to run LLM reasoning."
    )
