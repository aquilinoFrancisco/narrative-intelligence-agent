"""
sidebar.py
Streamlit sidebar component — pipeline status and architecture overview.

Renders:
  - Current pipeline step with live status badge
  - Animated progress bar mapped from NodeStatus
  - Architecture diagram (text-based)
  - About section with links

No business logic lives here.  All data comes from the NarrativeState dict
passed in from main_app.py.
"""

from __future__ import annotations

import streamlit as st

# Status → (label, progress 0.0–1.0, colour)
_STATUS_MAP: dict[str, tuple[str, float, str]] = {
    "pending":           ("⏳ Pending",              0.00, "gray"),
    "collecting":        ("📥 Loading Data",         0.15, "blue"),
    "analysing":         ("🤖 Agents Running",       0.50, "orange"),
    "evaluating":        ("🔍 Evaluating Threat",    0.70, "orange"),
    "briefing_alert":    ("🚨 Alert Briefing",       0.85, "red"),
    "briefing_standard": ("📋 Standard Briefing",    0.85, "green"),
    "saving":            ("💾 Saving Report",        0.95, "blue"),
    "complete":          ("✅ Complete",             1.00, "green"),
    "error":             ("❌ Error",               1.00, "red"),
}


def render_sidebar(pipeline_result: dict | None) -> None:
    """
    Render the full sidebar.

    Args:
        pipeline_result: The NarrativeState dict from the last graph.invoke()
            call, or None if no run has completed yet.
    """
    with st.sidebar:
        st.markdown("## 🔭 Pipeline Status")

        status = "pending"
        if pipeline_result:
            status = pipeline_result.get("status", "pending")

        label, progress, _ = _STATUS_MAP.get(status, ("Unknown", 0.0, "gray"))
        st.markdown(f"**Current step:** {label}")
        st.progress(progress)

        # --- Last run stats -------------------------------------------------
        if pipeline_result:
            st.markdown("---")
            st.markdown("### 📊 Last Run")

            period = pipeline_result.get("analysis_period", "—")
            st.markdown(f"**Period:** {period}")

            n_loaded = len(pipeline_result.get("narratives_loaded") or [])
            st.markdown(f"**Records loaded:** {n_loaded}")

            report_path = pipeline_result.get("report_path")
            if report_path:
                # Show just the filename, not the full absolute path
                from pathlib import Path
                st.markdown(f"**Report saved:** `{Path(report_path).name}`")

            errors = pipeline_result.get("errors") or []
            if errors:
                st.error(f"⚠️ {len(errors)} error(s) recorded")

        # --- Architecture diagram --------------------------------------------
        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        st.code(
            "Streamlit UI\n"
            "    ↓\n"
            "LangGraph (orchestrator)\n"
            "    ↓\n"
            "CrewAI (3 agents)\n"
            "  • DataCollector\n"
            "  • NarrativeAnalyst\n"
            "  • IntelligenceReporter\n"
            "    ↓\n"
            "MCP Tools (5)\n"
            "    ↓\n"
            "Mock JSON Dataset",
            language=None,
        )

        # --- Node legend ----------------------------------------------------
        st.markdown("### 🔀 Workflow Nodes")
        st.markdown(
            "1. `collect_data`\n"
            "2. `analyze_narratives`\n"
            "3. `evaluate_threat_level`\n"
            "4a. `generate_alert_briefing` *(CRITICAL)*\n"
            "4b. `generate_standard_briefing` *(other)*\n"
            "5. `save_report`\n"
            "6. `handle_error` *(on failure)*"
        )

        # --- About ----------------------------------------------------------
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            "Political Narrative Intelligence Agent  \n"
            "Portfolio project — civic intelligence demo.  \n"
            "Python 3.11 · LangGraph · CrewAI · MCP"
        )
