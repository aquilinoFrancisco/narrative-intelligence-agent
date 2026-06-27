"""
main_app.py
Streamlit entry point — Political Narrative Intelligence Agent.

Responsibilities of this file
------------------------------
* Configure the Streamlit page.
* Manage st.session_state for persistent pipeline results.
* Render the sidebar and main layout.
* Accept user input (analysis period selection).
* Call build_graph() and graph.invoke() inside st.spinner().
* Delegate all display logic to component modules.

What this file must NOT do
---------------------------
* Contain narrative analysis logic.
* Call MCP tools directly.
* Call CrewAI agents directly.
* Contain LLM prompt text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure political-agent/ is on the Python path so all workspace imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Political Narrative Intelligence Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Component imports (after sys.path is set)
# ---------------------------------------------------------------------------
from app.components.sidebar  import render_sidebar
from app.components.uploader import render_data_source
from app.components.results  import render_results

# ---------------------------------------------------------------------------
# Session state initialisation
# All keys are set here once so components can read them without KeyError.
# ---------------------------------------------------------------------------
_STATE_DEFAULTS: dict = {
    "pipeline_result": None,   # NarrativeState dict from last graph.invoke()
    "is_running":      False,  # guard flag while the pipeline is active
    "last_period":     None,   # analysis_period string used for the last run
}
for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Sidebar (always rendered, even before first run)
# ---------------------------------------------------------------------------
render_sidebar(st.session_state.pipeline_result)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔍 Political Narrative Intelligence Agent")
st.caption(
    "**Educational portfolio project using simulated data only.**  \n"
    "Demonstrates LangGraph orchestration, CrewAI multi-agent reasoning, "
    "and MCP tool protocol using fictional narrative data."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Data source panel
# ---------------------------------------------------------------------------
render_data_source()
st.markdown("")

# ---------------------------------------------------------------------------
# Analysis period selector
# ---------------------------------------------------------------------------
ANALYSIS_PERIODS: list[str] = [
    "2025-06-27 — Morning Intelligence Cycle",
    "2025-06-27 — Afternoon Intelligence Cycle",
    "2025-06-27 — Evening Intelligence Cycle",
    "2025-06-26 — Morning Intelligence Cycle",
    "2025-06-26 — Evening Intelligence Cycle",
]

col_select, col_run = st.columns([3, 1])

with col_select:
    selected_period: str = st.selectbox(
        "Select Analysis Period",
        options=ANALYSIS_PERIODS,
        index=0,
        help="Each period triggers a fresh pipeline run using the same mock dataset.",
    )

with col_run:
    st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical alignment spacer
    run_pressed: bool = st.button(
        "▶ Run Analysis",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_pressed and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.pipeline_result = None  # clear previous result

    with st.spinner(
        "🤖 Pipeline running — LangGraph is orchestrating CrewAI agents…  \n"
        "*(This may take 30–120 seconds depending on LLM response time.)*"
    ):
        try:
            # Import here (not at module level) to avoid loading heavy CrewAI
            # dependencies on every Streamlit rerun before the user presses Run.
            from graph.workflow import build_graph
            from graph.state    import NodeStatus

            initial_state = {
                "analysis_period":   selected_period,
                "narratives_loaded": [],
                "analysis_results":  {},
                "threat_assessment": {},
                "briefing":          {},
                "briefing_type":     "",
                "report_path":       None,
                "errors":            [],
                "status":            NodeStatus.PENDING,
            }

            graph  = build_graph()
            result = graph.invoke(initial_state)

            st.session_state.pipeline_result = result
            st.session_state.last_period     = selected_period

            if result.get("status") == NodeStatus.COMPLETE:
                st.success("✅ Pipeline completed successfully!")
            elif result.get("status") == NodeStatus.ERROR:
                st.error("❌ Pipeline finished with errors — see Results below.")

        except Exception as exc:
            st.error(f"❌ Unexpected pipeline error: {exc}")
            # Persist a minimal error state so results panel can render
            st.session_state.pipeline_result = {
                "analysis_period":   selected_period,
                "narratives_loaded": [],
                "analysis_results":  {},
                "threat_assessment": {},
                "briefing":          {},
                "briefing_type":     "",
                "report_path":       None,
                "errors":            [f"Unhandled exception: {exc}"],
                "status":            "error",
            }
        finally:
            st.session_state.is_running = False

    # Force a rerun so the sidebar progress bar and results panel refresh
    st.rerun()

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
if st.session_state.pipeline_result is not None:
    render_results(st.session_state.pipeline_result)
else:
    # Pre-run placeholder
    st.markdown("### 📋 Results")
    st.info(
        "Select an analysis period and click **▶ Run Analysis** to start the pipeline.  \n\n"
        "The pipeline will:\n"
        "1. Load 10 fictional narrative records via the MCP layer\n"
        "2. Run 3 CrewAI agents (DataCollector → NarrativeAnalyst → IntelligenceReporter)\n"
        "3. Evaluate threat level and route to the appropriate briefing branch\n"
        "4. Save an intelligence briefing to the `briefings/` directory\n\n"
        "⚠️ **Note:** Full agent execution requires `OPENAI_API_KEY` to be set.  \n"
        "Without it the pipeline will complete with an error — all other layers "
        "(LangGraph, MCP, data validation) still run and are visible in the logs."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Political Narrative Intelligence Agent · Portfolio project · "
    "Python 3.11 · LangGraph · CrewAI · MCP · Streamlit  \n"
    "All data is fictional. No real political actors, events, or organisations are referenced."
)
