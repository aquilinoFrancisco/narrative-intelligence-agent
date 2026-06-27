"""
workflow.py
LangGraph StateGraph workflow for the Political Narrative Intelligence Agent.

Public API
----------
* ``build_graph()``          — compile and return the runnable graph
* ``run_pipeline(period)``   — convenience wrapper: build + run in one call
* ``GRAPH_ENTRY``            — name of the entry-point node
* ``GRAPH_END``              — LangGraph's END sentinel, exported for Streamlit

Architecture rules (enforced here)
-----------------------------------
1. Nodes are thin.  No data analysis logic lives inside a node function.
2. Only ``analyze_narratives`` calls PoliticalCrew.run().
3. All other nodes only route, update state fields, or load persisted data.
4. The graph has no direct OpenAI calls, no raw prompt text, no Streamlit imports.
5. Conditional routing is driven solely by state["threat_assessment"].

Node responsibilities
---------------------
collect_data            Verify data is loadable; populate narratives_loaded.
analyze_narratives      Call PoliticalCrew.run(); store result in analysis_results.
evaluate_threat_level   Extract threat level from saved briefing; update threat_assessment.
generate_alert_briefing  Confirm CRITICAL path; set briefing_type and load briefing dict.
generate_standard_briefing Confirm non-critical path; set briefing_type and load briefing dict.
save_report             Stamp report_path into top-level state; mark complete.
handle_error            Record final error status; ensure errors list is populated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langgraph.graph import END, StateGraph

from graph.state import NarrativeState, NodeStatus

# ---------------------------------------------------------------------------
# Node name constants — avoids magic strings in edges and tests
# ---------------------------------------------------------------------------

GRAPH_ENTRY            = "collect_data"
NODE_COLLECT           = "collect_data"
NODE_ANALYSE           = "analyze_narratives"
NODE_EVALUATE          = "evaluate_threat_level"
NODE_ALERT_BRIEFING    = "generate_alert_briefing"
NODE_STANDARD_BRIEFING = "generate_standard_briefing"
NODE_SAVE              = "save_report"
NODE_ERROR             = "handle_error"
GRAPH_END              = END   # re-exported so Streamlit never imports from langgraph directly


# ===========================================================================
# Node functions
# Each function receives the full NarrativeState and returns a *partial*
# state dict containing only the keys it modifies.
# ===========================================================================


def collect_data(state: NarrativeState) -> dict[str, Any]:
    """
    Node: collect_data
    ------------------
    Verify that the mock dataset is reachable and pre-populate
    ``narratives_loaded`` so downstream nodes can reference record IDs
    without re-loading the file.

    This node does NOT analyse or score narratives — it is a thin
    read-and-verify step.
    """
    from agent_mcp.server import dispatch  # local import keeps graph decoupled at module level

    result = dispatch("read_data")

    if not result["success"]:
        return {
            "status": NodeStatus.ERROR,
            "errors": [f"collect_data: failed to load dataset — {result['error']}"],
            "narratives_loaded": [],
        }

    return {
        "status": NodeStatus.COLLECTING,
        "narratives_loaded": result["records"],
    }


def analyze_narratives(state: NarrativeState) -> dict[str, Any]:
    """
    Node: analyze_narratives
    ------------------------
    The only node that invokes PoliticalCrew.run().

    Passes ``analysis_period`` from state into the crew and stores the
    full return dict in ``analysis_results``.  No interpretation of the
    crew output happens here — that is evaluate_threat_level's job.
    """
    # Guard: abort early if a prior node already errored
    if state.get("errors"):
        return {"status": NodeStatus.ERROR}

    from crew.political_crew import PoliticalCrew  # local import keeps graph decoupled

    crew = PoliticalCrew(analysis_period=state["analysis_period"])
    crew_result = crew.run()

    if not crew_result["success"]:
        return {
            "status": NodeStatus.ERROR,
            "analysis_results": crew_result,
            "errors": [f"analyze_narratives: crew failed — {crew_result['error']}"],
        }

    return {
        "status": NodeStatus.ANALYSING,
        "analysis_results": crew_result,
    }


def evaluate_threat_level(state: NarrativeState) -> dict[str, Any]:
    """
    Node: evaluate_threat_level
    ---------------------------
    Extract the threat level from the saved briefing file produced by
    PoliticalCrew, then populate ``threat_assessment`` for the conditional
    routing edge.

    Resolution order (most reliable → least reliable):
    1. Read ``overall_risk_level`` from the saved briefing JSON file.
    2. Scan ``raw_output`` for the first occurrence of a known level keyword.
    3. Fall back to "LOW" if neither source yields a recognisable level.

    This is routing-only logic — no scoring or analysis is performed here.
    """
    if state.get("errors"):
        return {"status": NodeStatus.ERROR}

    analysis = state.get("analysis_results", {})
    known_levels = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    # --- Strategy 1: read the saved briefing file ----------------------------
    saved_path = analysis.get("saved_path")
    if saved_path:
        try:
            briefing_data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
            level = briefing_data.get("overall_risk_level", "").upper()
            score = float(briefing_data.get("overall_risk_score", 0.0))
            if level in known_levels:
                return {
                    "status": NodeStatus.EVALUATING,
                    "threat_assessment": {
                        "overall_risk_level": level,
                        "overall_risk_score": score,
                        "source": "briefing_file",
                    },
                }
        except Exception as exc:  # noqa: BLE001
            # Non-fatal — fall through to next strategy
            pass

    # --- Strategy 2: heuristic scan of raw LLM output -----------------------
    raw_output: str = analysis.get("raw_output", "")
    for level in known_levels:
        if level in raw_output.upper():
            return {
                "status": NodeStatus.EVALUATING,
                "threat_assessment": {
                    "overall_risk_level": level,
                    "overall_risk_score": 0.0,
                    "source": "raw_output_heuristic",
                },
            }

    # --- Strategy 3: fallback -------------------------------------------------
    return {
        "status": NodeStatus.EVALUATING,
        "threat_assessment": {
            "overall_risk_level": "LOW",
            "overall_risk_score": 0.0,
            "source": "fallback",
        },
    }


def generate_alert_briefing(state: NarrativeState) -> dict[str, Any]:
    """
    Node: generate_alert_briefing  (CRITICAL branch)
    -------------------------------------------------
    Thin node.  The briefing was already produced and saved by PoliticalCrew.
    This node confirms briefing_type="alert", loads the briefing dict from
    disk for the UI, and stamps status.
    """
    saved_path = state.get("analysis_results", {}).get("saved_path")
    briefing = _load_briefing_from_disk(saved_path)

    return {
        "status": NodeStatus.BRIEFING_ALERT,
        "briefing_type": "alert",
        "briefing": briefing,
    }


def generate_standard_briefing(state: NarrativeState) -> dict[str, Any]:
    """
    Node: generate_standard_briefing  (non-CRITICAL branch)
    --------------------------------------------------------
    Thin node.  Same pattern as generate_alert_briefing but for the
    standard (non-critical) path.
    """
    saved_path = state.get("analysis_results", {}).get("saved_path")
    briefing = _load_briefing_from_disk(saved_path)

    return {
        "status": NodeStatus.BRIEFING_STANDARD,
        "briefing_type": "standard",
        "briefing": briefing,
    }


def save_report(state: NarrativeState) -> dict[str, Any]:
    """
    Node: save_report
    -----------------
    Thin node.  The file was already persisted by the MCP save_briefing tool
    inside PoliticalCrew.  This node promotes ``saved_path`` to the top-level
    ``report_path`` field and marks the pipeline complete.
    """
    saved_path = (
        state.get("analysis_results", {}).get("saved_path")
        or state.get("report_path")
    )

    return {
        "status": NodeStatus.COMPLETE,
        "report_path": saved_path,
    }


def handle_error(state: NarrativeState) -> dict[str, Any]:
    """
    Node: handle_error
    ------------------
    Terminal error node.  Ensures the status field always reflects an
    error state and that at least one error message is present.
    If no errors were recorded by prior nodes, a generic sentinel is added.
    """
    existing_errors: list[str] = state.get("errors") or []

    new_errors: list[str] = []
    if not existing_errors:
        new_errors = ["handle_error: pipeline terminated with unknown error."]

    return {
        "status": NodeStatus.ERROR,
        "errors": new_errors,  # operator.add merges this with existing
    }


# ===========================================================================
# Conditional routing function
# ===========================================================================


def _route_after_evaluation(state: NarrativeState) -> str:
    """
    Conditional edge: evaluate_threat_level → branch node.

    Routing table:
      errors present          → handle_error
      CRITICAL risk level     → generate_alert_briefing
      HIGH / MEDIUM / LOW     → generate_standard_briefing
    """
    if state.get("errors"):
        return NODE_ERROR

    level = state.get("threat_assessment", {}).get("overall_risk_level", "LOW")

    if level == "CRITICAL":
        return NODE_ALERT_BRIEFING

    # HIGH, MEDIUM, LOW, and any unrecognised value → standard path
    return NODE_STANDARD_BRIEFING


# ===========================================================================
# Private helpers
# ===========================================================================


def _load_briefing_from_disk(saved_path: str | None) -> dict[str, Any]:
    """
    Load the briefing JSON from disk.  Returns an empty dict on any failure
    so briefing nodes never raise — the UI can detect an empty dict and
    display an appropriate message.
    """
    if not saved_path:
        return {}
    try:
        return json.loads(Path(saved_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ===========================================================================
# Graph builder
# ===========================================================================


def build_graph() -> Any:
    """
    Compile and return the LangGraph StateGraph for the narrative pipeline.

    Graph topology::

        collect_data
            ↓
        analyze_narratives
            ↓
        evaluate_threat_level
            ↓ (conditional)
        ┌───────────────────────────────┐
        │ CRITICAL → alert_briefing     │
        │ OTHER    → standard_briefing  │
        │ ERROR    → handle_error       │
        └───────────────────────────────┘
            ↓ (all non-error paths)
        save_report
            ↓
           END

        handle_error
            ↓
           END

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to invoke with
        ``graph.invoke(initial_state)``.
    """
    graph = StateGraph(NarrativeState)

    # --- Register nodes ------------------------------------------------------
    graph.add_node(NODE_COLLECT,           collect_data)
    graph.add_node(NODE_ANALYSE,           analyze_narratives)
    graph.add_node(NODE_EVALUATE,          evaluate_threat_level)
    graph.add_node(NODE_ALERT_BRIEFING,    generate_alert_briefing)
    graph.add_node(NODE_STANDARD_BRIEFING, generate_standard_briefing)
    graph.add_node(NODE_SAVE,              save_report)
    graph.add_node(NODE_ERROR,             handle_error)

    # --- Set entry point -----------------------------------------------------
    graph.set_entry_point(GRAPH_ENTRY)

    # --- Linear edges --------------------------------------------------------
    graph.add_edge(NODE_COLLECT,  NODE_ANALYSE)
    graph.add_edge(NODE_ANALYSE,  NODE_EVALUATE)

    # --- Conditional edge after threat evaluation ----------------------------
    graph.add_conditional_edges(
        NODE_EVALUATE,
        _route_after_evaluation,
        {
            NODE_ALERT_BRIEFING:    NODE_ALERT_BRIEFING,
            NODE_STANDARD_BRIEFING: NODE_STANDARD_BRIEFING,
            NODE_ERROR:             NODE_ERROR,
        },
    )

    # --- Convergence: both briefing branches → save_report -------------------
    graph.add_edge(NODE_ALERT_BRIEFING,    NODE_SAVE)
    graph.add_edge(NODE_STANDARD_BRIEFING, NODE_SAVE)

    # --- Terminal edges -------------------------------------------------------
    graph.add_edge(NODE_SAVE,  END)
    graph.add_edge(NODE_ERROR, END)

    return graph.compile()


# ===========================================================================
# Convenience runner
# ===========================================================================


def run_pipeline(analysis_period: str) -> NarrativeState:
    """
    Build the graph and run a full pipeline for *analysis_period*.

    This is the primary entry point for the Streamlit UI in Phase 6.
    It constructs the initial state, invokes the compiled graph, and
    returns the final NarrativeState.

    Args:
        analysis_period: Human-readable label for this run,
            e.g. ``"2025-06-27 morning cycle"``.

    Returns:
        The final NarrativeState after all nodes have executed.
    """
    initial_state: NarrativeState = {
        "analysis_period":  analysis_period,
        "narratives_loaded": [],
        "analysis_results":  {},
        "threat_assessment": {},
        "briefing":          {},
        "briefing_type":     "",
        "report_path":       None,
        "errors":            [],
        "status":            NodeStatus.PENDING,
    }

    compiled = build_graph()
    return compiled.invoke(initial_state)
