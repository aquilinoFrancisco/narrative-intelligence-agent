"""
state.py
LangGraph pipeline state for the Political Narrative Intelligence Agent.

NarrativeState is the single shared context object that flows through
every node in the graph.  Each node reads what it needs and writes only
the keys it is responsible for.

Design decisions
----------------
* TypedDict (not Pydantic) — LangGraph works natively with TypedDict; the
  Pydantic models in models/schemas.py are for MCP-layer validation, not
  for graph-level state.
* ``errors`` uses ``Annotated[list[str], operator.add]`` so each node can
  append errors without overwriting errors raised by a prior node.  All
  other fields use default (last-write-wins) semantics.
* ``status`` is a plain string sentinel used by the Streamlit UI and by
  graph-level logging.  Valid values are defined in ``NodeStatus``.
* ``briefing`` holds the raw dict representation of the saved
  IntelligenceBriefing so the Streamlit UI can render it without reading
  the filesystem.
* ``threat_assessment`` is a plain dict extracted from the saved briefing
  file.  It is intentionally kept separate from ``briefing`` so the
  conditional routing function can inspect it without parsing the full
  briefing payload.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Status sentinels
# Keep these as plain strings (not an Enum) so they serialise trivially to
# JSON when the Streamlit UI persists graph state.
# ---------------------------------------------------------------------------

class NodeStatus:
    """Recognised values for NarrativeState['status']."""
    PENDING          = "pending"
    COLLECTING       = "collecting"
    ANALYSING        = "analysing"
    EVALUATING       = "evaluating"
    BRIEFING_ALERT   = "briefing_alert"
    BRIEFING_STANDARD = "briefing_standard"
    SAVING           = "saving"
    COMPLETE         = "complete"
    ERROR            = "error"


# ---------------------------------------------------------------------------
# NarrativeState
# ---------------------------------------------------------------------------

class NarrativeState(TypedDict):
    """
    Full pipeline context shared across all LangGraph nodes.

    Fields are grouped by the node that primarily writes each one:

    **collect_data**
    * ``analysis_period``   — human-readable label for this run (set at entry)
    * ``narratives_loaded`` — raw list of validated narrative dicts from MCP

    **analyze_narratives**
    * ``analysis_results``  — full return dict from PoliticalCrew.run()

    **evaluate_threat_level**
    * ``threat_assessment`` — extracted dict: overall_risk_level, score, source

    **generate_alert_briefing / generate_standard_briefing**
    * ``briefing``          — IntelligenceBriefing dict loaded from saved file
    * ``briefing_type``     — "alert" or "standard"

    **save_report**
    * ``report_path``       — absolute path of the saved briefing JSON file

    **All nodes**
    * ``errors``            — accumulating list of error messages (never overwritten)
    * ``status``            — current pipeline stage (last-write-wins)
    """

    # --- Entry point ---------------------------------------------------------
    analysis_period: str
    """Human-readable label for this analysis run, e.g. '2025-06-27 morning'."""

    # --- collect_data writes -------------------------------------------------
    narratives_loaded: list[dict]
    """
    Validated NarrativeRecord dicts loaded from mock_narratives.json.
    Pre-populated by collect_data so downstream nodes have counts / IDs
    available even if PoliticalCrew is still running.
    """

    # --- analyze_narratives writes ------------------------------------------
    analysis_results: dict
    """
    The full return dict from PoliticalCrew.run(), e.g.:
      {
        "success": bool,
        "briefing_id": str,
        "raw_output": str,
        "briefing_saved": bool,
        "saved_path": str | None,
        "error": str | None,
      }
    """

    # --- evaluate_threat_level writes ---------------------------------------
    threat_assessment: dict
    """
    Extracted threat summary used for conditional routing:
      {
        "overall_risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
        "overall_risk_score": float,
        "source": "briefing_file" | "raw_output_heuristic" | "fallback",
      }
    """

    # --- briefing branch nodes write ----------------------------------------
    briefing: dict
    """
    The IntelligenceBriefing dict loaded from the saved JSON file.
    Populated after the alert or standard briefing node runs.
    """

    briefing_type: str
    """Confirmed briefing type: 'alert' or 'standard'."""

    # --- save_report writes -------------------------------------------------
    report_path: Optional[str]
    """
    Absolute filesystem path of the final saved briefing file.
    Set by save_report; may duplicate analysis_results['saved_path'] but
    is kept as a top-level field for easy access by the Streamlit UI.
    """

    # --- Shared across all nodes --------------------------------------------
    errors: Annotated[list[str], operator.add]
    """
    Accumulating error log.  Each node appends its own errors via:
        return {"errors": ["<message>"]}
    LangGraph merges all error lists with operator.add so no node can
    silently overwrite an earlier node's errors.
    """

    status: str
    """
    Current pipeline stage.  Use NodeStatus constants.
    Last-write-wins — each node stamps its own status on entry.
    """
