"""
narrative_analyst.py
CrewAI Agent: NarrativeAnalyst

Responsibility
--------------
The NarrativeAnalyst receives the structured dataset from the DataCollector
and applies two analytical lenses:

1. Per-narrative virality scoring — a weighted composite of spread velocity,
   coordination signals, and bot probability, yielding a risk tier.

2. Cross-narrative pattern detection — identifies structural coordination
   signals (shared peak hours, high-coordination clusters, bot surge) that
   may indicate inorganic or orchestrated amplification.

The agent's output is a structured analytical summary that the
IntelligenceReporter uses to produce the final executive briefing.

Design decisions
----------------
* Both tools call agent_mcp.server.dispatch() — same MCP boundary as all
  other agents.  No tool file is imported directly here.
* calculate_virality_score is called once per narrative record; the agent
  selects the top-risk narratives to include in its summary rather than
  returning scores for every record (keeps LLM context manageable).
* detect_patterns operates on the full record list so it can identify
  cross-narrative coordination that per-record scoring cannot see.
* Prompts deliberately avoid predictive targeting language.  The goal is
  analytical understanding, not audience manipulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from crewai import Agent
from crewai.tools import tool

from agent_mcp.server import dispatch


# ---------------------------------------------------------------------------
# MCP Tool wrappers
# ---------------------------------------------------------------------------


@tool("Calculate Narrative Virality Score")
def calculate_virality_score_tool(narrative_json: str) -> str:
    """
    Compute a composite virality and risk score for a single narrative record.

    Input: a JSON string representing one narrative record dict.
    The record must contain 'id', 'virality_score', 'coordination_score',
    and 'bot_probability' fields.

    Returns a JSON string with:
    - composite_score (float 0.0–1.0)
    - risk_tier (LOW / MEDIUM / HIGH / CRITICAL)
    - coordination_alert (bool)
    - interpretation (plain-English summary)
    """
    try:
        narrative = json.loads(narrative_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"success": False, "error": f"Invalid JSON input: {exc}"})

    result = dispatch("calculate_virality_score", narrative=narrative)
    return json.dumps(result, default=str)


@tool("Detect Coordination Patterns")
def detect_patterns_tool(narratives_json: str) -> str:
    """
    Scan a list of narrative records for suspicious coordination patterns.

    Input: a JSON string representing a list of narrative record dicts.
    Detects: peak-hour synchrony, high-coordination clusters, bot surge.

    Returns a JSON string with a list of detected patterns, each containing:
    - pattern_type
    - confidence (float 0.0–1.0)
    - affected_narratives (list of IDs)
    - description
    """
    try:
        narratives = json.loads(narratives_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"success": False, "error": f"Invalid JSON input: {exc}"})

    if not isinstance(narratives, list):
        return json.dumps(
            {"success": False, "error": "Input must be a JSON array of narrative records."}
        )

    result = dispatch("detect_patterns", narratives=narratives)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

def build_narrative_analyst_agent() -> Agent:
    """
    Construct and return the NarrativeAnalyst CrewAI Agent.
    """
    return Agent(
        role="Narrative Intelligence Analyst",

        goal=(
            "Analyse narrative records to identify which themes are growing fastest, "
            "which exhibit signs of coordinated or inauthentic amplification, and what "
            "the aggregate risk level of the current information environment is. "
            "Produce a clear, evidence-based analytical summary ready for executive reporting."
        ),

        backstory=(
            "You are a narrative intelligence analyst with expertise in information "
            "environment monitoring for civic and journalistic organisations. "
            "You have analysed hundreds of information campaigns — from organic grassroots "
            "movements to coordinated inauthentic behaviour operations. "
            "You approach every dataset with scepticism and rigour: you score what you can "
            "measure, flag what you cannot, and never overstate certainty. "
            "Your analysis is neutral, factual, and grounded in observable metrics. "
            "You do not speculate about intent beyond what the data supports, and you "
            "explicitly distinguish between high-confidence findings and working hypotheses."
        ),

        tools=[calculate_virality_score_tool, detect_patterns_tool],

        verbose=True,
        allow_delegation=False,
    )
