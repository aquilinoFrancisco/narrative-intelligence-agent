"""
intelligence_reporter.py
CrewAI Agent: IntelligenceReporter

Responsibility
--------------
The IntelligenceReporter is the terminal agent in the crew pipeline.
It receives the DataCollector's structured dataset and the NarrativeAnalyst's
risk assessment, synthesises them into a structured executive briefing, and
persists the briefing to disk via the save_briefing MCP tool.

The briefing is civic and analytical in tone — it identifies risk signals and
recommends monitoring actions.  It does not generate persuasion strategies,
targeting recommendations, or manipulation guidance.

Design decisions
----------------
* The agent's task receives both the DataCollector task and the
  NarrativeAnalyst task as context (via Task.context), so the LLM has
  access to the raw records AND the analytical summary when drafting
  the briefing.
* save_briefing validates the output against the IntelligenceBriefing
  Pydantic schema before writing.  If the LLM produces a malformed dict,
  the tool returns a descriptive error rather than writing bad data.
* The briefing_id is derived from the analysis_period string passed to
  PoliticalCrew, making each run's output uniquely identifiable.
* expected_output for the Task is a strict JSON schema description so the
  LLM is guided toward structured output the downstream parser can consume.
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
# MCP Tool wrapper
# ---------------------------------------------------------------------------


@tool("Save Intelligence Briefing")
def save_intelligence_briefing_tool(briefing_json: str) -> str:
    """
    Validate and persist a completed IntelligenceBriefing to disk.

    Input: a JSON string conforming to the IntelligenceBriefing schema.
    Required fields:
      briefing_id, briefing_type (standard|alert), executive_summary,
      overall_risk_level (LOW|MEDIUM|HIGH|CRITICAL), overall_risk_score (0.0–1.0),
      top_narratives (list), emerging_narratives (list),
      coordination_alerts (list), recommended_actions (list),
      monitoring_priorities (list).

    Returns a JSON string with saved_path and status.
    """
    try:
        briefing = json.loads(briefing_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"success": False, "error": f"Invalid JSON input: {exc}"})

    result = dispatch("save_briefing", briefing=briefing, validate=True)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

def build_intelligence_reporter_agent() -> Agent:
    """
    Construct and return the IntelligenceReporter CrewAI Agent.
    """
    return Agent(
        role="Executive Intelligence Reporter",

        goal=(
            "Synthesise the collected narrative data and analytical findings into a "
            "concise, structured executive intelligence briefing. "
            "The briefing must clearly state the overall risk level, identify the top "
            "narratives by threat, surface any detected coordination patterns, and "
            "provide actionable monitoring recommendations for the analysis team. "
            "Save the completed briefing to disk using the available tool."
        ),

        backstory=(
            "You are a senior intelligence reporter at a civic monitoring organisation. "
            "You have distilled complex, multi-source intelligence into executive briefings "
            "for editorial boards, civil society organisations, and research institutions. "
            "Your writing is precise, neutral, and evidence-driven. "
            "You never sensationalise findings, never attribute intent without evidence, "
            "and never recommend actions beyond further monitoring and analysis. "
            "You understand that your briefings will be read by non-technical decision-makers "
            "and must therefore translate analytical scores into plain, actionable language "
            "without losing accuracy. "
            "You always produce structured output that downstream systems can parse."
        ),

        tools=[save_intelligence_briefing_tool],

        verbose=True,
        allow_delegation=False,
    )
