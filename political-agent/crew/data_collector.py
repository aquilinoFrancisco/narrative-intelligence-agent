"""
data_collector.py
CrewAI Agent: DataCollector

Responsibility
--------------
The DataCollector is the entry-point agent in the sequential crew pipeline.
It loads raw narrative records from the mock dataset, validates them, and
structures the payload that all downstream agents depend on.

Design decisions
----------------
* Tools wrap agent_mcp.server.dispatch() so no tool file is imported
  directly — the MCP server is the single source of truth for tool access.
* Tool functions return JSON strings (str) because CrewAI passes tool
  output to the LLM as text.  The LLM then summarises / references it
  in its task output, which is forwarded via Task.context.
* The agent is intentionally narrow in scope — it does not analyse or score
  narratives.  Separation of concerns keeps each agent's reasoning focused.
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
# Tool functions must accept only str args and return str — CrewAI contract.
# ---------------------------------------------------------------------------


@tool("Load Narrative Dataset")
def load_narrative_dataset(filepath: str = "") -> str:
    """
    Load and validate the full narrative dataset from mock_narratives.json.

    Pass an empty string or 'default' to use the built-in mock dataset.
    Returns a JSON string containing all validated narrative records and
    load statistics (total, valid, invalid counts).
    """
    # Treat empty/placeholder inputs as "use default path"
    path_arg = filepath.strip() if filepath and filepath.strip() not in ("", "default") else None
    result = dispatch("read_data", filepath=path_arg)
    return json.dumps(result, default=str)


@tool("Search Narratives By Field")
def search_narratives_by_field(query_and_field: str) -> str:
    """
    Search narrative records by keyword within a specified field.

    Input format: '<query>|<field>'
    Example: 'fraud|topic' or 'negative|sentiment'

    Supported fields: topic, sentiment, sample_posts, peak_hour.
    Returns a JSON string with matching narrative records.
    """
    # Parse the combined argument — CrewAI tools receive a single string input
    parts = query_and_field.split("|", 1)
    query = parts[0].strip() if parts else ""
    field = parts[1].strip() if len(parts) > 1 else "topic"

    result = dispatch("search_narrative", query=query, field=field)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

def build_data_collector_agent() -> Agent:
    """
    Construct and return the DataCollector CrewAI Agent.

    The agent is created fresh on each call so the crew can be
    re-instantiated for multiple analysis runs without stale state.
    """
    return Agent(
        role="Civic Data Collection Specialist",

        goal=(
            "Retrieve, validate, and structure all narrative records from the "
            "intelligence dataset so that downstream analysts receive a complete, "
            "well-formed corpus to work with."
        ),

        backstory=(
            "You are a senior data specialist at a civic intelligence unit. "
            "Your background is in open-source research and information curation. "
            "You have spent years building pipelines that surface structured signals "
            "from noisy, unverified data sources. You are methodical, precise, and "
            "deeply aware that downstream decisions depend entirely on the quality "
            "of the data you provide. You do not speculate beyond what the data "
            "contains, and you flag anomalies rather than silently discarding them."
        ),

        tools=[load_narrative_dataset, search_narratives_by_field],

        # verbose=True surfaces per-step reasoning in logs — useful for a
        # portfolio demo but can be toggled off for production.
        verbose=True,

        # allow_delegation=False keeps each agent in its lane.
        # The DataCollector should never hand off to another agent mid-task.
        allow_delegation=False,
    )
