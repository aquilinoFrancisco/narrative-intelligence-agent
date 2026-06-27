"""
server.py
MCP Server — tool registry for the Political Narrative Intelligence Agent.

This is the single file that imports all MCP tools.  It exposes:

* ``TOOL_REGISTRY``  — a dict mapping tool names to their callable functions.
* ``dispatch()``     — a convenience function for invoking a tool by name.
* ``list_tools()``   — returns metadata about every registered tool.

CrewAI agents should import *only* from this module, not from individual
tool files, to maintain a clean dependency boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Tool imports — only server.py imports from agent_mcp.tools.*
# ---------------------------------------------------------------------------
from agent_mcp.tools.read_data import read_data
from agent_mcp.tools.search_narrative import search_narrative
from agent_mcp.tools.calculate_virality_score import calculate_virality_score
from agent_mcp.tools.detect_patterns import detect_patterns
from agent_mcp.tools.save_briefing import save_briefing

# ---------------------------------------------------------------------------
# Tool metadata catalogue
# ---------------------------------------------------------------------------

_TOOL_METADATA: list[dict[str, str]] = [
    {
        "name": "read_data",
        "description": (
            "Load and validate mock_narratives.json. "
            "Returns all NarrativeRecord dicts plus load statistics."
        ),
        "primary_agent": "DataCollector",
        "returns": "dict with 'records' list and load statistics",
    },
    {
        "name": "search_narrative",
        "description": (
            "Substring-search narrative records by field (topic, sentiment, "
            "sample_posts, peak_hour). Returns matching records."
        ),
        "primary_agent": "DataCollector",
        "returns": "dict with 'matches' list",
    },
    {
        "name": "calculate_virality_score",
        "description": (
            "Compute a weighted composite amplification score for a single "
            "narrative record and assign a risk tier (LOW/MEDIUM/HIGH/CRITICAL)."
        ),
        "primary_agent": "NarrativeAnalyst",
        "returns": "dict with composite_score, risk_tier, and interpretation",
    },
    {
        "name": "detect_patterns",
        "description": (
            "Detect coordination patterns across a list of narrative records: "
            "peak-hour synchrony, high-coordination clusters, and bot surge."
        ),
        "primary_agent": "NarrativeAnalyst",
        "returns": "dict with patterns list and summary",
    },
    {
        "name": "save_briefing",
        "description": (
            "Persist an IntelligenceBriefing dict to the briefings/ directory "
            "as a timestamped JSON file."
        ),
        "primary_agent": "IntelligenceReporter",
        "returns": "dict with saved_path and status",
    },
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "read_data":               read_data,
    "search_narrative":        search_narrative,
    "calculate_virality_score": calculate_virality_score,
    "detect_patterns":         detect_patterns,
    "save_briefing":           save_briefing,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_tools() -> list[dict[str, str]]:
    """
    Return metadata for every registered MCP tool.

    Returns:
        A list of dicts, each containing:
        * ``name``           — the tool's registry key
        * ``description``    — what the tool does
        * ``primary_agent``  — which CrewAI agent primarily uses it
        * ``returns``        — description of the return value shape
    """
    return [meta.copy() for meta in _TOOL_METADATA]


def dispatch(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Invoke a registered MCP tool by name and return its result.

    This is the primary entry-point for CrewAI agents.  All arguments
    beyond ``tool_name`` are forwarded as keyword arguments to the tool
    function.

    Args:
        tool_name: The name of the tool to call.  Must be a key in
            ``TOOL_REGISTRY``.
        **kwargs: Keyword arguments forwarded verbatim to the tool function.

    Returns:
        The JSON-safe dict returned by the tool, or an error dict if:
        * ``tool_name`` is not registered
        * The tool raises an unexpected exception

    The error dict always has the shape::

        {
            "success": False,
            "tool_name": "<name>",
            "error": "<message>"
        }
    """
    if tool_name not in TOOL_REGISTRY:
        available = sorted(TOOL_REGISTRY.keys())
        return {
            "success": False,
            "tool_name": tool_name,
            "error": (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {available}"
            ),
        }

    try:
        result: dict[str, Any] = TOOL_REGISTRY[tool_name](**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"Tool raised an unexpected exception: {type(exc).__name__}: {exc}",
        }

    return result
