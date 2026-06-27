"""
search_narrative.py
MCP Tool: search_narrative

Filters narrative records from mock_narratives.json by keyword query
against a specified field.  Operates entirely in-memory and returns only
the matching records as a JSON-safe dictionary.

This tool is used by the DataCollector CrewAI agent to focus the dataset
before passing it downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# read_data is imported at call-time inside the function to keep tools decoupled.
_SEARCHABLE_FIELDS = {
    "topic",
    "sentiment",
    "sample_posts",
    "peak_hour",
}


def search_narrative(
    query: str,
    field: str = "topic",
    case_sensitive: bool = False,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Search narrative records by keyword within a specified field.

    The search is a substring match.  When *records* is ``None`` the tool
    calls :func:`read_data` internally to load the full dataset — this allows
    the tool to be used standalone (e.g. in tests) without a preceding
    ``read_data`` call.

    Args:
        query: The substring to search for.  Cannot be empty.
        field: The record field to search within.  Must be one of:
            ``"topic"``, ``"sentiment"``, ``"sample_posts"``, ``"peak_hour"``.
            Defaults to ``"topic"``.
        case_sensitive: When ``False`` (default) the comparison is
            case-insensitive.
        records: Optional pre-loaded list of validated narrative dicts (e.g.
            the ``"records"`` value from a prior ``read_data`` call).  When
            provided, skips the internal ``read_data`` call.

    Returns:
        A dict with the following keys:

        * ``success`` (bool): ``True`` if the search completed without error.
        * ``query`` (str): The query string that was searched.
        * ``field`` (str): The field that was searched.
        * ``total_searched`` (int): Total records examined.
        * ``match_count`` (int): Number of matching records.
        * ``matches`` (list[dict]): Matching narrative records serialised as dicts.
        * ``error`` (str | None): Error message if search could not run.
    """
    # --- Input validation ----------------------------------------------------
    if not query or not query.strip():
        return {
            "success": False,
            "query": query,
            "field": field,
            "total_searched": 0,
            "match_count": 0,
            "matches": [],
            "error": "query must be a non-empty string.",
        }

    if field not in _SEARCHABLE_FIELDS:
        return {
            "success": False,
            "query": query,
            "field": field,
            "total_searched": 0,
            "match_count": 0,
            "matches": [],
            "error": (
                f"field '{field}' is not searchable. "
                f"Allowed fields: {sorted(_SEARCHABLE_FIELDS)}"
            ),
        }

    # --- Load records if not supplied ----------------------------------------
    if records is None:
        from agent_mcp.tools.read_data import read_data  # local import keeps tools decoupled
        load_result = read_data()
        if not load_result["success"]:
            return {
                "success": False,
                "query": query,
                "field": field,
                "total_searched": 0,
                "match_count": 0,
                "matches": [],
                "error": f"Failed to load data: {load_result['error']}",
            }
        records = load_result["records"]

    # --- Substring search ----------------------------------------------------
    needle = query if case_sensitive else query.lower()
    matches: list[dict[str, Any]] = []

    for record in records:
        value = record.get(field)

        if value is None:
            continue

        if isinstance(value, list):
            # For list fields (e.g. sample_posts), search each element
            haystack = " ".join(str(v) for v in value)
        else:
            haystack = str(value)

        if not case_sensitive:
            haystack = haystack.lower()

        if needle in haystack:
            matches.append(record)

    return {
        "success": True,
        "query": query,
        "field": field,
        "total_searched": len(records),
        "match_count": len(matches),
        "matches": matches,
        "error": None,
    }
