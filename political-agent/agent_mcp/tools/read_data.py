"""
read_data.py
MCP Tool: read_data

Loads mock_narratives.json from the project data directory, validates every
record against the NarrativeRecord Pydantic model, and returns the dataset
as a JSON-safe dictionary.

This is the entry-point tool for the DataCollector CrewAI agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Project root is three levels up from this file:
# political-agent/agent_mcp/tools/read_data.py → political-agent/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "mock_narratives.json"


def _import_schemas() -> Any:
    """Lazy import of NarrativeRecord to avoid circular dependency issues at load time."""
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from models.schemas import NarrativeRecord  # noqa: PLC0415
    return NarrativeRecord


def read_data(filepath: str | None = None) -> dict[str, Any]:
    """
    Load and validate the narrative dataset from disk.

    Reads the JSON file at *filepath* (defaults to ``data/mock_narratives.json``
    relative to the project root), parses each entry against the
    ``NarrativeRecord`` Pydantic schema, and returns a summary dict that is
    safe to serialise to JSON.

    Args:
        filepath: Optional override path to a narratives JSON file.  When
            ``None`` the default ``data/mock_narratives.json`` is used.

    Returns:
        A dict with the following keys:

        * ``success`` (bool): ``True`` if the file was read without errors.
        * ``source_file`` (str): Resolved path that was read.
        * ``total`` (int): Total number of raw records found in the file.
        * ``valid`` (int): Number of records that passed schema validation.
        * ``invalid`` (int): Number of records that failed schema validation.
        * ``records`` (list[dict]): Validated records serialised as dicts.
        * ``validation_errors`` (list[dict]): Details of any schema failures.
        * ``error`` (str | None): Top-level I/O error message if the file
            could not be read, otherwise ``None``.

    Raises:
        Nothing — all errors are captured and returned inside the dict so the
        caller can decide how to handle partial failures.
    """
    NarrativeRecord = _import_schemas()

    target = Path(filepath).resolve() if filepath else _DEFAULT_DATA_PATH

    # --- Read raw JSON -------------------------------------------------------
    try:
        raw: list[dict[str, Any]] = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "success": False,
            "source_file": str(target),
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "records": [],
            "validation_errors": [],
            "error": f"File not found: {target}",
        }
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "source_file": str(target),
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "records": [],
            "validation_errors": [],
            "error": f"JSON parse error: {exc}",
        }

    # --- Validate each record ------------------------------------------------
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, entry in enumerate(raw):
        # Skip schema-comment placeholders inserted during Phase 1
        if "_comment" in entry:
            continue
        try:
            record = NarrativeRecord.model_validate(entry)
            validated.append(record.model_dump(mode="json"))
        except ValidationError as exc:
            errors.append(
                {
                    "index": idx,
                    "raw_id": entry.get("id", f"<row {idx}>"),
                    "errors": exc.errors(),
                }
            )

    return {
        "success": True,
        "source_file": str(target),
        "total": len(raw),
        "valid": len(validated),
        "invalid": len(errors),
        "records": validated,
        "validation_errors": errors,
        "error": None,
    }
