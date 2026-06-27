"""
save_briefing.py
MCP Tool: save_briefing

Persists a final IntelligenceBriefing to the project's ``briefings/``
directory as a timestamped JSON file.

The tool optionally validates the incoming dict against the
``IntelligenceBriefing`` Pydantic model before writing — this is on by
default to ensure only well-formed briefings are saved.

This tool is used by the IntelligenceReporter CrewAI agent as the
terminal step of the pipeline.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_BRIEFINGS_DIR = _PROJECT_ROOT / "briefings"


def save_briefing(
    briefing: dict[str, Any],
    filename: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Serialise and persist an intelligence briefing to disk.

    The briefing is written to ``<project_root>/briefings/`` as a
    pretty-printed JSON file.  The directory is created automatically if it
    does not exist.

    Args:
        briefing: A dict representing the ``IntelligenceBriefing`` payload
            produced by the IntelligenceReporter agent.  Must contain at
            minimum a ``"briefing_id"`` field.
        filename: Optional explicit filename (without directory path).  When
            ``None`` the filename is auto-generated as
            ``briefing_<YYYYMMDD_HHMMSS>_<briefing_id>.json``.
        validate: When ``True`` (default) the briefing dict is validated
            against the ``IntelligenceBriefing`` Pydantic model before
            writing.  Set to ``False`` to skip validation and write as-is
            (useful for debugging partial/draft briefings).

    Returns:
        A JSON-safe dict with:

        * ``success`` (bool): ``True`` if the file was saved.
        * ``saved_path`` (str | None): Absolute path of the written file,
            or ``None`` on failure.
        * ``filename`` (str | None): Just the filename component.
        * ``briefing_id`` (str): The ``briefing_id`` from the input dict.
        * ``validation_applied`` (bool): Whether Pydantic validation ran.
        * ``error`` (str | None): Error message if the save failed.
    """
    briefing_id: str = briefing.get("briefing_id", "<unknown>")

    # --- Optional Pydantic validation ----------------------------------------
    if validate:
        try:
            from models.schemas import IntelligenceBriefing  # noqa: PLC0415
            validated = IntelligenceBriefing.model_validate(briefing)
            # Round-trip through model to normalise datetime strings etc.
            briefing = validated.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "saved_path": None,
                "filename": None,
                "briefing_id": briefing_id,
                "validation_applied": True,
                "error": f"Briefing failed schema validation: {exc}",
            }

    # --- Build filename -------------------------------------------------------
    if filename is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in briefing_id)
        filename = f"briefing_{ts}_{safe_id}.json"

    # --- Ensure briefings directory exists -----------------------------------
    try:
        _BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "success": False,
            "saved_path": None,
            "filename": filename,
            "briefing_id": briefing_id,
            "validation_applied": validate,
            "error": f"Could not create briefings directory: {exc}",
        }

    # --- Write JSON file -----------------------------------------------------
    target = _BRIEFINGS_DIR / filename
    try:
        target.write_text(
            json.dumps(briefing, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        return {
            "success": False,
            "saved_path": None,
            "filename": filename,
            "briefing_id": briefing_id,
            "validation_applied": validate,
            "error": f"Failed to write file: {exc}",
        }

    return {
        "success": True,
        "saved_path": str(target),
        "filename": filename,
        "briefing_id": briefing_id,
        "validation_applied": validate,
        "error": None,
    }
