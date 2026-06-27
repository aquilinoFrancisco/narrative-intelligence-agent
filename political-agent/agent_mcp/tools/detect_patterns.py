"""
detect_patterns.py
MCP Tool: detect_patterns

Analyses a list of narrative records for suspicious coordination signals
and returns a structured pattern report.

Three pattern detectors are applied:

1. PEAK_HOUR_SYNCHRONY
   Multiple narratives sharing the same peak activity hour — suggests a
   common scheduler or coordinated posting window.

2. HIGH_COORDINATION_CLUSTER
   A group of narratives all scoring >= 0.65 coordination_score — suggests
   a shared inorganic amplification network.

3. BOT_AMPLIFICATION_SURGE
   Narratives where bot_probability >= 0.60 — signals heavy automated
   involvement pushing several themes simultaneously.

This tool is used by the NarrativeAnalyst CrewAI agent.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Detection thresholds
_COORDINATION_THRESHOLD = 0.65
_BOT_THRESHOLD = 0.60
_PEAK_HOUR_MIN_CLUSTER = 2          # minimum narratives to flag hour synchrony
_COORDINATION_MIN_CLUSTER = 2       # minimum narratives to flag coordination cluster
_BOT_MIN_CLUSTER = 2                # minimum narratives to flag bot surge


def _confidence(matching: int, total: int) -> float:
    """Return a normalised confidence score (0.0–1.0) for a pattern."""
    if total == 0:
        return 0.0
    return round(min(1.0, matching / total), 4)


def detect_patterns(narratives: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Scan a list of narrative records for suspicious coordination patterns.

    Each pattern entry in the result contains:
    * ``pattern_type``         — a stable identifier string for the pattern class
    * ``confidence``           — float in [0.0, 1.0]; higher is more suspicious
    * ``affected_narratives``  — list of ``id`` values involved in the pattern
    * ``description``          — plain-English explanation

    Args:
        narratives: A list of narrative dicts.  Each must contain at minimum:
            ``"id"``, ``"coordination_score"``, ``"bot_probability"``,
            ``"peak_hour"``.  Extra fields are ignored.

    Returns:
        A JSON-safe dict with:

        * ``success`` (bool): ``True`` if analysis completed without error.
        * ``total_analysed`` (int): Number of records processed.
        * ``pattern_count`` (int): Number of distinct patterns detected.
        * ``patterns`` (list[dict]): One entry per detected pattern.
        * ``summary`` (str): Plain-English summary of findings.
        * ``error`` (str | None): Error message if analysis failed.
    """
    if not narratives:
        return {
            "success": False,
            "total_analysed": 0,
            "pattern_count": 0,
            "patterns": [],
            "summary": "No narratives provided.",
            "error": "narratives list is empty.",
        }

    total = len(narratives)
    patterns: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. PEAK_HOUR_SYNCHRONY
    #    Group narratives by their peak_hour value and flag any hour shared
    #    by two or more narratives.
    # -------------------------------------------------------------------------
    hour_buckets: dict[str, list[str]] = defaultdict(list)
    for rec in narratives:
        peak = rec.get("peak_hour")
        nid  = rec.get("id", "<unknown>")
        if peak:
            hour_buckets[peak].append(nid)

    for hour, ids in hour_buckets.items():
        if len(ids) >= _PEAK_HOUR_MIN_CLUSTER:
            patterns.append(
                {
                    "pattern_type": "PEAK_HOUR_SYNCHRONY",
                    "confidence": _confidence(len(ids), total),
                    "affected_narratives": ids,
                    "description": (
                        f"{len(ids)} narrative(s) share the same peak activity window "
                        f"'{hour}', suggesting a coordinated posting schedule or "
                        "common scheduler infrastructure."
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # 2. HIGH_COORDINATION_CLUSTER
    #    Collect narratives with coordination_score >= threshold.
    # -------------------------------------------------------------------------
    high_coord_ids: list[str] = []
    for rec in narratives:
        try:
            if float(rec.get("coordination_score", 0.0)) >= _COORDINATION_THRESHOLD:
                high_coord_ids.append(rec.get("id", "<unknown>"))
        except (TypeError, ValueError):
            continue

    if len(high_coord_ids) >= _COORDINATION_MIN_CLUSTER:
        patterns.append(
            {
                "pattern_type": "HIGH_COORDINATION_CLUSTER",
                "confidence": _confidence(len(high_coord_ids), total),
                "affected_narratives": high_coord_ids,
                "description": (
                    f"{len(high_coord_ids)} narrative(s) exceed the coordination "
                    f"threshold ({_COORDINATION_THRESHOLD}), indicating a shared "
                    "inorganic amplification network may be promoting multiple themes."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # 3. BOT_AMPLIFICATION_SURGE
    #    Collect narratives with bot_probability >= threshold.
    # -------------------------------------------------------------------------
    high_bot_ids: list[str] = []
    for rec in narratives:
        try:
            if float(rec.get("bot_probability", 0.0)) >= _BOT_THRESHOLD:
                high_bot_ids.append(rec.get("id", "<unknown>"))
        except (TypeError, ValueError):
            continue

    if len(high_bot_ids) >= _BOT_MIN_CLUSTER:
        patterns.append(
            {
                "pattern_type": "BOT_AMPLIFICATION_SURGE",
                "confidence": _confidence(len(high_bot_ids), total),
                "affected_narratives": high_bot_ids,
                "description": (
                    f"{len(high_bot_ids)} narrative(s) have bot_probability >= "
                    f"{_BOT_THRESHOLD}, indicating simultaneous automated amplification "
                    "across multiple themes — a hallmark of coordinated inauthentic behaviour."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Build summary
    # -------------------------------------------------------------------------
    if patterns:
        type_labels = ", ".join(p["pattern_type"] for p in patterns)
        summary = (
            f"{len(patterns)} pattern(s) detected across {total} narratives: "
            f"{type_labels}."
        )
    else:
        summary = (
            f"No significant coordination patterns detected across {total} narratives."
        )

    return {
        "success": True,
        "total_analysed": total,
        "pattern_count": len(patterns),
        "patterns": patterns,
        "summary": summary,
        "error": None,
    }
