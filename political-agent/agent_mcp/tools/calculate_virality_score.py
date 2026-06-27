"""
calculate_virality_score.py
MCP Tool: calculate_virality_score

Computes a composite virality/amplification score for a single narrative
record and assigns it a categorical risk tier.

The composite score is a weighted combination of three signals already
present in each NarrativeRecord:
  - virality_score     (weight 0.40) — organic spread velocity
  - coordination_score (weight 0.35) — inorganic / coordinated amplification
  - bot_probability    (weight 0.25) — estimated automation fraction

Risk tier thresholds (applied to the composite score):
  LOW      [0.00 – 0.34]
  MEDIUM   [0.35 – 0.59]
  HIGH     [0.60 – 0.79]
  CRITICAL [0.80 – 1.00]

This tool is used by the NarrativeAnalyst CrewAI agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Weights must sum to 1.0
_W_VIRALITY = 0.40
_W_COORDINATION = 0.35
_W_BOT = 0.25

# Tier boundaries: (lower_inclusive, upper_inclusive)
_TIER_BOUNDARIES: list[tuple[str, float, float]] = [
    ("CRITICAL", 0.80, 1.00),
    ("HIGH",     0.60, 0.79),
    ("MEDIUM",   0.35, 0.59),
    ("LOW",      0.00, 0.34),
]


def _assign_tier(score: float) -> str:
    """Return the categorical risk tier for a given composite score."""
    for tier, lo, hi in _TIER_BOUNDARIES:
        if lo <= score <= hi:
            return tier
    # Clamp edge case (floating-point rounding)
    return "CRITICAL" if score > 0.80 else "LOW"


def calculate_virality_score(narrative: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a composite amplification score for a single narrative record.

    The function reads three pre-computed signal fields from *narrative* and
    combines them using fixed weights to produce a normalised score in
    ``[0.0, 1.0]``.  It also flags whether the narrative crosses the
    suspicious-coordination threshold and returns a human-readable
    interpretation.

    Args:
        narrative: A dict representing a single narrative record.  At minimum
            it must contain:
            * ``"id"``                  (str)
            * ``"virality_score"``      (float, 0.0–1.0)
            * ``"coordination_score"``  (float, 0.0–1.0)
            * ``"bot_probability"``     (float, 0.0–1.0)
            Additional fields are ignored.

    Returns:
        A JSON-safe dict with:

        * ``success`` (bool): ``True`` if the calculation completed.
        * ``narrative_id`` (str): The ``id`` field from the input.
        * ``composite_score`` (float): Weighted composite, rounded to 4 d.p.
        * ``risk_tier`` (str): Categorical label — LOW / MEDIUM / HIGH / CRITICAL.
        * ``weights_used`` (dict): The weights applied to each signal.
        * ``signal_values`` (dict): The raw signal values read from the record.
        * ``coordination_alert`` (bool): ``True`` when coordination_score >= 0.65.
        * ``interpretation`` (str): Plain-English summary of the score.
        * ``error`` (str | None): Error message if calculation failed.
    """
    narrative_id: str = narrative.get("id", "<unknown>")

    # --- Extract and validate signal fields ----------------------------------
    required_signals = ("virality_score", "coordination_score", "bot_probability")
    missing = [f for f in required_signals if f not in narrative]
    if missing:
        return {
            "success": False,
            "narrative_id": narrative_id,
            "composite_score": None,
            "risk_tier": None,
            "weights_used": {},
            "signal_values": {},
            "coordination_alert": False,
            "interpretation": "",
            "error": f"Missing required fields: {missing}",
        }

    try:
        v_score = float(narrative["virality_score"])
        c_score = float(narrative["coordination_score"])
        b_prob  = float(narrative["bot_probability"])
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "narrative_id": narrative_id,
            "composite_score": None,
            "risk_tier": None,
            "weights_used": {},
            "signal_values": {},
            "coordination_alert": False,
            "interpretation": "",
            "error": f"Non-numeric signal value: {exc}",
        }

    # Clamp each signal to [0.0, 1.0] defensively
    v_score = max(0.0, min(1.0, v_score))
    c_score = max(0.0, min(1.0, c_score))
    b_prob  = max(0.0, min(1.0, b_prob))

    # --- Compute composite score ---------------------------------------------
    composite = round(
        _W_VIRALITY * v_score + _W_COORDINATION * c_score + _W_BOT * b_prob,
        4,
    )
    tier = _assign_tier(composite)
    coordination_alert = c_score >= 0.65

    # --- Build interpretation ------------------------------------------------
    parts: list[str] = []
    if tier == "CRITICAL":
        parts.append("Extremely high amplification across all signals.")
    elif tier == "HIGH":
        parts.append("Significant amplification detected.")
    elif tier == "MEDIUM":
        parts.append("Moderate amplification; warrants monitoring.")
    else:
        parts.append("Low amplification; appears largely organic.")

    if coordination_alert:
        parts.append(
            f"Coordination score {c_score:.2f} exceeds the alert threshold (0.65) — "
            "suspected inorganic promotion."
        )
    if b_prob >= 0.60:
        parts.append(
            f"Bot probability {b_prob:.2f} indicates heavy automated involvement."
        )

    return {
        "success": True,
        "narrative_id": narrative_id,
        "composite_score": composite,
        "risk_tier": tier,
        "weights_used": {
            "virality_score":     _W_VIRALITY,
            "coordination_score": _W_COORDINATION,
            "bot_probability":    _W_BOT,
        },
        "signal_values": {
            "virality_score":     v_score,
            "coordination_score": c_score,
            "bot_probability":    b_prob,
        },
        "coordination_alert": coordination_alert,
        "interpretation": " ".join(parts),
        "error": None,
    }
