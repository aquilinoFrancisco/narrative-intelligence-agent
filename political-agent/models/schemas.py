"""
schemas.py
Pydantic data models for the Political Narrative Intelligence Agent pipeline.

All score fields are validated in the range [0.0, 1.0].
Risk/threat levels are validated against the allowed enum values.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    """Sentiment classification for a narrative."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class RiskLevel(str, Enum):
    """Threat/risk tier used throughout the pipeline."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BriefingType(str, Enum):
    """Determines which LangGraph branch produced the briefing."""

    STANDARD = "standard"
    ALERT = "alert"


# ---------------------------------------------------------------------------
# NarrativeRecord
# ---------------------------------------------------------------------------


class NarrativeRecord(BaseModel):
    """
    A single narrative observed in the mock dataset.

    Represents one distinct information theme being tracked —
    its spread metrics, source attribution signals, and sample content.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the narrative (e.g. 'narrative_001').",
    )
    topic: str = Field(
        ...,
        description="Short label describing the narrative theme.",
    )
    sentiment: SentimentLabel = Field(
        ...,
        description="Overall sentiment of the narrative content.",
    )
    volume: int = Field(
        ...,
        ge=0,
        description="Total number of posts/mentions observed.",
    )
    growth_rate: float = Field(
        ...,
        ge=0.0,
        description="Percentage growth in volume over the observation window (e.g. 3.5 = 350%).",
    )
    virality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite virality score normalised to [0.0, 1.0].",
    )
    coordination_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Likelihood that the spread is coordinated rather than organic. "
            "0.0 = fully organic, 1.0 = highly coordinated."
        ),
    )
    bot_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated fraction of accounts pushing this narrative that are automated.",
    )
    accounts_pushing: int = Field(
        ...,
        ge=0,
        description="Total number of distinct accounts spreading this narrative.",
    )
    suspicious_accounts: int = Field(
        ...,
        ge=0,
        description="Subset of accounts_pushing flagged as suspicious or automated.",
    )
    first_seen: datetime = Field(
        ...,
        description="UTC timestamp when the narrative was first observed.",
    )
    peak_hour: str = Field(
        ...,
        description="UTC hour range of highest activity (e.g. '14:00-15:00 UTC').",
    )
    sample_posts: list[str] = Field(
        default_factory=list,
        description="Up to 5 representative post excerpts for qualitative review.",
    )

    @field_validator("suspicious_accounts")
    @classmethod
    def suspicious_cannot_exceed_total(cls, v: int, info) -> int:  # noqa: ANN001
        """Suspicious accounts must not exceed the total accounts pushing the narrative."""
        total = info.data.get("accounts_pushing")
        if total is not None and v > total:
            raise ValueError(
                f"suspicious_accounts ({v}) cannot exceed accounts_pushing ({total})."
            )
        return v

    @field_validator("sample_posts")
    @classmethod
    def limit_sample_posts(cls, v: list[str]) -> list[str]:
        """Enforce a maximum of 5 sample posts per narrative."""
        if len(v) > 5:
            raise ValueError("sample_posts may contain at most 5 entries.")
        return v

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# NarrativeDataset
# ---------------------------------------------------------------------------


class NarrativeDataset(BaseModel):
    """
    The full collection of narrative records loaded from mock_narratives.json.

    Acts as the container passed into the pipeline by the DataCollector agent.
    """

    records: list[NarrativeRecord] = Field(
        default_factory=list,
        description="All narrative records in the dataset.",
    )
    loaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the dataset was loaded into memory.",
    )
    source_file: str = Field(
        default="data/mock_narratives.json",
        description="Path to the JSON file this dataset was loaded from.",
    )

    @property
    def count(self) -> int:
        """Total number of narrative records."""
        return len(self.records)

    @property
    def high_risk_records(self) -> list[NarrativeRecord]:
        """Records with virality_score >= 0.7 or coordination_score >= 0.7."""
        return [
            r
            for r in self.records
            if r.virality_score >= 0.7 or r.coordination_score >= 0.7
        ]


# ---------------------------------------------------------------------------
# NarrativeAnalysis
# ---------------------------------------------------------------------------


class NarrativeAnalysis(BaseModel):
    """
    Output produced by the NarrativeAnalyst CrewAI agent.

    Contains per-narrative assessments and an aggregate summary
    ready for threat evaluation by LangGraph.
    """

    narrative_id: str = Field(
        ...,
        description="Foreign key referencing NarrativeRecord.id.",
    )
    emerging: bool = Field(
        ...,
        description="True if this narrative is growing faster than the dataset average.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Composite risk score derived from virality, coordination, and bot signals. "
            "Normalised to [0.0, 1.0]."
        ),
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk tier derived from risk_score.",
    )
    coordination_detected: bool = Field(
        ...,
        description="True if coordination_score exceeds the suspicious threshold (>= 0.65).",
    )
    analyst_notes: str = Field(
        default="",
        description="Free-text observations produced by the NarrativeAnalyst agent.",
    )

    @model_validator(mode="after")
    def risk_level_consistent_with_score(self) -> "NarrativeAnalysis":
        """Verify that the categorical risk_level is consistent with the numeric risk_score."""
        score = self.risk_score
        level = self.risk_level

        boundaries = {
            RiskLevel.LOW: (0.0, 0.34),
            RiskLevel.MEDIUM: (0.35, 0.59),
            RiskLevel.HIGH: (0.60, 0.79),
            RiskLevel.CRITICAL: (0.80, 1.0),
        }
        lo, hi = boundaries[RiskLevel(level)]
        if not (lo <= score <= hi):
            raise ValueError(
                f"risk_level '{level}' is inconsistent with risk_score {score:.2f}. "
                f"Expected score in [{lo}, {hi}]."
            )
        return self

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# ThreatAssessment
# ---------------------------------------------------------------------------


class ThreatAssessment(BaseModel):
    """
    Aggregate threat assessment produced by the evaluate_threat_level
    LangGraph node from all NarrativeAnalysis results.

    Determines which briefing branch (CRITICAL vs NORMAL) is triggered.
    """

    overall_risk_level: RiskLevel = Field(
        ...,
        description="Highest risk level observed across all analysed narratives.",
    )
    overall_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean risk score across all narratives, normalised to [0.0, 1.0].",
    )
    critical_narratives: list[str] = Field(
        default_factory=list,
        description="List of narrative IDs classified as CRITICAL.",
    )
    high_narratives: list[str] = Field(
        default_factory=list,
        description="List of narrative IDs classified as HIGH.",
    )
    coordination_alerts: list[str] = Field(
        default_factory=list,
        description="Narrative IDs where coordinated amplification was detected.",
    )
    total_narratives_assessed: int = Field(
        ...,
        ge=0,
        description="Total number of narratives included in this assessment.",
    )
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of when the threat assessment was produced.",
    )
    triggers_alert_branch: bool = Field(
        default=False,
        description=(
            "True if the overall_risk_level is CRITICAL, which routes the "
            "LangGraph workflow to the alert_briefing node instead of standard_briefing."
        ),
    )

    @model_validator(mode="after")
    def set_alert_branch_flag(self) -> "ThreatAssessment":
        """Auto-set triggers_alert_branch based on overall_risk_level."""
        self.triggers_alert_branch = self.overall_risk_level == RiskLevel.CRITICAL
        return self

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# IntelligenceBriefing
# ---------------------------------------------------------------------------


class IntelligenceBriefing(BaseModel):
    """
    Final executive briefing produced by the IntelligenceReporter CrewAI agent.

    This is the terminal output of the pipeline — persisted to disk by
    the save_briefing MCP tool and displayed in the Streamlit UI.
    """

    briefing_id: str = Field(
        ...,
        description="Unique identifier for this briefing (e.g. 'briefing_20250627_143022').",
    )
    briefing_type: BriefingType = Field(
        ...,
        description="Whether this is a STANDARD or ALERT briefing.",
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the briefing was generated.",
    )

    # --- Executive summary ---
    executive_summary: str = Field(
        ...,
        description="2–4 sentence high-level summary of the current narrative landscape.",
    )
    overall_risk_level: RiskLevel = Field(
        ...,
        description="Copied from ThreatAssessment for top-level visibility.",
    )
    overall_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric risk score (0.0–1.0) accompanying the risk level.",
    )

    # --- Top narratives ---
    top_narratives: list[str] = Field(
        default_factory=list,
        description="Ordered list of the most significant narrative IDs (highest risk first).",
    )
    emerging_narratives: list[str] = Field(
        default_factory=list,
        description="Narrative IDs flagged as rapidly growing.",
    )
    coordination_alerts: list[str] = Field(
        default_factory=list,
        description="Narrative IDs with confirmed coordination signals.",
    )

    # --- Recommendations ---
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Ordered list of recommended actions for analysts or decision-makers.",
    )
    monitoring_priorities: list[str] = Field(
        default_factory=list,
        description="Narrative IDs or topics that should be monitored closely in the next cycle.",
    )

    # --- Metadata ---
    analyst_agent: str = Field(
        default="IntelligenceReporter",
        description="Name of the CrewAI agent that produced this briefing.",
    )
    report_path: Optional[str] = Field(
        default=None,
        description="Filesystem path where the briefing JSON was saved (set after save_briefing).",
    )

    model_config = {"use_enum_values": True}
