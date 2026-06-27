"""
political_crew.py
CrewAI Crew orchestrator: PoliticalCrew

This module is the top-level entry point for the CrewAI layer.
It wires together the three agents and their tasks into a sequential crew
and exposes a single public method: PoliticalCrew.run().

Sequential process rationale
-----------------------------
The pipeline has a strict data dependency chain:
  DataCollector → NarrativeAnalyst → IntelligenceReporter

Each stage needs the output of the previous stage, so a sequential process
(rather than hierarchical or parallel) is the correct fit.  Task.context
threading makes each agent's input explicitly dependent on prior outputs.

PoliticalCrew.run() return contract
-------------------------------------
run() always returns a dict with at minimum:
  {
    "success": bool,
    "analysis_period": str,
    "raw_output": str,        # raw LLM text from the final task
    "briefing_saved": bool,
    "saved_path": str | None,
    "error": str | None,
  }

This makes the return value JSON-safe and predictable for the LangGraph
workflow that will call this crew in Phase 5.

Design decisions
----------------
* Agents and tasks are constructed inside run() rather than __init__ so
  each call to run() gets a clean agent/task state.  CrewAI agents carry
  memory between tasks within a run; re-instantiating prevents bleed
  between separate PoliticalCrew.run() calls.
* Task expected_output strings are detailed schema descriptions.  They
  guide the LLM toward structured output without hard-coding prompts in
  a separate file (prompts/ is reserved for Phase 5+ iteration).
* The briefing_id is derived from analysis_period to make saved files
  traceable back to the run that produced them.
* All exceptions are caught and surfaced in the return dict rather than
  raised — the LangGraph workflow needs a predictable interface even when
  the crew fails.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from crewai import Crew, Process, Task

from crew.data_collector import build_data_collector_agent
from crew.narrative_analyst import build_narrative_analyst_agent
from crew.intelligence_reporter import build_intelligence_reporter_agent


# ---------------------------------------------------------------------------
# Task factory helpers
# Kept separate from agent builders so tasks can reference each other
# via context without creating circular imports.
# ---------------------------------------------------------------------------


def _build_collection_task(agent) -> Task:  # type: ignore[no-untyped-def]
    """
    Task 1: Load and structure the narrative dataset.

    The DataCollector calls load_narrative_dataset(), summarises the
    records, and produces a structured report that the NarrativeAnalyst
    receives as context.
    """
    return Task(
        description=(
            "Load the complete narrative dataset from the mock data source. "
            "Use the 'Load Narrative Dataset' tool to retrieve all records. "
            "Review the dataset for completeness — note the total record count, "
            "any validation errors, and a brief characterisation of the range of "
            "topics, sentiments, and volume levels present. "
            "Do NOT analyse risk or score narratives — that is the analyst's role. "
            "Your output must be a structured summary that the analyst can act on."
        ),
        expected_output=(
            "A structured data collection report containing:\n"
            "1. Total narratives loaded (count)\n"
            "2. Any load or validation errors\n"
            "3. A list of all narrative IDs and their topics\n"
            "4. A brief characterisation of the dataset "
            "(sentiment distribution, volume range, date range)\n"
            "5. The full JSON records list for downstream use"
        ),
        agent=agent,
        # No context — this is the first task in the chain.
    )


def _build_analysis_task(agent, collection_task: Task) -> Task:  # type: ignore[no-untyped-def]
    """
    Task 2: Score each narrative and detect coordination patterns.

    Receives the DataCollector's output via context, applies both
    analytical tools, and produces a risk-ranked analytical summary.
    """
    return Task(
        description=(
            "Using the narrative records provided by the data collection step, "
            "perform two analytical operations:\n\n"
            "OPERATION 1 — Virality scoring:\n"
            "For each narrative record, use the 'Calculate Narrative Virality Score' "
            "tool to compute its composite risk score and tier. "
            "Identify the top 5 highest-scoring narratives.\n\n"
            "OPERATION 2 — Pattern detection:\n"
            "Pass all narrative records as a JSON array to the "
            "'Detect Coordination Patterns' tool. "
            "Summarise any patterns found, including which narrative IDs are affected "
            "and the confidence level.\n\n"
            "Combine both operations into a single analytical summary. "
            "Assign an overall risk level (LOW / MEDIUM / HIGH / CRITICAL) based on "
            "the distribution of individual scores and the presence of coordination patterns. "
            "Keep your language neutral and evidence-based."
        ),
        expected_output=(
            "A structured analytical summary containing:\n"
            "1. Per-narrative risk scores (id, composite_score, risk_tier) "
            "for all records, sorted highest to lowest\n"
            "2. Top 5 highest-risk narrative IDs with brief justification\n"
            "3. Detected coordination patterns (type, confidence, affected IDs)\n"
            "4. Overall risk level assessment (LOW / MEDIUM / HIGH / CRITICAL) "
            "with supporting rationale\n"
            "5. List of narrative IDs showing signs of coordinated amplification"
        ),
        agent=agent,
        # context threads the DataCollector's output into this task
        context=[collection_task],
    )


def _build_reporting_task(
    agent,
    collection_task: Task,
    analysis_task: Task,
    briefing_id: str,
    analysis_period: str,
) -> Task:  # type: ignore[no-untyped-def]
    """
    Task 3: Generate and save the executive intelligence briefing.

    Receives both prior tasks as context — the raw dataset from Task 1
    and the analytical summary from Task 2 — then produces and saves
    a structured IntelligenceBriefing.
    """
    return Task(
        description=(
            f"Using the data collection report and the analytical summary from the "
            f"previous steps, produce a complete executive intelligence briefing "
            f"for the analysis period: '{analysis_period}'.\n\n"
            f"The briefing must be structured as a JSON object with these exact fields:\n"
            f"  briefing_id: '{briefing_id}'\n"
            f"  briefing_type: 'alert' if overall risk is CRITICAL, otherwise 'standard'\n"
            f"  executive_summary: 2–4 sentence overview of the narrative landscape\n"
            f"  overall_risk_level: one of LOW / MEDIUM / HIGH / CRITICAL\n"
            f"  overall_risk_score: float 0.0–1.0 (mean of all composite scores)\n"
            f"  top_narratives: list of up to 5 highest-risk narrative IDs\n"
            f"  emerging_narratives: list of narrative IDs with growth_rate > 3.0\n"
            f"  coordination_alerts: list of narrative IDs with confirmed coordination patterns\n"
            f"  recommended_actions: list of 3–5 specific monitoring actions\n"
            f"  monitoring_priorities: list of narrative IDs to watch next cycle\n"
            f"  analyst_agent: 'IntelligenceReporter'\n\n"
            f"Once you have constructed the briefing JSON, call the "
            f"'Save Intelligence Briefing' tool with the JSON string as input. "
            f"The tool will validate and persist the file. "
            f"Include the saved_path in your final response.\n\n"
            f"Keep all language civic, neutral, and analytical. "
            f"Do not include persuasion strategies, audience targeting, "
            f"or any form of manipulation guidance."
        ),
        expected_output=(
            "A confirmation that the intelligence briefing was successfully saved, "
            "including:\n"
            "1. The complete briefing JSON that was submitted\n"
            "2. The saved_path returned by the save tool\n"
            "3. A plain-English summary of the key findings for the record"
        ),
        agent=agent,
        # Both prior tasks flow into the reporter for full context
        context=[collection_task, analysis_task],
    )


# ---------------------------------------------------------------------------
# PoliticalCrew
# ---------------------------------------------------------------------------


class PoliticalCrew:
    """
    Orchestrates the three-agent CrewAI pipeline for narrative intelligence analysis.

    Usage::

        crew = PoliticalCrew(analysis_period="2025-06-27 morning cycle")
        result = crew.run()
        print(result["raw_output"])

    Args:
        analysis_period: A human-readable label for this analysis run
            (e.g. ``"2025-06-27 morning cycle"``).  Used in task prompts
            and to generate the briefing_id.
    """

    def __init__(self, analysis_period: str) -> None:
        self.analysis_period = analysis_period
        # Derive a filesystem-safe briefing ID from the period string
        safe_period = re.sub(r"[^a-zA-Z0-9_-]", "_", analysis_period.strip())
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.briefing_id = f"briefing_{ts}_{safe_period}"

    def run(self) -> dict:
        """
        Execute the full three-agent analysis pipeline.

        Builds agents and tasks fresh on each call, assembles the crew,
        and kicks off the sequential run.  All exceptions are caught and
        returned in the result dict so callers always receive a predictable
        JSON-safe response.

        Returns:
            A dict containing:

            * ``success`` (bool): True if the crew completed without error.
            * ``analysis_period`` (str): The period label passed at init.
            * ``briefing_id`` (str): The ID used for the saved briefing file.
            * ``raw_output`` (str): The final LLM output from the reporter task.
            * ``briefing_saved`` (bool): True if save_briefing reported success.
            * ``saved_path`` (str | None): Path to the saved briefing file.
            * ``error`` (str | None): Error message if the run failed.
        """
        try:
            # -- Build agents (fresh instances per run) -----------------------
            collector_agent  = build_data_collector_agent()
            analyst_agent    = build_narrative_analyst_agent()
            reporter_agent   = build_intelligence_reporter_agent()

            # -- Build tasks with explicit context chaining -------------------
            collection_task = _build_collection_task(collector_agent)
            analysis_task   = _build_analysis_task(analyst_agent, collection_task)
            reporting_task  = _build_reporting_task(
                reporter_agent,
                collection_task,
                analysis_task,
                self.briefing_id,
                self.analysis_period,
            )

            # -- Assemble crew ------------------------------------------------
            crew = Crew(
                agents=[collector_agent, analyst_agent, reporter_agent],
                tasks=[collection_task, analysis_task, reporting_task],
                process=Process.sequential,
                verbose=True,
            )

            # -- Execute ------------------------------------------------------
            crew_output = crew.kickoff()

            # crew_output is a CrewOutput object in crewai v1.x
            # .raw gives the final task's text output
            raw_output: str = (
                crew_output.raw
                if hasattr(crew_output, "raw")
                else str(crew_output)
            )

            # -- Extract saved_path from output if present --------------------
            saved_path = self._extract_saved_path(raw_output)

            return {
                "success": True,
                "analysis_period": self.analysis_period,
                "briefing_id": self.briefing_id,
                "raw_output": raw_output,
                "briefing_saved": saved_path is not None,
                "saved_path": saved_path,
                "error": None,
            }

        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "analysis_period": self.analysis_period,
                "briefing_id": self.briefing_id,
                "raw_output": "",
                "briefing_saved": False,
                "saved_path": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_saved_path(self, text: str) -> str | None:
        """
        Attempt to parse a saved_path from the reporter's raw output.

        The IntelligenceReporter is instructed to include the save tool's
        returned saved_path in its final response.  This helper extracts
        it via a simple path pattern so callers can reference the file
        without re-parsing the full LLM output.
        """
        # Look for an absolute or relative path ending in .json
        match = re.search(r'(/[^\s"\']+briefings/[^\s"\']+\.json)', text)
        if match:
            return match.group(1)

        # Fallback: look for a JSON fragment containing saved_path
        try:
            snippet_match = re.search(r'\{[^}]*"saved_path"\s*:\s*"([^"]+)"[^}]*\}', text)
            if snippet_match:
                return snippet_match.group(1)
        except Exception:  # noqa: BLE001
            pass

        return None
