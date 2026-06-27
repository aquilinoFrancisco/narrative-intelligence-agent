# Political Narrative Intelligence Agent

A technical portfolio project demonstrating civic intelligence and misinformation monitoring
using a multi-agent AI architecture.

> **Note:** This is a portfolio/demonstration project using mock data only.
> It is not a real political manipulation tool.

## Architecture

```
mock_narratives.json
       ↓
  [MCP Tools]  ←  read_data | search_narrative | calculate_virality_score | detect_patterns
       ↓
  [CrewAI Agents]
  DataCollector → NarrativeAnalyst → IntelligenceReporter
       ↓
  [LangGraph Workflow]
  collect_data → analyze_narratives → evaluate_threat_level
                                            ↓              ↓
                                       CRITICAL         NORMAL
