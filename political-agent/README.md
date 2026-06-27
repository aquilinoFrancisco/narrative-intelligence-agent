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
                                       alert_briefing   standard_briefing
                                            ↓              ↓
                                         save_report   save_report
       ↓
  [Streamlit UI — port 8000]
```

## Tech Stack

- Python 3.11
- Streamlit (UI)
- LangGraph (workflow orchestration)
- CrewAI (multi-agent framework)
- MCP protocol (tool serving)
- OpenAI API (LLM reasoning)
- Mock JSON data (no real APIs)

## Project Structure

```
political-agent/
├── app/                    # Streamlit UI
│   ├── main_app.py
│   └── components/
├── graph/                  # LangGraph workflow
│   ├── state.py
│   └── workflow.py
├── crew/                   # CrewAI agents
│   ├── data_collector.py
│   ├── narrative_analyst.py
│   ├── intelligence_reporter.py
│   └── political_crew.py
├── agent_mcp/              # MCP server + tools
│   ├── server.py
│   └── tools/
│       ├── read_data.py
│       ├── search_narrative.py
│       ├── calculate_virality_score.py
│       ├── detect_patterns.py
│       └── save_briefing.py
├── data/
│   └── mock_narratives.json
├── briefings/              # Generated reports saved here
├── prompts/                # Agent prompt templates
├── models/                 # Pydantic data models
├── tests/
├── pyproject.toml
└── README.md
```

## Development Phases

- **Phase 1** ✅ Folder structure and placeholder files
- **Phase 2** 🔜 Business logic implementation
- **Phase 3** 🔜 Streamlit UI and visualization
- **Phase 4** 🔜 Testing and documentation

## Setup (Phase 2+)

```bash
cd political-agent
pip install -e ".[dev]"
streamlit run app/main_app.py --server.port 8000
```
