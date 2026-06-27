"""
uploader.py
Streamlit component — mock dataset source panel.

For this MVP there is no file upload.  This component shows the user
what data source the pipeline will read from and provides a quick
preview of the record schema and count.

No file upload logic, no analysis logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_FILE    = _PROJECT_ROOT / "data" / "mock_narratives.json"


def render_data_source() -> None:
    """
    Display mock dataset information and a record count preview.

    Reads the JSON file directly (no MCP call) so this panel is
    always populated even before the pipeline runs.
    """
    with st.expander("📂 Data Source", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Source file**")
            st.code(str(_DATA_FILE.relative_to(_PROJECT_ROOT.parent)), language=None)
            st.caption(
                "All narrative records are fictional and generated for "
                "demonstration purposes only."
            )

        with col2:
            # Quick record count without going through MCP
            try:
                records = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
                real_records = [r for r in records if "_comment" not in r]
                st.metric("Records", len(real_records))
                st.metric("Fields / record", 14)
            except Exception:
                st.warning("Could not read data file.")

        # --- Field schema preview -------------------------------------------
        st.markdown("**Record schema**")
        schema_cols = st.columns(3)
        fields = [
            ("id",                   "str"),
            ("topic",                "str"),
            ("sentiment",            "str"),
            ("volume",               "int"),
            ("growth_rate",          "float"),
            ("virality_score",       "float 0–1"),
            ("coordination_score",   "float 0–1"),
            ("bot_probability",      "float 0–1"),
            ("accounts_pushing",     "int"),
            ("suspicious_accounts",  "int"),
            ("first_seen",           "datetime"),
            ("peak_hour",            "str"),
            ("sample_posts",         "list[str]"),
        ]
        chunk = (len(fields) + 2) // 3
        for i, col in enumerate(schema_cols):
            with col:
                for name, typ in fields[i * chunk: (i + 1) * chunk]:
                    st.markdown(f"`{name}` — *{typ}*")

        # --- Sample record preview ------------------------------------------
        st.markdown("**Sample record (narrative_001)**")
        try:
            sample = next(
                (r for r in real_records if r.get("id") == "narrative_001"), real_records[0]
            )
            st.json(
                {k: v for k, v in sample.items() if k != "sample_posts"},
                expanded=False,
            )
        except Exception:
            pass
