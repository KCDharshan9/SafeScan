"""SafeScan application entry point.

Run from the project root:

    python -m streamlit run app/streamlit_app.py

The app has four pages:

* Overview: product purpose and recorded performance
* Triage: scan a file and set the review threshold
* Model Insights: evaluation tables, curves, and feature importance
* Dataset & Method: data, features, and pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="SafeScan | Security Triage",
    page_icon=str(Path(__file__).resolve().parent / "static" / "safescan-mark.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import init_theme  # noqa: E402

init_theme()

pages = [
    st.Page("views/overview.py", title="Overview", url_path="overview", default=True),
    st.Page("views/triage.py", title="Triage", url_path="triage"),
    st.Page("views/model_insights.py", title="Model Insights", url_path="model-insights"),
    st.Page("views/dataset_method.py", title="Dataset & Method", url_path="dataset-method"),
]

navigation = st.navigation(pages, position="hidden")
navigation.run()
