"""Shared design system, data helpers and UI components for SafeScan.

The app is split into four Streamlit pages (Overview, Triage, Model Insights,
Dataset & Method).  This module keeps them visually consistent and avoids
duplicating the model/data loading logic.

Run the app from the project root:

    python -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import re
import sys
import tempfile
from pathlib import Path
from string import Template

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Make `src/` importable when Streamlit runs a page directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ALERTS_CSV  # noqa: E402
from features import CONTEXT_COLUMNS  # noqa: E402
from predict import analyze as _analyze  # noqa: E402
from predict import load_model  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DEFAULT_THRESHOLD = 0.30

THEME_TOGGLE = st.components.v2.component(
    "safescan.theme_toggle",
    html='<button type="button" aria-label="Toggle colour theme"><span class="toggle-track"><span class="toggle-knob"></span></span><span class="toggle-label"></span></button>',
    css="""
    button { align-items:center; background:transparent; border:1px solid var(--st-border-color); border-radius:999px; color:var(--st-text-color); cursor:pointer; display:flex; font:500 12px/1.1 system-ui,sans-serif; gap:8px; padding:6px 9px; }
    button:hover { border-color:var(--st-primary-color); }
    button:focus-visible { outline:2px solid var(--st-primary-color); outline-offset:2px; }
    .toggle-track { background:var(--st-secondary-background-color); border:1px solid var(--st-border-color); border-radius:999px; display:block; height:14px; padding:2px; width:27px; }
    .toggle-knob { background:var(--st-primary-color); border-radius:50%; display:block; height:14px; transform:translateX(0); transition:transform .2s ease; width:14px; }
    button[data-theme='dark'] .toggle-knob { transform:translateX(13px); }
    """,
    js="""
    export default function(component) {
      const { data, parentElement, setStateValue } = component;
      const key = 'safescan.theme';
      const saved = window.localStorage.getItem(key);
      const initial = saved === 'light' || saved === 'dark' ? saved : data.systemTheme;
      const button = parentElement.querySelector('button');
      const label = parentElement.querySelector('.toggle-label');
      const render = (theme) => {
        button.dataset.theme = theme;
        button.setAttribute('aria-label', `Use ${theme === 'dark' ? 'light' : 'dark'} theme`);
        label.textContent = theme === 'dark' ? 'Dark' : 'Light';
      };
      render(data.theme || initial);
      button.onclick = () => {
        const next = button.dataset.theme === 'dark' ? 'light' : 'dark';
        window.localStorage.setItem(key, next);
        render(next);
        setStateValue('theme', next);
      };
      if (data.theme !== initial) setStateValue('theme', initial);
    }
    """,
)

# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "bg": "#fffcf7",
        "bg_soft": "#f7f3ed",
        "card": "#fffefa",
        "card_soft": "#f8f4ee",
        "border": "#e9e2d9",
        "border_strong": "#d8cec2",
        "text": "#172033",
        "muted": "#647084",
        "faint": "#8c96a6",
        "accent": "#155eef",
        "accent_soft": "rgba(21,94,239,0.09)",
        "accent_border": "rgba(21,94,239,0.22)",
        "good": "#067a46",
        "good_soft": "rgba(6,122,70,0.09)",
        "warn": "#b45309",
        "warn_soft": "rgba(180,83,9,0.10)",
        "danger": "#c81e1e",
        "danger_soft": "rgba(200,30,30,0.08)",
        "code_bg": "#f8f4ee",
        "btn_bg": "#172033",
        "btn_fg": "#ffffff",
        "shadow": "0 1px 2px rgba(0,0,0,0.04)",
    },
    "dark": {
        "bg": "#000000",
        "bg_soft": "#0a0a0a",
        "card": "#0c0c0c",
        "card_soft": "#141414",
        "border": "#282828",
        "border_strong": "#3a3a3a",
        "text": "#ededed",
        "muted": "#9b9b9b",
        "faint": "#6b6b6b",
        "accent": "#52a8ff",
        "accent_soft": "rgba(82,168,255,0.13)",
        "accent_border": "rgba(82,168,255,0.32)",
        "good": "#3ddc97",
        "good_soft": "rgba(61,220,151,0.12)",
        "warn": "#f5a623",
        "warn_soft": "rgba(245,166,35,0.12)",
        "danger": "#ff6166",
        "danger_soft": "rgba(255,97,102,0.12)",
        "code_bg": "#0a0a0a",
        "btn_bg": "#ededed",
        "btn_fg": "#0a0a0a",
        "shadow": "0 1px 2px rgba(0,0,0,0.5)",
    },
}

CSS = Template(
    """
:root {
    --bg: $bg;
    --bg-soft: $bg_soft;
    --card: $card;
    --card-soft: $card_soft;
    --border: $border;
    --border-strong: $border_strong;
    --text: $text;
    --muted: $muted;
    --faint: $faint;
    --accent: $accent;
    --accent-soft: $accent_soft;
    --accent-border: $accent_border;
    --good: $good;
    --good-soft: $good_soft;
    --warn: $warn;
    --warn-soft: $warn_soft;
    --danger: $danger;
    --danger-soft: $danger_soft;
    --code-bg: $code_bg;
    --btn-bg: $btn_bg;
    --btn-fg: $btn_fg;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu, footer { display: none !important; }

.block-container {
    max-width: 1180px;
    padding: 1.1rem 2rem 4.5rem 2rem;
}
@media (max-width: 760px) {
    .block-container { padding: 1rem 1rem 3rem 1rem; }
}

h1, h2, h3, h4, h5, h6 { color: var(--text); letter-spacing: -0.03em; }
p, li, span, label, div { color: inherit; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---------------------------------------------------------------- top bar */
.brand {
    display: flex; align-items: center; gap: 0.5rem;
    color: var(--text); padding-top: 0.3rem; white-space: nowrap;
}
.brand-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent-soft), var(--good-soft));
    color: var(--accent); border: 1px solid var(--accent-border);
}
.brand-mark svg { width: 18px; height: 18px; display: block; }
.brand-name { font-size: 1.08rem; font-weight: 760; letter-spacing: -0.045em; }
.brand-label { border-left: 1px solid var(--border-strong); color: var(--faint); font-family: 'JetBrains Mono', monospace; font-size: .62rem; letter-spacing: .1em; margin-left: .1rem; padding-left: .5rem; text-transform: uppercase; }
.brand-dot { color: var(--accent); }
.topbar-line { height: 1px; background: var(--border); margin: 0.55rem 0 2.4rem 0; }

.st-key-topnav [data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.15rem !important;
    min-height: 0 !important;
    justify-content: flex-end;
}
.st-key-topnav [data-testid="stButton"] button p { font-size: 0.88rem !important; }
.st-key-topnav [data-testid="stButton"] button:hover { color: var(--text) !important; }
.st-key-topnav [data-testid="stButton"] button[kind="primary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: inset 0 -2px 0 var(--text) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
.st-key-topnav [data-testid="stButton"] button[kind="primary"] p {
    font-weight: 600 !important;
    color: var(--text) !important;
}
.st-key-topnav [data-testid="stButton"] button[kind="primary"]::after {
    display: none;
}

/* ------------------------------------------------------------------ hero */
.hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--border); border-radius: 18px;
    padding: 4rem 3rem 3.4rem 3rem;
    background:
        radial-gradient(900px 380px at 8% -20%, var(--accent-soft), transparent 65%),
        radial-gradient(700px 320px at 105% 0%, var(--warn-soft), transparent 60%),
        var(--card);
    margin-bottom: 1.1rem;
}
.hero h1 {
    font-size: clamp(2.6rem, 6vw, 4.2rem);
    line-height: 1.02; letter-spacing: -0.055em;
    font-weight: 800; margin: 1rem 0 1.1rem 0; max-width: 900px;
}
.hero p.lede {
    font-size: 1.05rem; line-height: 1.65; color: var(--muted);
    max-width: 700px; margin: 0 0 1.7rem 0;
}
.hero .grad {
    background: linear-gradient(92deg, #007cf0 0%, #00b8d9 52%, #00dfd8 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.meta-row {
    display: flex; flex-wrap: wrap; gap: 0.45rem 1.4rem;
    color: var(--faint); font-size: 0.78rem; margin-top: 1.6rem;
    font-family: 'JetBrains Mono', monospace;
}
.meta-row span::before { content: "·"; margin-right: 1.4rem; color: var(--border-strong); }
.meta-row span:first-child::before { content: none; margin: 0; }

.st-key-hero, .st-key-hero[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 3.2rem 3rem 2.1rem 3rem !important;
    background:
        radial-gradient(900px 380px at 8% -20%, var(--accent-soft), transparent 65%),
        radial-gradient(700px 320px at 105% 0%, var(--warn-soft), transparent 60%),
        var(--card) !important;
    overflow: hidden;
}
.st-key-hero::before {
    content: ""; position: absolute; width: 360px; height: 360px; border-radius: 50%;
    right: -150px; top: -210px; pointer-events: none;
    background: radial-gradient(circle, rgba(20,184,166,.18), transparent 68%);
}
.st-key-hero [data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important; background: transparent !important; padding: 0 !important;
}
.st-key-hero h1 {
    font-size: clamp(2.6rem, 6vw, 4.2rem);
    line-height: 1.02; letter-spacing: -0.055em;
    font-weight: 800; margin: 1rem 0 1.1rem 0; max-width: 900px;
    color: var(--text);
}
.st-key-hero p.lede {
    font-size: 1.05rem; line-height: 1.65; color: var(--muted);
    max-width: 720px; margin: 0 0 1.55rem 0;
}
.st-key-hero .grad {
    background: linear-gradient(92deg, #007cf0 0%, #00b8d9 52%, #00dfd8 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
@media (max-width: 760px) {
    .st-key-hero, .st-key-hero[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 2rem 1.4rem 1.6rem 1.4rem !important;
    }
}

/* ------------------------------------------------------------------ bits */
.pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    border: 1px solid var(--border); background: var(--card-soft);
    color: var(--muted); border-radius: 999px;
    padding: 0.28rem 0.7rem; font-size: 0.75rem; font-weight: 500;
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 0.55rem;
}
.muted { color: var(--muted); }
.faint { color: var(--faint); }
.mono { font-family: 'JetBrains Mono', monospace; }
.section { margin: 3.2rem 0 1.2rem 0; position: relative; padding-left: 1rem; }
.section::before { background: linear-gradient(180deg, var(--accent), #14b8a6); border-radius: 99px; content: ""; height: 2.6rem; left: 0; position: absolute; top: .1rem; width: 3px; }
.section h2 { font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 700; margin: 0 0 0.35rem 0; }
.section p { color: var(--muted); max-width: 720px; margin: 0; line-height: 1.6; font-size: 0.95rem; }
.divider { height: 1px; background: var(--border); margin: 2.6rem 0; }

/* ----------------------------------------------------------------- cards */
.grid-2, .grid-3, .grid-4 {
    display: grid; gap: 12px; margin: 0.4rem 0 0.6rem 0;
}
.grid-2 > .card, .grid-3 > .card, .grid-4 > .card { height: 100%; box-sizing: border-box; }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 980px) { .grid-3, .grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; } }
.card {
    border: 1px solid var(--border); background: var(--card);
    border-radius: 12px; padding: 1.15rem 1.2rem;
    box-shadow: var(--shadow);
}
.card h3, .card h4 { margin: 0 0 0.35rem 0; font-size: 0.98rem; font-weight: 600; }
.card p { margin: 0; color: var(--muted); font-size: 0.86rem; line-height: 1.55; }
.card .num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--accent); border: 1px solid var(--accent-border);
    background: var(--accent-soft); border-radius: 6px;
    padding: 0.12rem 0.4rem; display: inline-block; margin-bottom: 0.75rem;
}
.card .ico { font-size: 1.25rem; margin-bottom: 0.6rem; display: block; }
.card .foot { margin-top: 0.8rem; font-size: 0.78rem; color: var(--faint); }

/* ----------------------------------------------------------------- stats */
.stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
@media (max-width: 980px) { .stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 560px) { .stat-grid { grid-template-columns: 1fr; } }
.stat {
    border: 1px solid var(--border); border-radius: 12px;
    background: var(--card); padding: 1rem 1.1rem; box-shadow: var(--shadow);
}
.stat-label {
    font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--muted); font-weight: 600;
}
.stat-value {
    font-size: 1.95rem; font-weight: 700; letter-spacing: -0.04em;
    line-height: 1.2; margin: 0.35rem 0 0.15rem 0; color: var(--text);
}
.stat-value small { font-size: 1rem; font-weight: 500; color: var(--muted); }
.stat-sub { font-size: 0.77rem; color: var(--faint); line-height: 1.4; }

/* --------------------------------------------------------------- badges */
.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.14rem 0.5rem; border-radius: 5px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.01em;
    border: 1px solid var(--border); color: var(--muted); background: var(--card-soft);
    white-space: nowrap;
}
.badge-keep { color: var(--accent); background: var(--accent-soft); border-color: var(--accent-border); }
.badge-good { color: var(--good); background: var(--good-soft); border-color: var(--good); }
.badge-warn { color: var(--warn); background: var(--warn-soft); border-color: var(--warn); }
.badge-danger { color: var(--danger); background: var(--danger-soft); border-color: var(--danger); }
.badge-plain { color: var(--muted); }

/* --------------------------------------------------------------- tables */
.tbl-wrap {
    border: 1px solid var(--border); border-radius: 12px;
    overflow: auto; background: var(--card); margin: 0.4rem 0 0.5rem 0;
}
table.vtbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.vtbl thead th {
    position: sticky; top: 0; z-index: 2;
    background: var(--card-soft); color: var(--muted);
    text-align: left; padding: 0.6rem 0.85rem;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    border-bottom: 1px solid var(--border); white-space: nowrap;
}
.vtbl td {
    padding: 0.58rem 0.85rem; border-top: 1px solid var(--border);
    color: var(--text); vertical-align: middle;
}
.vtbl tbody tr:first-child td { border-top: none; }
.vtbl tbody tr:hover td { background: var(--card-soft); }
.vtbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
.vtbl td.muted, .vtbl td .muted { color: var(--muted); }
.vtbl .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
.vtbl td.tight { padding-top: 0.45rem; padding-bottom: 0.45rem; }
.row-suppressed td { opacity: 0.62; }

/* probability bar */
.pbar { display: flex; align-items: center; gap: 0.55rem; min-width: 108px; }
.pbar .track {
    position: relative; height: 5px; flex: 1; min-width: 52px;
    background: var(--border); border-radius: 999px; overflow: hidden;
}
.pbar .fill {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: var(--accent); border-radius: 999px;
}
.pbar.suppress .fill { background: var(--faint); }
.pbar .val { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--muted); min-width: 36px; text-align: right; }

/* ----------------------------------------------------------------- code */
.vcode {
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 10px; overflow: auto; padding: 0.9rem 0.2rem 0.9rem 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.65;
}
.vcode pre { margin: 0; background: transparent; }
.vcode .row { display: flex; }
.vcode .ln {
    width: 2.6rem; flex: 0 0 2.6rem; text-align: right; padding-right: 0.9rem;
    color: var(--faint); user-select: none;
}
.vcode .lc { white-space: pre; color: var(--text); padding-right: 1rem; }
.vcode .row.hl { background: var(--accent-soft); box-shadow: inset 2px 0 0 var(--accent); }
.vcode .row.hl .lc { color: var(--text); }
.vcode .k { color: var(--accent); }
.vcode .s { color: var(--good); }
.vcode .c { color: var(--faint); font-style: italic; }

[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre {
    background: var(--code-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
[data-testid="stCode"] code, [data-testid="stCodeBlock"] code {
    font-family: 'JetBrains Mono', monospace !important;
}

/* -------------------------------------------------------------- widgets */
input, textarea, [contenteditable="true"] {
    background: var(--card) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
input::placeholder, textarea::placeholder { color: var(--faint) !important; }

/* React-Aria select / multiselect / number inputs */
[data-testid="stSelectbox"] [role="group"],
[data-testid="stMultiSelect"] [role="group"],
[data-testid="stNumberInput"] [role="group"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] button,
[data-testid="stMultiSelect"] button {
    background: transparent !important;
    color: var(--muted) !important;
}
[role="listbox"], [data-testid="stSelectboxVirtualDropdown"] {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
[role="option"] {
    background: transparent !important;
    color: var(--text) !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
    background: var(--card-soft) !important;
}

/* Segmented control (st.segmented_control -> stButtonGroup) */
[data-testid="stButtonGroup"] [role="radiogroup"] {
    background: var(--card-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    padding: 3px !important;
    gap: 3px !important;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    font-size: 0.84rem !important;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {
    color: inherit !important;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {
    background: var(--card) !important;
    color: var(--text) !important;
    box-shadow: var(--shadow) !important;
    font-weight: 600 !important;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: var(--card) !important; border-color: var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
[data-baseweb="select"] div, [data-baseweb="input"] input { color: var(--text) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {
    background: var(--card) !important; border: 1px solid var(--border) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--card-soft) !important; border: 1px dashed var(--border-strong) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid var(--border) !important; background: var(--card) !important;
    color: var(--text) !important; border-radius: 7px !important;
}
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span,
[data-testid="stWidgetLabel"] p, [data-testid="stSlider"] label p { color: var(--muted) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSlider"] div[data-baseweb="slider"] div { color: var(--accent) !important; }

/* buttons */
[data-testid="stButton"] button {
    border-radius: 8px; font-weight: 600; font-size: 0.87rem;
    transition: opacity 0.15s ease, transform 0.15s ease;
}
[data-testid="stButton"] button[kind="primary"] {
    background: var(--btn-bg) !important; color: var(--btn-fg) !important;
    border: 1px solid var(--btn-bg) !important;
    padding: 0.5rem 1.05rem;
}
[data-testid="stButton"] button[kind="primary"] p { color: var(--btn-fg) !important; }
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.85; }
[data-testid="stButton"] button[kind="secondary"] {
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stButton"] button[kind="secondary"] p { color: var(--text) !important; }
[data-testid="stButton"] button[kind="secondary"]:hover { border-color: var(--border-strong) !important; }
[data-testid="stDownloadButton"] button {
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stDownloadButton"] button p { color: var(--text) !important; }

/* segmented control */
[data-testid="stSegmentedControl"] button {
    background: transparent !important; color: var(--muted) !important;
    border: none !important; font-size: 0.84rem !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"] {
    background: var(--card) !important; color: var(--text) !important;
    box-shadow: var(--shadow);
}
[data-testid="stSegmentedControl"] {
    background: var(--card-soft) !important; border: 1px solid var(--border) !important;
    border-radius: 9px !important; padding: 3px !important;
}

/* expander / containers */
details[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    background: var(--card) !important;
}
details[data-testid="stExpander"] summary { color: var(--text) !important; font-size: 0.86rem; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important; border-radius: 12px !important;
    background: var(--card) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.15rem 0.2rem; }

/* tabs */
button[data-baseweb="tab"] { color: var(--muted) !important; font-size: 0.87rem !important; font-weight: 500 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background: var(--text) !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }

/* misc */
[data-testid="stMarkdownContainer"] hr { border-color: var(--border); }
[data-testid="stAlert"] { border-radius: 10px; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stCaptionContainer"] p { color: var(--faint) !important; font-size: 0.78rem; }
.vega-embed { width: 100%; }
[data-testid="stVegaLiteChart"] { overflow: visible; }

/* Analyst workspace: clear hierarchy without turning the page into a neon SOC mockup. */
.workspace-kicker {
    color: var(--faint); font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.04em; margin: -0.65rem 0 1.15rem;
}
.explore-card { min-height: 138px; display: flex; flex-direction: column; justify-content: space-between; }
.explore-card h3 { margin: 0 0 .55rem; font-size: 1.02rem; }
.explore-card p { margin: 0; color: var(--muted); font-size: .86rem; line-height: 1.6; }
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(110deg, #155eef 0%, #0f7eea 48%, #0e9f9a 100%) !important;
    border: 1px solid rgba(21,94,239,.18) !important;
    box-shadow: 0 8px 18px rgba(21,94,239,.16) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.04); transform: translateY(-1px); }
[data-testid="stButton"] button { transition: transform .16s ease, filter .16s ease, border-color .16s ease; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important; border-radius: 12px !important;
    background: var(--card) !important; box-shadow: var(--shadow);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
}
[data-testid="stDataFrame"] [role="gridcell"] { font-size: 0.82rem; }
[data-testid="stDataFrame"] [role="columnheader"] { font-size: 0.72rem; font-weight: 650; }
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="slider"]:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
@media (max-width: 760px) {
    .workspace-kicker { line-height: 1.65; }
    .topbar-line { margin-bottom: 1.45rem; }
    .page-head { margin-bottom: 1.15rem; }
}

/* footer */
.footer {
    border-top: 1px solid var(--border); margin-top: 3.5rem; padding-top: 1.1rem;
    display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
    color: var(--faint); font-size: 0.75rem;
}
.footer .mono { font-size: 0.72rem; }

/* page headers */
.page-head { background: linear-gradient(115deg, var(--card) 0%, var(--accent-soft) 62%, var(--good-soft) 100%); border: 1px solid var(--border); border-radius: 16px; margin: .6rem 0 1.6rem; overflow: hidden; padding: 1.6rem 1.7rem; position: relative; }
.page-head::after { border: 1px solid var(--accent-border); border-radius: 50%; content: ""; height: 180px; position: absolute; right: -85px; top: -100px; width: 180px; }
.page-head h1 { font-size: clamp(2rem, 4vw, 2.75rem); font-weight: 800; letter-spacing: -0.045em; margin: 0.35rem 0 0.5rem 0; position: relative; z-index: 1; }
.page-head p { color: var(--muted); max-width: 760px; line-height: 1.6; margin: 0; font-size: 0.97rem; position: relative; z-index: 1; }

/* inline key-value */
.kv { display: flex; justify-content: space-between; gap: 1rem; padding: 0.42rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--muted); }
.kv .v { color: var(--text); font-weight: 500; text-align: right; }

/* confusion matrix */
.cm { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px; align-items: stretch; }
.cm .cell {
    border: 1px solid var(--border); border-radius: 10px; padding: 1rem;
    background: var(--card); text-align: center;
}
.cm .cell .v { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; }
.cm .cell .l { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; }
.cm .cell.good { background: var(--good-soft); border-color: var(--good); }
.cm .cell.bad { background: var(--danger-soft); border-color: var(--danger); }
.cm .head { display: flex; align-items: center; justify-content: center; font-size: 0.72rem; color: var(--muted); text-align: center; }
.cm .side { display: flex; align-items: center; justify-content: flex-end; padding-right: 0.5rem; font-size: 0.72rem; color: var(--muted); text-align: right; }
@media (max-width: 640px) { .cm { grid-template-columns: 70px 1fr 1fr; } .cm .cell { padding: 0.6rem; } .cm .cell .v { font-size: 1.2rem; } }
"""
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def current_theme() -> str:
    state = st.session_state.get("safescan_theme", {})
    selected = state.get("theme") if isinstance(state, dict) else None
    if selected in PALETTES:
        return selected
    return st.context.theme.type if st.context.theme.type in PALETTES else "light"


def colors() -> dict:
    return PALETTES[current_theme()]


def init_theme() -> None:
    st.markdown(f"<style>{CSS.substitute(**colors())}</style>", unsafe_allow_html=True)


def _on_theme_change() -> None:
    """Register the theme state so Streamlit can apply the component update."""


def theme_toggle() -> None:
    selected = current_theme()
    system_theme = st.context.theme.type if st.context.theme.type in PALETTES else "light"
    THEME_TOGGLE(
        data={"theme": selected, "systemTheme": system_theme},
        default={"theme": selected},
        key="safescan_theme",
        on_theme_change=_on_theme_change,
    )


# ---------------------------------------------------------------------------
# Cached data / model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_model()


@st.cache_data(show_spinner=False)
def get_alerts() -> pd.DataFrame:
    return pd.read_csv(ALERTS_CSV)


@st.cache_data(show_spinner=False)
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data(show_spinner=False)
def analyze_bytes(filename: str, content: bytes) -> pd.DataFrame:
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."} or Path(safe_name).suffix.lower() != ".py":
        raise ValueError("Provide a Python filename ending in .py.")
    if len(content) > 1_000_000:
        raise ValueError("Source is limited to 1 MB per scan.")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / safe_name
        target.write_bytes(content)
        return _analyze(target, bundle=get_bundle())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def format_snippet(code: str) -> str:
    """Remove Bandit's line-number prefixes so the snippet reads like code."""
    lines = []
    for line in str(code).splitlines():
        lines.append(re.sub(r"^\s*\d+\s?", "", line))
    return "\n".join(lines)


def pretty_feature(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "").replace("text__", "text: ")
    return name.replace("_", " ")


FEATURE_HELP = {
    "ctx_has_request": "User input is read before the sink.",
    "ctx_taint": "The backward slice reaches a known taint source.",
    "ctx_sanitizer": "A sanitizer appears in the data-flow path.",
    "ctx_validation": "An if-guard validates the value before the sink.",
    "ctx_last_literal": "The last assignment to the sink value is a constant.",
    "ctx_n_assign_sink": "How many times the sink value is reassigned.",
    "ctx_depth": "How deep the backward slice went.",
    "ctx_sink_literal": "The sink argument itself is a constant.",
    "ctx_sink_name": "The sink argument is a plain variable.",
    "ctx_has_try": "The sink is inside a try block.",
    "ctx_func_len": "Lines in the enclosing function.",
    "ctx_string_built": "The sink value is built with an f-string, +, or .format().",
    "ctx_shell_true": "The code uses shell=True.",
    "ctx_list_args": "subprocess is called with a list argument.",
    "ctx_defusedxml": "The function uses defusedxml.",
}


def explain_alert(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if int(row.get("ctx_has_request", 0)) == 1:
        reasons.append("User input is read in the same function, before the sink.")
    if int(row.get("ctx_taint", 0)) == 1:
        reasons.append("The flagged value can be traced back to user input.")
    if int(row.get("ctx_validation", 0)) == 1:
        reasons.append("A validation guard runs before the sink, which is common in false positives.")
    if int(row.get("ctx_sanitizer", 0)) == 1:
        reasons.append("A sanitizer appears in the data-flow path.")
    if int(row.get("ctx_sink_literal", 0)) == 1:
        reasons.append("The sink argument is a constant, so there is no user-controlled data.")
    if int(row.get("ctx_last_literal", 0)) == 1:
        reasons.append("The last assignment to the sink value is a constant.")
    if int(row.get("ctx_string_built", 0)) == 1:
        reasons.append("The sink value is assembled from strings, a pattern that often carries input.")
    if int(row.get("ctx_shell_true", 0)) == 1:
        reasons.append("The call uses shell=True, which raises the risk of command injection.")
    if int(row.get("ctx_list_args", 0)) == 1:
        reasons.append("subprocess receives a list argument instead of a shell string.")
    if int(row.get("ctx_has_try", 0)) == 1:
        reasons.append("The sink sits inside a try block; errors are handled locally.")
    if not reasons:
        reasons.append(
            "No individual context signal fired; the score relies on the rule, "
            "code text, and their interaction."
        )
    reasons.append(
        "The final score combines these signals with the Bandit rule and the flagged code text."
    )
    return reasons


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def stylize(chart: alt.Chart, height: int = 300) -> alt.Chart:
    c = colors()
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=c["muted"],
            titleColor=c["muted"],
            gridColor=c["border"],
            domainColor=c["border"],
            tickColor=c["border"],
            labelFontSize=10.5,
            titleFontSize=10.5,
            titleFontWeight=500,
            gridOpacity=0.65,
        )
        .configure_legend(
            labelColor=c["muted"],
            titleColor=c["muted"],
            orient="top",
            direction="horizontal",
            labelFontSize=10.5,
            titleFontSize=10.5,
            symbolSize=70,
        )
        .configure_title(
            color=c["text"], fontSize=12.5, fontWeight=600, anchor="start"
        )
        .configure_range(
            category=[c["accent"], c["good"], c["warn"], c["danger"], c["faint"]]
        )
    )


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------
def stat_grid(items: list[tuple[str, str, str]]) -> None:
    cells = []
    for label, value, sub in items:
        cells.append(
            f'<div class="stat"><div class="stat-label">{esc(label)}</div>'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-sub">{esc(sub)}</div></div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def stat_grid_html(items: list[tuple[str, str, str]]) -> str:
    cells = []
    for label, value, sub in items:
        cells.append(
            f'<div class="stat"><div class="stat-label">{esc(label)}</div>'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-sub">{esc(sub)}</div></div>'
        )
    return f'<div class="stat-grid">{"".join(cells)}</div>'


def section(eyebrow: str, title: str, desc: str = "") -> None:
    st.markdown(
        f'<div class="section"><div class="eyebrow">{esc(eyebrow)}</div>'
        f"<h2>{esc(title)}</h2><p>{esc(desc)}</p></div>",
        unsafe_allow_html=True,
    )


def card(title: str, body: str, *, num: str = "", foot: str = "", icon: str = "") -> str:
    return (
        '<div class="card">'
        + (f'<div class="num">{esc(num)}</div>' if num else "")
        + (f'<span class="ico">{icon}</span>' if icon else "")
        + f"<h3>{esc(title)}</h3><p>{body}</p>"
        + (f'<div class="foot">{foot}</div>' if foot else "")
        + "</div>"
    )


def cards_html(cards: list[str], columns: int = 3) -> str:
    return f'<div class="grid-{columns}">{"".join(cards)}</div>'


_PY_KEYWORDS = (
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield",
)


def _syntax(line: str) -> str:
    """Tiny, dependency-free syntax highlighter for short snippets."""
    token_re = re.compile(r"(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|#[^\n]*)")
    keyword_re = re.compile(rf"\b({'|'.join(_PY_KEYWORDS)})\b")
    parts: list[str] = []
    for token in token_re.split(line):
        if not token:
            continue
        if token.startswith("#"):
            parts.append(f'<span class="c">{html.escape(token, quote=False)}</span>')
        elif token[0] in "\"'":
            parts.append(f'<span class="s">{html.escape(token, quote=False)}</span>')
        else:
            escaped = html.escape(token, quote=False)
            parts.append(
                keyword_re.sub(lambda m: f'<span class="k">{m.group(0)}</span>', escaped)
            )
    return "".join(parts)


def code_block(code: str, highlight: int | None = None) -> str:
    rows = []
    for number, line in enumerate(code.splitlines(), start=1):
        cls = " hl" if highlight == number else ""
        rows.append(
            f'<div class="row{cls}"><span class="ln">{number}</span>'
            f'<span class="lc">{_syntax(line) or " "}</span></div>'
        )
    return f'<div class="vcode"><pre>{"".join(rows)}</pre></div>'


def table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_height: int | None = None,
    numeric: set[int] | None = None,
    row_classes: list[str] | None = None,
) -> str:
    numeric = numeric or set()
    style = f' style="max-height:{max_height}px"' if max_height else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        cls = row_classes[index] if row_classes else ""
        cells = "".join(
            f'<td class="{"num" if col in numeric else ""}">{cell}</td>'
            for col, cell in enumerate(row)
        )
        body.append(f'<tr class="{cls}">{cells}</tr>')
    return (
        f'<div class="tbl-wrap"{style}><table class="vtbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def prob_cell(probability: float, kept: bool) -> str:
    cls = "" if kept else " suppress"
    return (
        f'<div class="pbar{cls}"><div class="track">'
        f'<div class="fill" style="width:{max(0, min(100, probability * 100)):.1f}%"></div>'
        f'</div><span class="val">{probability:.2f}</span></div>'
    )


def pill(text: str, dot: bool = False) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="pill">{marker}{esc(text)}</span>'


def footer() -> None:
    st.markdown(
        '<div class="footer">'
        "<span>SafeScan · Context-aware static analysis · Bandit + scikit-learn</span>"
        '<span class="mono">Local source review</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = [
    ("Overview", "overview", "views/overview.py"),
    ("Triage", "triage", "views/triage.py"),
    ("Model Insights", "model-insights", "views/model_insights.py"),
    ("Dataset & Method", "dataset-method", "views/dataset_method.py"),
]


def topbar(active: str) -> None:
    with st.container(key="topnav"):
        columns = st.columns([3.2, 1.05, 0.95, 1.35, 1.5, 1.0], vertical_alignment="center")
        columns[0].markdown(
            '<div class="brand"><span class="brand-mark" aria-hidden="true">'
            '<svg viewBox="0 0 28 28" role="img"><path d="M14 2.5 24 6.6v6.1c0 6.3-3.9 10.7-10 12.8-6.1-2.1-10-6.5-10-12.8V6.6L14 2.5Z" fill="none" stroke="currentColor" stroke-width="2.2"/>'
            '<path d="m9.2 14.1 3.1 3.1 6.4-6.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></span>'
            '<span class="brand-name">SafeScan</span><span class="brand-label">Security triage</span></div>',
            unsafe_allow_html=True,
        )
        for column, (label, _key, path) in zip(columns[1:5], PAGES):
            with column:
                if st.button(
                    label,
                    key=f"nav_{_key}",
                    type="primary" if label == active else "tertiary",
                    width="stretch",
                ):
                    if label != active:
                        st.switch_page(path)
        with columns[5]:
            theme_toggle()
    st.markdown('<div class="topbar-line"></div>', unsafe_allow_html=True)


def page_head(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{esc(eyebrow)}</div>'
        f"<h1>{title}</h1><p>{esc(description)}</p></div>",
        unsafe_allow_html=True,
    )
