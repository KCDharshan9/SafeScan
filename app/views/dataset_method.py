"""Dataset and method page."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as ui  # noqa: E402

ui.topbar("Dataset & Method")
ui.page_head(
    "Method",
    "Dataset & method",
    "A transparent description of the data, the labelling rule, the engineered "
    "features and the validation protocol, the parts that decide whether the "
    "reported numbers mean anything.",
)

alerts = ui.get_alerts()
total = len(alerts)
true_positive = int((alerts["label"] == 1).sum())
false_positive = total - true_positive

ui.stat_grid(
    [
        ("Benchmark test cases", "1,230", "452 vulnerable · 778 safe"),
        ("Bandit alerts", f"{total}", "one row per finding"),
        ("True positives", f"{true_positive}", f"{100 * true_positive / total:.0f}% of alerts"),
        ("False positives", f"{false_positive}", f"{100 * false_positive / total:.0f}% of alerts"),
    ]
)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
ui.section(
    "Data source",
    "OWASP Benchmark for Python v0.1",
    "A deliberately vulnerable test suite with known ground truth, built to measure "
    "how well security tools find specific vulnerability classes.",
)

source_col, fact_col = st.columns([1.5, 1], gap="large")
with source_col:
    st.markdown(
        '<div class="card" style="height:100%">'
        '<div class="eyebrow">Dataset card</div>'
        '<div class="kv"><span class="k">Source</span><span class="v">OWASP-Benchmark/BenchmarkPython</span></div>'
        '<div class="kv"><span class="k">Version</span><span class="v">0.1</span></div>'
        '<div class="kv"><span class="k">Language</span><span class="v">Python</span></div>'
        '<div class="kv"><span class="k">Test cases</span><span class="v">1,230</span></div>'
        '<div class="kv"><span class="k">Vulnerable / safe</span><span class="v">452 / 778</span></div>'
        '<div class="kv"><span class="k">Vulnerability categories</span><span class="v">6</span></div>'
        '<div class="kv"><span class="k">License</span><span class="v">GPL-3.0</span></div>'
        '<p style="margin-top:.8rem">Each test case is a small Flask-style Python file. '
        "The benchmark supplies an expected-results CSV with the true category, a "
        "real-vs-safe flag and a CWE ID.</p></div>",
        unsafe_allow_html=True,
    )
with fact_col:
    st.markdown(
        '<div class="card" style="height:100%">'
        '<div class="eyebrow">Why this dataset</div>'
        "<h3>Labels without manual annotation</h3>"
        "<p>The benchmark knows which file contains which vulnerability. That lets us "
        "label every Bandit alert automatically and measure precision and recall, "
        "which is impossible on an unlabelled production repository.</p>"
        '<p style="margin-top:.6rem">The trade-off is size and artificiality: 1,230 '
        "generated cases are enough to validate the method but not for a "
        "production claim.</p></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Labelling rule
# ---------------------------------------------------------------------------
ui.section(
    "Labels",
    "When does an alert count as a true positive?",
    "The rule mirrors how SAST benchmarks score tools: an alert only counts if the "
    "rule belongs to the vulnerability that was actually injected.",
)

rule_col_1, rule_col_2, rule_col_3 = st.columns(3, gap="medium")
with rule_col_1:
    st.markdown(
        '<div class="card" style="height:100%"><div class="num">1</div>'
        "<h3>The file is vulnerable</h3>"
        "<p>The benchmark's expected-results file marks the test case as real, and "
        "gives its category (SQLi, command injection, XXE, …).</p></div>",
        unsafe_allow_html=True,
    )
with rule_col_2:
    st.markdown(
        '<div class="card" style="height:100%"><div class="num">2</div>'
        "<h3>The rule matches the category</h3>"
        "<p>Bandit's rule ID must be mapped to that category, e.g. B301 for "
        "deserialization or B608 for SQL injection.</p></div>",
        unsafe_allow_html=True,
    )
with rule_col_3:
    st.markdown(
        '<div class="card" style="height:100%"><div class="num">3</div>'
        "<h3>Everything else is a false positive</h3>"
        "<p>Generic import warnings such as B403 (import pickle) or B406 (import "
        "xml.sax) do not prove the injected vulnerability, so they are negative.</p></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Class balance + categories
# ---------------------------------------------------------------------------
ui.section(
    "Distribution",
    "What the model is actually learning from",
    "The classes are imbalanced (40% positive) and concentrated in XXE and code "
    "injection. Accuracy alone would be misleading, which is why F1, ROC-AUC and "
    "PR-AUC are reported together.",
)

balance_col, category_col = st.columns([0.75, 1.55], gap="large")
with balance_col:
    st.markdown(
        f'<div class="card" style="height:100%">'
        '<div class="eyebrow">Class balance</div>'
        f'<div class="stat-value" style="font-size:2rem">{true_positive} <small>/</small> {total}</div>'
        '<div class="stat-sub" style="margin-bottom:.8rem">alerts are real bugs</div>'
        '<div style="display:flex;height:10px;border-radius:999px;overflow:hidden;'
        'background:var(--border);margin:.9rem 0 .6rem 0">'
        f'<div style="width:{100 * true_positive / total:.1f}%;background:var(--good)"></div>'
        f'<div style="width:{100 * false_positive / total:.1f}%;background:var(--faint)"></div>'
        "</div>"
        '<div style="display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted)">'
        f"<span>● True positive · {100 * true_positive / total:.0f}%</span>"
        f"<span>● False positive · {100 * false_positive / total:.0f}%</span></div>"
        '<p style="margin-top:.9rem">Random Forest uses balanced class weights so the '
        "minority class is not ignored.</p></div>",
        unsafe_allow_html=True,
    )
with category_col:
    with st.container(border=True):
        st.markdown(
            '<div class="eyebrow" style="margin-bottom:.1rem">By family</div>'
            '<h3 style="font-size:1rem;margin:.1rem 0 .6rem 0">Alerts per vulnerability category</h3>',
            unsafe_allow_html=True,
        )
        frame = alerts.copy()
        frame["Class"] = frame["label"].map({1: "True positive", 0: "False positive"})
        grouped = (
            frame.groupby(["gt_category", "Class"]).size().reset_index(name="Alerts")
        )
        totals = grouped.groupby("gt_category")["Alerts"].sum()
        grouped["total"] = grouped["gt_category"].map(totals)
        bars = (
            alt.Chart(grouped)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("gt_category:N", sort="-x", title=None),
                x=alt.X("Alerts:Q", title="Alerts", axis=alt.Axis(tickMinStep=10)),
                color=alt.Color(
                    "Class:N",
                    scale=alt.Scale(
                        domain=["True positive", "False positive"],
                        range=[ui.colors()["good"], ui.colors()["border_strong"]],
                    ),
                    legend=alt.Legend(title=None),
                ),
                order=alt.Order("Class:N"),
                tooltip=["gt_category:N", "Class:N", "Alerts:Q"],
            )
        )
        st.altair_chart(ui.stylize(bars, 250), width="stretch")
        st.caption("XXE has the most alerts, but many are generic import warnings.")

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
ui.section(
    "Preprocessing",
    "From code to a feature vector",
    "Bandit gives a rule, severity, confidence and the flagged line. On its own that "
    "is not enough: the label often depends on upstream context, so the extractor "
    "walks the AST and adds 15 signals.",
)

feature_cards = [
    ui.card(
        "Metadata (categorical)",
        "Rule ID, severity and confidence, one-hot encoded. This lets the model learn "
        "that, for example, B608 is usually a real SQL-injection finding.",
        num="A",
        foot="3 columns · one-hot",
    ),
    ui.card(
        "AST context (numeric)",
        "Taint, request source, validation, sanitizer, literals, slice depth, "
        "reassignment count, shell=True, list arguments, defusedxml and more.",
        num="B",
        foot="15 binary / counting signals",
    ),
    ui.card(
        "Code text (TF-IDF)",
        "The flagged line is normalised (numbers and identifiers replaced) and "
        "vectorised with word and character n-grams.",
        num="C",
        foot="word + char n-grams",
    ),
]
st.markdown(ui.cards_html(feature_cards, columns=3), unsafe_allow_html=True)

with st.container(border=True):
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown(
            '<div class="eyebrow">Context signals</div>'
            '<h3 style="font-size:1rem;margin:.1rem 0 .5rem 0">The 15 AST features</h3>'
            '<p class="muted" style="font-size:.83rem;line-height:1.55">Binary flags are '
            "1 when the signal is present in the backward slice from the sink; counts "
            "capture how far the extractor had to walk.</p>",
            unsafe_allow_html=True,
        )
    with right:
        items = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:1rem;'
            f'padding:.36rem 0;border-bottom:1px solid var(--border);font-size:.8rem">'
            f'<span class="mono" style="color:var(--text)">{ui.esc(name)}</span>'
            f'<span class="muted" style="text-align:right">{ui.esc(text)}</span></div>'
            for name, text in ui.FEATURE_HELP.items()
        )
        st.markdown(items, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Validation protocol
# ---------------------------------------------------------------------------
ui.section(
    "Validation",
    "How the model is tested honestly",
    "The benchmark generates near-duplicate variants, so a plain random split can be "
    "too optimistic. Three different protocols are reported to keep the claim honest.",
)

protocol = [
    (
        "01",
        "Stratified 5-fold CV",
        "Random folds preserve the class ratio. This is the headline comparison table "
        "for all candidate models.",
    ),
    (
        "02",
        "Grouped by family",
        "Whole variant families are held out together, so the model cannot memorise a "
        "near-duplicate test case from the training fold.",
    ),
    (
        "03",
        "Leave one family out",
        "Train on every family except one; test on the unseen family. This exposes "
        "where the model relies on family-specific text.",
    ),
    (
        "04",
        "20% holdout",
        "A stratified holdout is kept aside from all tuning and used once for the final "
        "confusion matrix and threshold sweep.",
    ),
]
columns = st.columns(4, gap="medium")
for column, (num, title, body) in zip(columns, protocol):
    with column:
        st.markdown(ui.card(title, body, num=num), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
ui.section(
    "Pipeline",
    "End-to-end, reproducible",
    "Every stage is a script in src/, so the dataset, the figures and the metrics can "
    "be rebuilt from the raw benchmark.",
)

stages = [
    ("collect.py", "Bandit to labelled alerts.csv", "325 alerts"),
    ("features.py", "metadata + AST context + text", "feature rows"),
    ("train.py", "pipelines, CV, tuning, save model", "rf_pipeline.joblib"),
    ("predict.py", "score any Python file", "P(true positive)"),
    ("streamlit_app.py", "overview, triage, insights, method", "application pages"),
]
stage_html = "".join(
    '<div style="display:flex;align-items:center;gap:.7rem;border:1px solid var(--border);'
    'border-radius:10px;padding:.75rem .9rem;background:var(--card)">'
    f'<span class="mono" style="font-size:.75rem;color:var(--accent)">{ui.esc(name)}</span>'
    f'<span style="font-size:.78rem;color:var(--muted)">{ui.esc(desc)}</span>'
    f'<span class="badge" style="margin-left:auto">{ui.esc(out)}</span></div>'
    for name, desc, out in stages
)
st.markdown(
    f'<div style="display:grid;grid-template-columns:1fr;gap:8px">{stage_html}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Limitations + references
# ---------------------------------------------------------------------------
ui.section("Limitations", "What this project does not do")
limitations = [
    ui.card(
        "Synthetic benchmark",
        "The OWASP Benchmark is generated, not scraped from real projects. Real code "
        "has different noise, style and rule distributions.",
        num="01",
    ),
    ui.card(
        "Single analyzer",
        "Only 17 Bandit rules appear. Adding Semgrep or CodeQL means remapping rules "
        "and retraining, not just plugging in a new output file.",
        num="02",
    ),
    ui.card(
        "Shallow taint model",
        "Direct assignments and simple guards are tracked. Objects, containers and "
        "cross-function flows are out of scope.",
        num="03",
    ),
    ui.card(
        "Not a gatekeeper",
        "The output is a ranking aid. A suppressed alert is not proven safe; it is "
        "only less likely to be the injected bug.",
        num="04",
    ),
]
st.markdown(ui.cards_html(limitations, columns=4), unsafe_allow_html=True)

ui.section("References", "Sources and documentation")
references = [
    "OWASP Benchmark for Python v0.1: github.com/OWASP-Benchmark/BenchmarkPython (GPL-3.0).",
    "Bandit 1.9: PyCQA static security analyzer for Python. bandit.readthedocs.io.",
    "scikit-learn 1.7: Pedregosa et al., “Scikit-learn: Machine Learning in Python”, JMLR 12 (2011).",
    "Breiman, L. (2001). “Random Forests.” Machine Learning, 45(1), 5 to 32.",
    "OWASP Foundation. “Source Code Analysis Tools” and the OWASP Top 10.",
    "Project source, figures and metrics: the false-positive-reducer repository.",
]
st.markdown(
    '<ol style="font-size:.85rem;line-height:1.8;color:var(--muted);padding-left:1.2rem">'
    + "".join(f"<li>{ui.esc(item)}</li>" for item in references)
    + "</ol>",
    unsafe_allow_html=True,
)

# Quick in-app stats sanity check.
with st.expander("Dataset sanity check"):
    st.markdown(
        f'<div class="kv"><span class="k">Rows in alerts.csv</span><span class="v">{total}</span></div>'
        f'<div class="kv"><span class="k">Positive labels</span><span class="v">{true_positive}</span></div>'
        f'<div class="kv"><span class="k">Negative labels</span><span class="v">{false_positive}</span></div>'
        f'<div class="kv"><span class="k">Unique rules</span><span class="v">{alerts["rule"].nunique()}</span></div>'
        f'<div class="kv"><span class="k">Unique test cases</span><span class="v">{alerts["test_name"].nunique()}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These counts should match the README. If they do not, rerun src/collect.py "
        "before trusting the app."
    )

ui.footer()
