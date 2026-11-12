"""SafeScan overview page."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as ui  # noqa: E402


ui.topbar("Overview")

with st.container(border=True, key="hero"):
    st.markdown(
        '<span class="pill"><span class="dot"></span>Context-aware security triage</span>'
        '<h1>See the signal.<br><span class="grad">Prioritise the risk.</span></h1>'
        '<p class="lede">SafeScan adds a contextual machine learning layer to Python static analysis. '
        'It ranks findings by the likelihood they deserve attention, then gives reviewers the evidence they need to make the final call.</p>',
        unsafe_allow_html=True,
    )
    action_left, action_right, _ = st.columns([1.3, 1.5, 2.5], vertical_alignment="center")
    with action_left:
        if st.button("Open workspace", type="primary", width="stretch", key="hero_triage"):
            st.switch_page("views/triage.py")
    with action_right:
        if st.button("Explore model evidence", width="stretch", key="hero_model"):
            st.switch_page("views/model_insights.py")
    st.markdown(
        '<div class="meta-row"><span>Python source scanning</span><span>Bandit detection</span>'
        '<span>Contextual ranking</span><span>Local review workflow</span></div>',
        unsafe_allow_html=True,
    )

ui.stat_grid([
    ("Alerts analysed", "325", "labelled findings used for evaluation"),
    ("Recorded precision gain", "80<small>%</small>", "false positives hidden at the saved threshold"),
    ("True positives retained", "96<small>%</small>", "at the saved threshold on evaluation data"),
    ("Ranking quality", "0.97", "cross-validated ROC-AUC"),
])

ui.section(
    "How SafeScan works",
    "From a raw alert to a focused review queue",
    "The scanner detects potential issues. SafeScan enriches them with context and helps a human reviewer decide what to investigate first.",
)
steps = [
    ui.card("Scan source", "Bandit checks Python code and returns findings with a rule, severity, confidence, and flagged line.", num="01", foot="Source stays in the local session"),
    ui.card("Read context", "The pipeline combines rule metadata, code text, and AST context such as validation, sources, and sinks.", num="02", foot="Code-aware feature extraction"),
    ui.card("Rank findings", "A trained Random Forest produces a score for every finding without making a final security decision.", num="03", foot="Probability-based prioritisation"),
    ui.card("Review with evidence", "Filter the queue, inspect the source, and use the score as one input to a human review decision.", num="04", foot="Clear handoff to the analyst"),
]
st.markdown(ui.cards_html(steps, columns=4), unsafe_allow_html=True)

ui.section(
    "Why context matters",
    "The same security rule can describe two very different situations",
    "The alert line alone does not always reveal whether user-controlled data reaches a dangerous operation. SafeScan looks at the nearby code to make a better ranking decision.",
)
context_left, context_right = st.columns(2, gap="medium")
with context_left:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Higher-priority pattern</div><h3>Input reaches a sensitive operation</h3>', unsafe_allow_html=True)
        st.markdown(ui.code_block('value = request.args["expression"]\nexec(value)', highlight=2), unsafe_allow_html=True)
        st.caption("A request-derived value reaches the execution sink. This is a strong reason to place the finding near the top of the review queue.")
with context_right:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Lower-priority pattern</div><h3>A guard is visible before the operation</h3>', unsafe_allow_html=True)
        st.markdown(ui.code_block('value = get_setting()\nif not value.startswith("literal:"):\n    raise ValueError("invalid value")\nexec(value)', highlight=4), unsafe_allow_html=True)
        st.caption("The code still needs review, but a visible validation path is useful evidence for ranking it lower than an unguarded call.")

ui.section(
    "Operating point",
    "Set the queue size for the review you need",
    "The threshold changes the balance between retaining possible risks and removing likely noise. SafeScan keeps this decision visible and adjustable instead of burying it in model configuration.",
)
chart_col, guide_col = st.columns([1.6, 1], gap="medium")
with chart_col:
    with st.container(border=True):
        sweep = ui.load_report("threshold_random_forest_all.csv").rename(columns={"fp_suppressed_pct": "Likely noise removed", "tp_kept_pct": "Possible risks retained"})
        long = sweep.melt(id_vars="threshold", value_vars=["Likely noise removed", "Possible risks retained"], var_name="Measure", value_name="Percent")
        line = alt.Chart(long).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=48, filled=True)).encode(
            x=alt.X("threshold:Q", title="Review threshold", scale=alt.Scale(domain=[0.05, 0.95])),
            y=alt.Y("Percent:Q", title="Percent", scale=alt.Scale(domain=[55, 102])),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Likely noise removed", "Possible risks retained"], range=[ui.colors()["accent"], ui.colors()["good"]]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("threshold:Q", title="Threshold", format=".2f"), "Measure:N", alt.Tooltip("Percent:Q", format=".1f")],
        )
        marker = alt.Chart(alt.Data(values=[{"threshold": 0.30}])).mark_rule(color=ui.colors()["faint"], strokeDash=[4, 4]).encode(x="threshold:Q")
        st.altair_chart(ui.stylize(line + marker, 280), width="stretch")
        st.caption("The marker shows the saved default threshold of 0.30, measured on held-out evaluation outputs.")
with guide_col:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Review guide</div><h3>Choose deliberately</h3>', unsafe_allow_html=True)
        st.markdown("**Lower threshold**  ")
        st.caption("Keep more findings when missing a potential issue is costly.")
        st.markdown("**Higher threshold**  ")
        st.caption("Reduce queue volume when an analyst needs a tighter starting point.")
        st.markdown("**Always inspect evidence**  ")
        st.caption("The score helps order work. It does not establish exploitability or safety.")

ui.section(
    "Built for review",
    "Simple controls. Useful evidence. No hidden decisions.",
    "Each part of the workflow has one job, so a reviewer can move from source code to a reasoned next action without hunting through a dense dashboard.",
)
capabilities = [
    ui.card("Local source input", "Upload one Python file or paste a focused snippet. Analysis runs only when you request it.", foot="No automatic rescan while editing"),
    ui.card("A readable queue", "Sort, filter, and search by finding. The top result opens directly into an evidence panel.", foot="Built for focused review"),
    ui.card("Transparent context", "See the flagged line, Bandit metadata, model score, and detected context signals together.", foot="No black-box handoff"),
]
st.markdown(ui.cards_html(capabilities, columns=3), unsafe_allow_html=True)

ui.section("Explore", "Understand the model before you trust it", "SafeScan exposes its evaluation and method so the behaviour behind the interface is inspectable.")
pages = [
    ("Triage workspace", "Scan real Python source, set a review threshold, and investigate individual findings with their context.", "Open workspace", "views/triage.py", "triage_card"),
    ("Model evidence", "Compare models, inspect ranking curves, review threshold trade-offs, and see validation limits.", "Open evidence", "views/model_insights.py", "model_card"),
    ("Data and method", "See how the training data, features, labels, and evaluation checks were constructed.", "Open method", "views/dataset_method.py", "dataset_card"),
]
columns = st.columns(3, gap="medium")
for column, (title, body, label, path, key) in zip(columns, pages):
    with column:
        with st.container(border=True):
            st.markdown(f'<div class="explore-card"><h3>{ui.esc(title)}</h3><p>{ui.esc(body)}</p></div>', unsafe_allow_html=True)
            if st.button(label, key=key, width="stretch"):
                st.switch_page(path)

ui.footer()
