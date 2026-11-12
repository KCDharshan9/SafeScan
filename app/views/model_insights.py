"""Model insights page."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as ui  # noqa: E402

ui.topbar("Model Insights")
ui.page_head(
    "Evaluation",
    "Model insights",
    "Everything below comes from the saved experiment outputs in reports/: stratified "
    "cross-validation, a 20% holdout, grouped validation on unseen vulnerability "
    "families, feature ablation and the operating-point sweep.",
)

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
ui.stat_grid(
    [
        ("CV accuracy", "0.908", "5-fold stratified · 325 alerts"),
        ("CV F1", "0.886", "Random Forest, all features"),
        ("ROC-AUC", "0.966", "ranking quality, CV"),
        ("Holdout accuracy", "0.892", "20% never seen during tuning"),
    ]
)

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
ui.section(
    "Comparison",
    "Five models and two baselines",
    "The Random Forest wins on F1 and PR-AUC. Logistic regression stays close and is "
    "much easier to explain, so it is a useful sanity check rather than a discarded "
    "model.",
)

comparison = ui.load_report("model_comparison.csv").sort_values("f1", ascending=False)
chart_col, table_col = st.columns([1.15, 1.55], gap="large")
with chart_col:
    with st.container(border=True):
        st.markdown(
            '<div class="eyebrow" style="margin-bottom:.1rem">Cross-validation</div>'
            '<h3 style="font-size:1rem;margin:.1rem 0 .7rem 0">F1 by model</h3>',
            unsafe_allow_html=True,
        )
        plot = comparison.copy()
        short_names = {
            "Random Forest (all)": "Random Forest",
            "Decision Tree (all)": "Decision Tree",
            "Logistic Regression (all)": "Logistic (all)",
            "Logistic Regression (meta + context)": "Logistic (meta+ctx)",
            "Naive Bayes (text only)": "Naive Bayes",
            "Logistic Regression (meta + text)": "Logistic (meta+text)",
            "Rule-prior baseline (shallow tree)": "Rule prior",
            "Majority baseline": "Majority",
        }
        plot["short"] = plot["model"].map(short_names).fillna(plot["model"])
        bars = (
            alt.Chart(plot)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                y=alt.Y("short:N", sort="-x", title=None),
                x=alt.X("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(
                    alt.datum.short == "Random Forest",
                    alt.value(ui.colors()["accent"]),
                    alt.value(ui.colors()["border_strong"]),
                ),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("f1:Q", title="F1", format=".3f"),
                    alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".3f"),
                ],
            )
        )
        st.altair_chart(ui.stylize(bars, 285), width="stretch")
        st.caption("The majority baseline has F1 = 0 because it never predicts a bug.")

with table_col:
    rows = []
    for _, row in comparison.iterrows():
        highlight = (
            '<span class="badge badge-keep">best</span>'
            if row["model"] == "Random Forest (all)"
            else ""
        )
        rows.append(
            [
                f'<span style="font-weight:500">{ui.esc(row["model"])}</span> {highlight}',
                f'{row["accuracy"]:.3f}',
                f'{row["precision"]:.3f}',
                f'{row["recall"]:.3f}',
                f'<b>{row["f1"]:.3f}</b>' if row["model"] == "Random Forest (all)"
                else f'{row["f1"]:.3f}',
                f'{row["roc_auc"]:.3f}',
                f'{row["pr_auc"]:.3f}',
            ]
        )
    st.markdown(
        ui.table_html(
            ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC", "PR"],
            rows,
            numeric={1, 2, 3, 4, 5, 6},
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Metrics use the standard 0.50 cut-off for the CV table; the application "
        "explores a different operating point."
    )

# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------
ui.section(
    "Ranking quality",
    "ROC and precision and recall curves",
    "Computed from out-of-fold predictions, so every alert is scored by a model that "
    "did not see it during training.",
)

oof = ui.load_report("oof_probabilities.csv")
curve_models = {
    "Random Forest": "p::Random Forest (all)",
    "Logistic Regression": "p::Logistic Regression (all)",
    "Naive Bayes (text)": "p::Naive Bayes (text only)",
}
y_true = oof["label"].astype(int).to_numpy()

roc_rows, pr_rows, auc_rows = [], [], []
for label, column in curve_models.items():
    probabilities = oof[column].to_numpy()
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_rows.extend(
        [{"Model": label, "FPR": x, "TPR": y} for x, y in zip(fpr, tpr)]
    )
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_rows.extend(
        [{"Model": label, "Recall": r, "Precision": p} for r, p in zip(recall, precision)]
    )
    auc_rows.append(
        {
            "Model": label,
            "ROC-AUC": roc_auc_score(y_true, probabilities),
            "PR-AUC": auc(recall, precision),
        }
    )

roc_frame = pd.DataFrame(roc_rows)
pr_frame = pd.DataFrame(pr_rows)
metrics_rows = pd.DataFrame(auc_rows)

curve_colors = [ui.colors()["accent"], ui.colors()["good"], ui.colors()["faint"]]
roc_chart = (
    alt.Chart(roc_frame)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("TPR:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=list(curve_models), range=curve_colors),
            legend=alt.Legend(title=None),
        ),
        tooltip=["Model:N", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
    )
)
diagonal = (
    alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    .mark_line(strokeDash=[4, 4], color=ui.colors()["border_strong"])
    .encode(x="x:Q", y="y:Q")
)
pr_chart = (
    alt.Chart(pr_frame)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=list(curve_models), range=curve_colors),
            legend=alt.Legend(title=None),
        ),
        tooltip=["Model:N", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
    )
)

roc_col, pr_col = st.columns(2, gap="large")
with roc_col:
    with st.container(border=True):
        st.markdown(
            '<div class="eyebrow" style="margin-bottom:.1rem">ROC</div>'
            '<h3 style="font-size:1rem;margin:.1rem 0 .5rem 0">True vs. false positive rate</h3>',
            unsafe_allow_html=True,
        )
        st.altair_chart(ui.stylize(roc_chart + diagonal, 280), width="stretch")
with pr_col:
    with st.container(border=True):
        st.markdown(
            '<div class="eyebrow" style="margin-bottom:.1rem">Precision and recall</div>'
            '<h3 style="font-size:1rem;margin:.1rem 0 .5rem 0">The curve that matters under imbalance</h3>',
            unsafe_allow_html=True,
        )
        st.altair_chart(ui.stylize(pr_chart, 280), width="stretch")

st.markdown(
    ui.table_html(
        ["Model", "ROC-AUC", "PR-AUC"],
        [
            [ui.esc(row["Model"]), f'{row["ROC-AUC"]:.3f}', f'{row["PR-AUC"]:.3f}']
            for _, row in metrics_rows.iterrows()
        ],
        numeric={1, 2},
    ),
    unsafe_allow_html=True,
)
st.caption("PR-AUC is more informative than ROC-AUC when only 40% of alerts are real bugs.")

# ---------------------------------------------------------------------------
# Holdout confusion matrix
# ---------------------------------------------------------------------------
ui.section(
    "Holdout",
    "What the errors look like",
    "A stratified 20% split (65 alerts) held out from all tuning. False positives "
    "dominate the error budget, which is the safer direction for a triage tool.",
)

matrix = pd.read_csv(ui.REPORTS / "holdout_confusion_matrix.csv", index_col=0)
tn = int(matrix.loc["actual_fp", "predicted_fp"])
fp = int(matrix.loc["actual_fp", "predicted_tp"])
fn = int(matrix.loc["actual_tp", "predicted_fp"])
tp = int(matrix.loc["actual_tp", "predicted_tp"])
holdout = ui.load_report("holdout_metrics.csv").iloc[0]

cm_col, meta_col = st.columns([1.15, 1], gap="large")
with cm_col:
    st.markdown(
        f'<div class="cm">'
        f'<div></div><div class="head">Predicted false positive</div>'
        f'<div class="head">Predicted true positive</div>'
        f'<div class="side">Actual<br>false positive</div>'
        f'<div class="cell good"><div class="l">correctly suppressed</div>'
        f'<div class="v">{tn}</div></div>'
        f'<div class="cell bad"><div class="l">false alarm kept</div><div class="v">{fp}</div></div>'
        f'<div class="side">Actual<br>true positive</div>'
        f'<div class="cell bad"><div class="l">bug missed</div><div class="v">{fn}</div></div>'
        f'<div class="cell good"><div class="l">correctly kept</div><div class="v">{tp}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Confusion matrix at the 0.50 cut-off · {int(holdout['n_test'])} holdout alerts.")
with meta_col:
    st.markdown(
        f'<div class="card" style="height:100%">'
        '<div class="eyebrow">Holdout metrics</div>'
        f'<div class="kv"><span class="k">Accuracy</span><span class="v">{holdout["accuracy"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">Precision</span><span class="v">{holdout["precision"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">Recall</span><span class="v">{holdout["recall"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">F1</span><span class="v">{holdout["f1"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">ROC-AUC</span><span class="v">{holdout["roc_auc"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">PR-AUC</span><span class="v">{holdout["pr_auc"]:.3f}</span></div>'
        '<p style="margin-top:.8rem">7 of 65 alerts are wrong: 3 false alarms are kept '
        "and 4 real bugs are hidden. The tool is tuned to fail quietly rather than "
        "flood the analyst.</p></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Threshold explorer
# ---------------------------------------------------------------------------
ui.section(
    "Operating point",
    "Choose a threshold, see the cost",
    "The saved model does not force a single cut-off. This sweep is computed from "
    "out-of-fold scores and mirrors the slider in the Triage lab.",
)

sweep = ui.load_report("threshold_random_forest_all.csv").sort_values("threshold")
selected = st.select_slider(
    "Decision threshold",
    options=[float(value) for value in sweep["threshold"]],
    value=0.30,
    format_func=lambda value: f"{value:.2f}",
    key="insights_threshold",
)
nearest = sweep.iloc[(sweep["threshold"] - selected).abs().argmin()]
ui.stat_grid(
    [
        ("Alerts shown", f"{int(nearest['alerts_shown'])} <small>/ 325</small>", "visible queue"),
        ("False positives hidden", f"{nearest['fp_suppressed_pct']:.0f}%", "noise removed"),
        ("True positives kept", f"{nearest['tp_kept_pct']:.0f}%", "real bugs still visible"),
        ("Precision", f"{nearest['precision']:.2f}", "of the alerts shown are real"),
    ]
)

long = sweep.rename(
    columns={
        "fp_suppressed_pct": "False positives hidden",
        "tp_kept_pct": "True positives kept",
    }
).melt(
    id_vars="threshold",
    value_vars=["False positives hidden", "True positives kept"],
    var_name="Metric",
    value_name="Percent",
)
line = (
    alt.Chart(long)
    .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=35, filled=True))
    .encode(
        x=alt.X(
            "threshold:Q",
            title="Decision threshold",
            axis=alt.Axis(format=".2f", tickMinStep=0.1, tickCount=9),
        ),
        y=alt.Y("Percent:Q", title="% of alerts", scale=alt.Scale(domain=[55, 102])),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["False positives hidden", "True positives kept"],
                range=[ui.colors()["accent"], ui.colors()["good"]],
            ),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("threshold:Q", format=".2f"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Percent:Q", format=".1f"),
        ],
    )
)
rule = (
    alt.Chart(pd.DataFrame({"threshold": [selected]}))
    .mark_rule(color=ui.colors()["faint"], strokeDash=[4, 4])
    .encode(x="threshold:Q")
)
with st.container(border=True):
    st.altair_chart(ui.stylize(line + rule, 270), width="stretch")
    st.caption(
        "At 0.30 the model hides 80% of false positives and keeps 96% of true positives. "
        "Above 0.60 recall falls quickly."
    )

# ---------------------------------------------------------------------------
# Feature importance + ablation
# ---------------------------------------------------------------------------
ui.section(
    "Features",
    "What the forest actually uses",
    "Impurity-based importance, grouped by the original feature family. Text features "
    "are shown per token; context features use their `ctx_` prefix.",
)

importance = ui.load_report("feature_importance.csv").head(14).copy()
importance["label"] = importance["feature"].map(ui.pretty_feature)
importance["family"] = importance["feature"].map(
    lambda name: "Text (TF-IDF)"
    if name.startswith("text__")
    else "Metadata"
    if name.startswith("cat__")
    else "AST context"
)
imp_col, abl_col = st.columns([1.2, 1], gap="large")
with imp_col:
    chart = (
        alt.Chart(importance)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None),
            x=alt.X("importance:Q", title="Importance"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(
                    domain=["AST context", "Text (TF-IDF)", "Metadata"],
                    range=[ui.colors()["accent"], ui.colors()["warn"], ui.colors()["good"]],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", format=".4f"),
            ],
        )
    )
    st.altair_chart(ui.stylize(chart, 360), width="stretch")
    st.caption(
        "`ctx_func_len` is useful on this benchmark but may be an artefact of how "
        "variants are generated. See the ablation."
    )
with abl_col:
    ablation = ui.load_report("feature_ablation.csv")
    st.markdown(
        '<div class="eyebrow" style="margin-bottom:.4rem">Ablation</div>'
        '<h3 style="font-size:1rem;margin:.1rem 0 .6rem 0">Do the context features earn their place?</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.table_html(
            ["Feature set", "F1", "ROC-AUC"],
            [
                [
                    ui.esc(row["feature_set"]),
                    f'{row["f1"]:.3f}',
                    f'{row["roc_auc"]:.3f}',
                ]
                for _, row in ablation.iterrows()
            ],
            numeric={1, 2},
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted" style="font-size:.82rem;line-height:1.55;margin-top:.5rem">'
        "Removing <span class='mono'>ctx_func_len</span> costs about 4 F1 points, and "
        "keeping only the core taint/validation signals also hurts. The context block "
        "is doing real work, not just decorating the model.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Generalization
# ---------------------------------------------------------------------------
ui.section(
    "Generalization",
    "Does it survive unseen vulnerability families?",
    "The benchmark contains near-duplicate variants. Grouped cross-validation holds "
    "out a whole family, which is a much harder test than a random split.",
)

grouped = ui.load_report("grouped_cv.csv").iloc[0]
lofo = ui.load_report("leave_one_family_out.csv")
gen_col, lofo_col = st.columns([1, 1.5], gap="large")
with gen_col:
    st.markdown(
        f'<div class="card" style="height:100%">'
        '<div class="eyebrow">Grouped CV</div>'
        f'<div class="stat-value" style="font-size:2.2rem">{grouped["accuracy"]:.3f}</div>'
        '<div class="stat-sub" style="margin-bottom:.7rem">accuracy on an unseen family</div>'
        f'<div class="kv"><span class="k">F1</span><span class="v">{grouped["f1"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">ROC-AUC</span><span class="v">{grouped["roc_auc"]:.3f}</span></div>'
        f'<div class="kv"><span class="k">PR-AUC</span><span class="v">{grouped["pr_auc"]:.3f}</span></div>'
        '<p style="margin-top:.8rem">Lower than random CV, as expected. This is the '
        "number to quote when someone asks whether the model generalises beyond the "
        "benchmark.</p></div>",
        unsafe_allow_html=True,
    )
with lofo_col:
    rows = []
    for _, row in lofo.iterrows():
        f1_value = float(row["f1"])
        badge = (
            '<span class="badge badge-danger">weak</span>'
            if f1_value < 0.5
            else '<span class="badge">partial</span>'
        )
        rows.append(
            [
                f'<span class="mono">{ui.esc(row["family"])}</span>',
                str(int(row["n_test"])),
                f'{row["accuracy"]:.3f}',
                f'{row["recall"]:.3f}',
                f'{row["f1"]:.3f}',
                badge,
            ]
        )
    st.markdown(
        ui.table_html(
            ["Held-out family", "Alerts", "Accuracy", "Recall", "F1", "Verdict"],
            rows,
            numeric={1, 2, 3, 4},
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Command injection fails when held out entirely: the model learned family-specific "
        "text rather than a universal taint rule. The app is a triage assistant, not a "
        "replacement for a human reviewer."
    )

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
ui.section("Caveats", "Read this before trusting a number")
caveats = [
    ui.card(
        "Small benchmark",
        "325 alerts from 1,230 generated test cases. Real repositories are larger, "
        "messier and have different rule distributions.",
        num="01",
    ),
    ui.card(
        "Near-duplicate variants",
        "Random CV benefits from near-identical test cases across folds. Grouped CV "
        "and leave-one-family-out exist to expose that optimism.",
        num="02",
    ),
    ui.card(
        "Shallow data flow",
        "The context extractor follows direct assignments. Object attributes, "
        "containers and inter-procedural flows are missed.",
        num="03",
    ),
    ui.card(
        "One analyzer",
        "Only Bandit rules were used. Semgrep, CodeQL or commercial tools would need "
        "their own rule mapping and retraining.",
        num="04",
    ),
]
st.markdown(ui.cards_html(caveats, columns=4), unsafe_allow_html=True)

ui.footer()
