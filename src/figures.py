"""Generate the analysis figures.

Run from the project root:

    python src/figures.py

Every figure is written to reports/figures/ as a 300 dpi PNG. The script only
reads the CSV files produced by collect.py and train.py, so the figures can be
regenerated without retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ALERTS_CSV = ROOT / "data" / "processed" / "alerts.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

# A small, colour-blind-friendly palette.
TP_COLOR = "#2e7d32"
FP_COLOR = "#c62828"
BLUE = "#1565c0"
ORANGE = "#ef6c00"
GREY = "#607d8b"
PURPLE = "#6a1b9a"

SHORT_NAMES = {
    "Random Forest (all)": "Random Forest",
    "Decision Tree (all)": "Decision Tree",
    "Logistic Regression (all)": "Logistic Regression",
    "Logistic Regression (meta + context)": "LR (meta+context)",
    "Naive Bayes (text only)": "Naive Bayes",
    "Logistic Regression (meta + text)": "LR (meta+text)",
    "Rule-prior baseline (shallow tree)": "Rule-prior tree",
    "Majority baseline": "Majority",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def clean_feature(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("text__", "text: ")
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Dataset figures
# ---------------------------------------------------------------------------
def fig_class_distribution(alerts: pd.DataFrame) -> None:
    counts = alerts["label"].value_counts().sort_index()
    labels = ["False positive", "True positive"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, values, color=[FP_COLOR, TP_COLOR], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("Number of alerts")
    ax.set_title("Class distribution of the 325 Bandit alerts")
    ax.grid(axis="x", visible=False)
    save(fig, "fig01_class_distribution.png")


def fig_category_distribution(alerts: pd.DataFrame) -> None:
    table = (
        alerts.groupby(["gt_category", "label"]).size().unstack(fill_value=0)
        .rename(columns={0: "FP", 1: "TP"})
    )
    table = table.loc[table.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(table.index, table["FP"], color=FP_COLOR, label="False positive")
    ax.barh(table.index, table["TP"], left=table["FP"], color=TP_COLOR, label="True positive")
    for index, (fp, tp) in enumerate(zip(table["FP"], table["TP"])):
        ax.text(fp + tp + 1.5, index, f"{fp + tp}", va="center", fontsize=9)
    ax.set_xlabel("Number of alerts")
    ax.set_title("Alerts by benchmark vulnerability category")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig02_alerts_by_category.png")


def fig_rule_distribution(alerts: pd.DataFrame) -> None:
    table = (
        alerts.groupby(["rule", "label"]).size().unstack(fill_value=0)
        .rename(columns={0: "FP", 1: "TP"})
    )
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total").tail(14)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(table.index, table["FP"], color=FP_COLOR, label="False positive")
    ax.barh(table.index, table["TP"], left=table["FP"], color=TP_COLOR, label="True positive")
    for index, (fp, tp) in enumerate(zip(table["FP"], table["TP"])):
        ax.text(fp + tp + 0.7, index, f"{fp + tp}", va="center", fontsize=8)
    ax.set_xlabel("Number of alerts")
    ax.set_title("Alerts by Bandit rule (top 14 rules)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig03_alerts_by_rule.png")


# ---------------------------------------------------------------------------
# Model comparison and evaluation
# ---------------------------------------------------------------------------
def fig_model_comparison(comparison: pd.DataFrame) -> None:
    table = comparison.copy()
    table["short"] = table["model"].map(SHORT_NAMES).fillna(table["model"])
    table = table.sort_values("f1")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(table))
    height = 0.36
    ax.barh(y - height / 2, table["accuracy"], height=height, color=GREY, label="Accuracy")
    ax.barh(y + height / 2, table["f1"], height=height, color=BLUE, label="F1 score")
    for index, (accuracy, f1) in enumerate(zip(table["accuracy"], table["f1"])):
        ax.text(accuracy + 0.008, index - height / 2, f"{accuracy:.3f}", va="center", fontsize=8)
        ax.text(f1 + 0.008, index + height / 2, f"{f1:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(table["short"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score (5-fold cross-validation)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig04_model_comparison.png")


def fig_roc_pr(oof: pd.DataFrame) -> None:
    y_true = oof["label"].to_numpy()
    models = {
        "Random Forest (all)": (BLUE, "-"),
        "Logistic Regression (all)": (ORANGE, "-"),
        "Logistic Regression (meta + context)": (PURPLE, "--"),
        "Naive Bayes (text only)": (GREY, "-"),
        "Rule-prior baseline (shallow tree)": ("#8d6e63", ":"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (color, linestyle) in models.items():
        column = f"p::{name}"
        if column not in oof.columns:
            continue
        probabilities = oof[column].to_numpy()
        label = SHORT_NAMES.get(name, name)

        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)
        axes[0].plot(fpr, tpr, color=color, linestyle=linestyle,
                     label=f"{label} ({auc_value:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap_value = average_precision_score(y_true, probabilities)
        axes[1].plot(recall, precision, color=color, linestyle=linestyle,
                     label=f"{label} ({ap_value:.3f})")

    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves (out-of-fold)")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision and recall curves (out-of-fold)")
    axes[1].legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    save(fig, "fig05_roc_pr_curves.png")


def fig_confusion_matrix() -> None:
    matrix = pd.read_csv(REPORTS / "holdout_confusion_matrix.csv", index_col=0)
    values = matrix.to_numpy()
    metrics = pd.read_csv(REPORTS / "holdout_metrics.csv").iloc[0]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=values.max() * 1.15)

    labels = [["TN", "FP"], ["FN", "TP"]]
    row_labels = ["Actual FP", "Actual TP"]
    column_labels = ["Predicted FP", "Predicted TP"]
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            color = "white" if value > values.max() * 0.55 else "#1a1a1a"
            ax.text(column, row - 0.08, labels[row][column], ha="center",
                    va="center", fontsize=11, color=color)
            ax.text(column, row + 0.18, str(value), ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color)

    ax.set_xticks([0, 1], column_labels)
    ax.set_yticks([0, 1], row_labels)
    ax.set_title("80/20 holdout confusion matrix")
    ax.text(
        0.5, -0.22,
        f"accuracy {metrics['accuracy']:.3f} · precision {metrics['precision']:.3f} · "
        f"recall {metrics['recall']:.3f} · F1 {metrics['f1']:.3f}",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    ax.grid(False)
    save(fig, "fig06_confusion_matrix.png")


def fig_threshold_tradeoff() -> None:
    table = pd.read_csv(REPORTS / "threshold_random_forest_all.csv")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(table["threshold"], table["fp_suppressed_pct"], marker="o",
            color=FP_COLOR, label="False positives suppressed")
    ax.plot(table["threshold"], table["tp_kept_pct"], marker="s",
            color=TP_COLOR, label="True positives kept")
    ax.axvline(0.30, color=BLUE, linestyle="--", linewidth=1)
    row = table.loc[(table["threshold"] - 0.30).abs().idxmin()]
    ax.annotate(
        f"chosen threshold 0.30\n{row['fp_suppressed_pct']:.0f}% FP suppressed\n"
        f"{row['tp_kept_pct']:.0f}% TP kept",
        xy=(0.30, row["fp_suppressed_pct"]),
        xytext=(0.42, 66),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.9},
        color=BLUE,
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Percentage of the class")
    ax.set_ylim(55, 102)
    ax.set_xlim(0.05, 0.95)
    ax.set_title("Threshold trade-off (out-of-fold, Random Forest)")
    ax.legend(loc="lower left")
    save(fig, "fig07_threshold_tradeoff.png")


def fig_feature_importance() -> None:
    table = pd.read_csv(REPORTS / "feature_importance.csv").head(15)
    table = table.sort_values("importance")

    colors = [
        ORANGE if feature.startswith("text__") else BLUE
        for feature in table["feature"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    bars = ax.barh(
        [clean_feature(name) for name in table["feature"]],
        table["importance"],
        color=colors,
    )
    for bar, value in zip(bars, table["importance"]):
        ax.text(value + 0.0012, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Top 15 features")
    ax.set_xlim(0, table["importance"].max() * 1.16)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=BLUE, label="Metadata / AST context"),
            Patch(color=ORANGE, label="TF-IDF text"),
        ],
        loc="lower right",
    )
    ax.grid(axis="y", visible=False)
    save(fig, "fig08_feature_importance.png")


def fig_feature_ablation() -> None:
    table = pd.read_csv(REPORTS / "feature_ablation.csv")
    labels = {
        "all context": "All context\nfeatures",
        "without func_len": "Without\nctx_func_len",
        "core signals only": "Core signals\nonly",
    }
    table["label"] = table["feature_set"].map(labels).fillna(table["feature_set"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(table))
    width = 0.34
    ax.bar(x - width / 2, table["accuracy"], width, color=GREY, label="Accuracy")
    ax.bar(x + width / 2, table["f1"], width, color=BLUE, label="F1 score")
    for index, (accuracy, f1) in enumerate(zip(table["accuracy"], table["f1"])):
        ax.text(index - width / 2, accuracy + 0.008, f"{accuracy:.3f}",
                ha="center", fontsize=8)
        ax.text(index + width / 2, f1 + 0.008, f"{f1:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x, table["label"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score (5-fold CV)")
    ax.set_title("Feature ablation: context signals carry real signal")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig09_feature_ablation.png")


def fig_leave_one_family_out() -> None:
    table = pd.read_csv(REPORTS / "leave_one_family_out.csv").sort_values("f1")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = [FP_COLOR if value == 0 else BLUE for value in table["f1"]]
    bars = ax.barh(table["family"], table["f1"], color=colors)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"F1 {row['f1']:.2f} · n={int(row['n_test'])}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 score on the held-out family")
    ax.set_title("Leave-one-family-out: performance on unseen variants")
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_leave_one_family_out.png")


def fig_oof_probability_distribution(oof: pd.DataFrame) -> None:
    probabilities = oof["p::Random Forest (all)"].to_numpy()
    y_true = oof["label"].to_numpy()
    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(probabilities[y_true == 0], bins=bins, color=FP_COLOR,
            alpha=0.65, label="False positives")
    ax.hist(probabilities[y_true == 1], bins=bins, color=TP_COLOR,
            alpha=0.65, label="True positives")
    ax.axvline(0.30, color=BLUE, linestyle="--", linewidth=1,
               label="Threshold 0.30")
    ax.set_xlabel("Predicted P(true positive)")
    ax.set_ylabel("Number of alerts")
    ax.set_title("Out-of-fold Random Forest scores separate the two classes")
    ax.legend(loc="upper center")
    save(fig, "fig11_oof_probability_distribution.png")


# ---------------------------------------------------------------------------
# Pipeline diagram
# ---------------------------------------------------------------------------
def fig_pipeline() -> None:
    steps = [
        ("OWASP Benchmark for Python\n1,230 labelled test cases"),
        ("Bandit static analysis\n325 alerts on the test code"),
        ("Label rule\n130 true positives / 195 false positives"),
        ("Feature extraction\nmetadata + TF-IDF + 15 AST signals"),
        ("Random Forest\n5-fold CV, 80/20 holdout, threshold study"),
        ("Streamlit application\nupload code, see triage, move threshold"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")

    box_width = 0.78
    box_height = 0.62
    colors = ["#e3f2fd", "#fff3e0", "#e8f5e9", "#f3e5f5", "#e0f2f1", "#fce4ec"]

    for index, (text, color) in enumerate(zip(steps, colors)):
        y = len(steps) - index - 0.72
        box = FancyBboxPatch(
            (0.11, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1,
            edgecolor="#455a64",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(0.5, y + box_height / 2, text, ha="center", va="center",
                fontsize=10)

        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (0.5, y - 0.02),
                (0.5, y - 0.30),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#455a64",
            )
            ax.add_patch(arrow)

    ax.set_title("Project pipeline", fontsize=14, pad=8)
    save(fig, "fig00_pipeline.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    alerts = pd.read_csv(ALERTS_CSV)
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    oof = pd.read_csv(REPORTS / "oof_probabilities.csv")

    fig_pipeline()
    fig_class_distribution(alerts)
    fig_category_distribution(alerts)
    fig_rule_distribution(alerts)
    fig_model_comparison(comparison)
    fig_roc_pr(oof)
    fig_confusion_matrix()
    fig_threshold_tradeoff()
    fig_feature_importance()
    fig_feature_ablation()
    fig_leave_one_family_out()
    fig_oof_probability_distribution(oof)

    print(f"\nAll figures written to {FIGURES}")


if __name__ == "__main__":
    main()
