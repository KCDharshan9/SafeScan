"""Feature ablation + grouped cross-validation.

This script answers two questions for the analysis:

1. Which context features actually help?
2. Does the model still work when near-duplicate benchmark variants are kept
   out of the training folds? (A grouped split is a harder, more honest test.)

Run:

    python src/ablation_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ALERTS_CSV, ROOT
from features import CONTEXT_COLUMNS, add_context_features

REPORTS_DIR = ROOT / "reports"
META_COLUMNS = ["rule", "severity", "confidence", "n_alerts_in_file", "code_len", "code_norm"]


def make_pipeline(context_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["rule", "severity", "confidence"]),
            ("num", "passthrough", ["n_alerts_in_file", "code_len"] + context_columns),
            (
                "text",
                TfidfVectorizer(
                    max_features=800, ngram_range=(1, 2), min_df=2, sublinear_tf=True
                ),
                "code_norm",
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def metrics_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_context_features(pd.read_csv(ALERTS_CSV))
    y = df["label"].to_numpy()

    feature_sets = {
        "all context": CONTEXT_COLUMNS,
        "without func_len": [c for c in CONTEXT_COLUMNS if c != "ctx_func_len"],
        "core signals only": [
            "ctx_has_request",
            "ctx_taint",
            "ctx_validation",
            "ctx_sanitizer",
            "ctx_last_literal",
            "ctx_sink_literal",
            "ctx_string_built",
            "ctx_shell_true",
            "ctx_list_args",
            "ctx_defusedxml",
        ],
    }

    # ------------------------------------------------------------------
    # 1. Feature ablation with ordinary stratified CV.
    # ------------------------------------------------------------------
    print("Feature ablation (Random Forest, 5-fold stratified CV)")
    rows = []
    for name, columns in feature_sets.items():
        X = df[META_COLUMNS + columns]
        model = make_pipeline(columns)
        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        row = {"feature_set": name, **metrics_from_probabilities(y, probabilities)}
        rows.append(row)
        print(
            f"  {name:22s} acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
            f"roc={row['roc_auc']:.3f} pr={row['pr_auc']:.3f}"
        )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(REPORTS_DIR / "feature_ablation.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Grouped CV by variant family: harder, less leaky.
    # ------------------------------------------------------------------
    print("\nGrouped CV by benchmark variant family (Random Forest, all context)")
    X = df[META_COLUMNS + CONTEXT_COLUMNS]
    model = make_pipeline(CONTEXT_COLUMNS)
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            X, y, groups=df["family"]
        ),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    grouped = metrics_from_probabilities(y, probabilities)
    print(
        f"  acc={grouped['accuracy']:.3f} prec={grouped['precision']:.3f} "
        f"rec={grouped['recall']:.3f} f1={grouped['f1']:.3f} "
        f"roc={grouped['roc_auc']:.3f} pr={grouped['pr_auc']:.3f}"
    )
    pd.DataFrame([{"evaluation": "grouped_by_family", **grouped}]).to_csv(
        REPORTS_DIR / "grouped_cv.csv", index=False
    )

    # ------------------------------------------------------------------
    # 3. Leave-one-family-out for families that contain both classes.
    #    This is the hardest test: the held-out variant family is unseen.
    # ------------------------------------------------------------------
    print("\nLeave-one-family-out (families with both TP and FP)")
    loo_rows = []
    for family in sorted(df["family"].unique()):
        test_mask = df["family"] == family
        if df.loc[test_mask, "label"].nunique() < 2:
            continue
        train_mask = ~test_mask
        model = make_pipeline(CONTEXT_COLUMNS)
        model.fit(X[train_mask], y[train_mask])
        probabilities = model.predict_proba(X[test_mask])[:, 1]
        row = {
            "family": family,
            "n_test": int(test_mask.sum()),
            **metrics_from_probabilities(y[test_mask], probabilities),
        }
        loo_rows.append(row)
        print(
            f"  {family:20s} n={row['n_test']:3d} acc={row['accuracy']:.3f} "
            f"f1={row['f1']:.3f} roc={row['roc_auc']:.3f}"
        )

    pd.DataFrame(loo_rows).to_csv(REPORTS_DIR / "leave_one_family_out.csv", index=False)
    print(f"\nSaved results to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
