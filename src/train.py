"""Step 2: train and compare models.

What you will learn here
------------------------
- `X` is the feature table (one row per alert).
- `y` is the label column: 1 = true positive, 0 = false positive.
- A `Pipeline` chains preprocessing + classifier so the exact same steps are
  applied at training time and prediction time.
- A `ColumnTransformer` applies different preprocessing to different columns:
    * categorical columns -> OneHotEncoder
    * numeric/context columns -> passthrough or StandardScaler
    * code text -> TF-IDF
- `cross_val_predict` gives one out-of-fold prediction per alert, which lets us
  compute honest precision/recall/F1 and a threshold trade-off table.

Run from the project root:

    python src/train.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import ALERTS_CSV, MODEL_PATH, MODELS_DIR, ROOT
from features import CONTEXT_COLUMNS, FEATURE_COLUMNS, add_context_features

REPORTS_DIR = ROOT / "reports"
CATEGORICAL_COLUMNS = ["rule", "severity", "confidence"]
BASE_NUMERIC_COLUMNS = ["n_alerts_in_file", "code_len"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor(
    feature_set: str,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing for a named feature set.

    feature_set:
      "text"            -> TF-IDF on code only
      "meta"            -> rule, severity, confidence, counts
      "meta_text"       -> metadata + TF-IDF
      "meta_ctx"        -> metadata + AST context features
      "meta_text_ctx"   -> everything
    """
    transformers = []

    if feature_set in {"meta", "meta_text", "meta_ctx", "meta_text_ctx"}:
        numeric_columns = list(BASE_NUMERIC_COLUMNS)
        if feature_set in {"meta_ctx", "meta_text_ctx"}:
            numeric_columns += CONTEXT_COLUMNS

        numeric_step = StandardScaler(with_mean=False) if scale_numeric else "passthrough"
        transformers += [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", numeric_step, numeric_columns),
        ]

    if feature_set in {"text", "meta_text", "meta_text_ctx"}:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=800,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "code_norm",
            )
        )

    return ColumnTransformer(transformers)


def make_model(feature_set: str, classifier, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(feature_set, scale_numeric=scale_numeric)),
            ("clf", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray, cv) -> tuple[dict, np.ndarray]:
    """Fit with cross-validation and return pooled out-of-fold metrics."""
    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
    }
    return metrics, probabilities


def threshold_table(y: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Show the FP-suppression vs. TP-retention trade-off."""
    rows = []
    total_fp = int((y == 0).sum())
    total_tp = int((y == 1).sum())

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        kept = probabilities >= threshold
        kept_fp = int((y[kept] == 0).sum())
        kept_tp = int((y[kept] == 1).sum())
        rows.append(
            {
                "threshold": threshold,
                "alerts_shown": int(kept.sum()),
                "fp_suppressed_pct": 100 * (1 - kept_fp / total_fp) if total_fp else 0.0,
                "tp_kept_pct": 100 * kept_tp / total_tp if total_tp else 0.0,
                "precision": kept_tp / kept.sum() if kept.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load alerts and add the AST context features.
    alerts = pd.read_csv(ALERTS_CSV)
    print(f"Loaded {len(alerts)} alerts from {ALERTS_CSV}")
    alerts = add_context_features(alerts)
    print("Added context features:", ", ".join(CONTEXT_COLUMNS))

    X = alerts[FEATURE_COLUMNS]
    y = alerts["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Define the models.
    #    The shallow metadata-only tree is our "rule-prior" baseline.
    models: dict[str, Pipeline] = {
        "Majority baseline": make_model(
            "meta", DummyClassifier(strategy="most_frequent")
        ),
        "Rule-prior baseline (shallow tree)": make_model(
            "meta",
            DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=0
            ),
        ),
        "Naive Bayes (text only)": make_model(
            "text", MultinomialNB()
        ),
        "Logistic Regression (meta + text)": make_model(
            "meta_text",
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
            scale_numeric=True,
        ),
        "Logistic Regression (meta + context)": make_model(
            "meta_ctx",
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
            scale_numeric=True,
        ),
        "Logistic Regression (all)": make_model(
            "meta_text_ctx",
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
            scale_numeric=True,
        ),
        "Decision Tree (all)": make_model(
            "meta_text_ctx",
            DecisionTreeClassifier(
                max_depth=7,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=0,
            ),
        ),
        "Random Forest (all)": make_model(
            "meta_text_ctx",
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=0,
                n_jobs=-1,
            ),
        ),
    }

    # 3. Evaluate every model with the same 5-fold split.
    print("\nEvaluating models with 5-fold cross-validation...\n")
    results = []
    all_probabilities: dict[str, np.ndarray] = {}

    for name, model in models.items():
        metrics, probabilities = evaluate_model(model, X, y, cv)
        all_probabilities[name] = probabilities
        results.append({"model": name, **metrics})
        print(
            f"{name:42s} "
            f"acc={metrics['accuracy']:.3f}  "
            f"prec={metrics['precision']:.3f}  "
            f"rec={metrics['recall']:.3f}  "
            f"f1={metrics['f1']:.3f}  "
            f"roc={metrics['roc_auc']:.3f}  "
            f"pr={metrics['pr_auc']:.3f}"
        )

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Save one out-of-fold probability per alert for every model. These files
    # are what the analysis figures (ROC, PR, calibration) are built from.
    probability_table = alerts.reset_index(drop=True)[
        ["test_name", "source_path", "family", "line", "rule", "label"]
    ].copy()
    for name, probabilities in all_probabilities.items():
        probability_table[f"p::{name}"] = probabilities
    probability_table.to_csv(REPORTS_DIR / "oof_probabilities.csv", index=False)

    # 4. Threshold analysis for the two strongest models.
    for model_name in [
        "Logistic Regression (all)",
        "Random Forest (all)",
    ]:
        table = threshold_table(y, all_probabilities[model_name])
        filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        table.to_csv(REPORTS_DIR / f"threshold_{filename}.csv", index=False)
        print(f"\nThreshold analysis: {model_name}")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # 5. A final 80/20 holdout as a sanity check.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    final_model = models["Random Forest (all)"]
    final_model.fit(X_train, y_train)
    holdout_probabilities = final_model.predict_proba(X_test)[:, 1]
    holdout_predictions = (holdout_probabilities >= 0.5).astype(int)

    print("\n80/20 holdout (Random Forest):")
    print(
        f"  accuracy={accuracy_score(y_test, holdout_predictions):.3f}  "
        f"precision={precision_score(y_test, holdout_predictions, zero_division=0):.3f}  "
        f"recall={recall_score(y_test, holdout_predictions, zero_division=0):.3f}  "
        f"f1={f1_score(y_test, holdout_predictions, zero_division=0):.3f}"
    )
    print("  confusion matrix [[TN, FP], [FN, TP]]:")
    print(" ", confusion_matrix(y_test, holdout_predictions).tolist())

    # Save the holdout rows and predictions for later analysis.
    holdout_rows = alerts.loc[X_test.index, [
        "test_name", "source_path", "family", "line", "rule", "label",
    ]].reset_index(drop=True)
    holdout = pd.concat(
        [
            holdout_rows,
            pd.DataFrame(
                {
                    "y_true": y_test,
                    "y_prob": holdout_probabilities,
                    "y_pred": holdout_predictions,
                }
            ),
        ],
        axis=1,
    )
    holdout.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)

    holdout_metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, holdout_predictions),
                "precision": precision_score(y_test, holdout_predictions, zero_division=0),
                "recall": recall_score(y_test, holdout_predictions, zero_division=0),
                "f1": f1_score(y_test, holdout_predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, holdout_probabilities),
                "pr_auc": average_precision_score(y_test, holdout_probabilities),
                "n_test": len(y_test),
            }
        ]
    )
    holdout_metrics.to_csv(REPORTS_DIR / "holdout_metrics.csv", index=False)

    matrix = confusion_matrix(y_test, holdout_predictions)
    pd.DataFrame(
        matrix,
        index=["actual_fp", "actual_tp"],
        columns=["predicted_fp", "predicted_tp"],
    ).to_csv(REPORTS_DIR / "holdout_confusion_matrix.csv")

    # 6. Fit the final model on all data and save it for the Streamlit app.
    final_model.fit(X, y)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": 0.3,
        },
        MODEL_PATH,
    )
    print(f"\nSaved final pipeline to {MODEL_PATH}")

    # 7. Save the most important features.
    pre = final_model.named_steps["pre"]
    clf = final_model.named_steps["clf"]
    feature_names = pre.get_feature_names_out()
    importances = clf.feature_importances_
    importance_table = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(30)
    )
    importance_table.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print("\nTop 15 features:")
    print(importance_table.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nSaved metrics to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
