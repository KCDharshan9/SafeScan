"""Score new Bandit alerts with the trained model.

This module is the bridge between the trained pipeline and the Streamlit app.
It does four things:

1. run Bandit on a file or folder,
2. turn Bandit's JSON into the same table shape used for training,
3. add the AST context features, and
4. return `P(true positive)` for every alert.

Command line examples
---------------------
    python src/predict.py path/to/source_file.py
    python src/predict.py path/to/folder --threshold 0.5
    python src/predict.py path/to/file.py --json
"""
from __future__ import annotations

import argparse
import json
# Runs Bandit with a fixed interpreter and module.
import subprocess  # nosec B404
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, RULE_NAMES
from features import FEATURE_COLUMNS, add_context_features, normalize_code


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: Path | str = MODEL_PATH) -> dict:
    """Load the joblib bundle saved by train.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )

    bundle = joblib.load(path)
    for key in ("model", "feature_columns"):
        if key not in bundle:
            raise KeyError(f"Model bundle is missing the '{key}' entry.")

    # Older bundles may not have a threshold; 0.5 is the standard default.
    bundle.setdefault("threshold", 0.5)
    return bundle


# ---------------------------------------------------------------------------
# Run Bandit
# ---------------------------------------------------------------------------
def run_bandit(target: Path | str) -> list[dict]:
    """Run Bandit on one file or a directory and return the results list."""
    target = Path(target)
    if not target.exists():
        raise FileNotFoundError(target)

    cmd = [sys.executable, "-m", "bandit", "-f", "json", "-q"]
    if target.is_dir():
        cmd += ["-r", str(target)]
    else:
        cmd += [str(target)]

    try:
        # The target is passed as one argument and is never shell-parsed.
        process = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Bandit exceeded the 60-second scan limit.") from exc

    # Bandit exits with 1 when it finds issues. Only 2+ is a real failure.
    if process.returncode not in (0, 1):
        raise RuntimeError(
            f"Bandit failed with exit code {process.returncode}.\n"
            f"{process.stderr.strip()}"
        )

    output = process.stdout.strip()
    if not output:
        return []

    try:
        report = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Bandit output: {exc}") from exc

    return report.get("results", [])


# ---------------------------------------------------------------------------
# Bandit results -> feature table
# ---------------------------------------------------------------------------
def alerts_to_frame(results: list[dict], base_dir: Path | str) -> pd.DataFrame:
    """Convert Bandit results into the alert-table shape used in training.

    `source_path` is stored relative to `base_dir` so the context extractor
    can find the file again. If a path cannot be made relative, it is kept
    as-is.
    """
    base_dir = Path(base_dir).resolve()

    alert_paths = [Path(alert["filename"]).resolve() for alert in results]
    counts: dict[Path, int] = {}
    for resolved in alert_paths:
        counts[resolved] = counts.get(resolved, 0) + 1

    rows: list[dict] = []
    for alert, resolved in zip(results, alert_paths):
        try:
            relative = resolved.relative_to(base_dir)
        except ValueError:
            relative = resolved

        code = alert.get("code", "") or ""
        rule = alert.get("test_id", "")
        rows.append(
            {
                "test_name": resolved.stem,
                "source_path": str(relative),
                "rule": rule,
                "rule_name": RULE_NAMES.get(rule, alert.get("test_name", "")),
                "severity": alert.get("issue_severity", ""),
                "confidence": alert.get("issue_confidence", ""),
                "line": int(alert.get("line_number", 0) or 0),
                "code": code,
                "code_norm": normalize_code(code),
                "code_len": len(code),
                "n_alerts_in_file": counts[resolved],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_frame(
    alerts: pd.DataFrame,
    base_dir: Path | str,
    bundle: dict,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Add context features, predict P(TP), and apply the threshold."""
    if alerts.empty:
        return alerts.copy()

    featured = add_context_features(alerts, Path(base_dir))

    # Defensive: if a future model expects a column we did not produce, fill
    # it with zero instead of crashing the app.
    for column in bundle["feature_columns"]:
        if column not in featured.columns:
            featured[column] = 0

    X = featured[bundle["feature_columns"]]
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    cut = float(bundle["threshold"] if threshold is None else threshold)
    featured["p_true_positive"] = probabilities
    featured["decision"] = np.where(
        probabilities >= cut, "keep", "suppress"
    )
    return featured


def analyze(
    target: Path | str,
    bundle: dict | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run the full pipeline on a file or directory.

    Returns one row per alert with `p_true_positive` and `decision`.
    """
    target = Path(target).resolve()
    bundle = bundle or load_model()

    results = run_bandit(target)
    base_dir = target if target.is_dir() else target.parent
    alerts = alerts_to_frame(results, base_dir)

    if alerts.empty:
        return alerts

    scored = score_frame(alerts, base_dir, bundle, threshold=threshold)
    return scored.sort_values("p_true_positive", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Score Bandit alerts with the trained model.")
    parser.add_argument("target", type=Path, help="Python file or folder to scan.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Hide alerts below this probability (default: model's saved threshold).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the scored alerts as JSON instead of a table.",
    )
    args = parser.parse_args()

    bundle = load_model()
    scored = analyze(args.target, bundle=bundle, threshold=args.threshold)

    if scored.empty:
        print("Bandit found no alerts.")
        return

    cut = float(bundle["threshold"] if args.threshold is None else args.threshold)
    kept = scored["p_true_positive"] >= cut
    suppressed = len(scored) - int(kept.sum())

    if args.json:
        columns = [
            "test_name", "line", "rule", "rule_name", "severity", "confidence",
            "p_true_positive", "decision", "code",
        ]
        print(scored[columns].to_json(orient="records", indent=2))
        return

    print(f"\nScanned: {Path(args.target).resolve()}")
    print(f"Alerts: {len(scored)}    kept: {int(kept.sum())}    "
          f"suppressed: {suppressed}    threshold: {cut:.2f}\n")

    view = scored[["line", "rule", "p_true_positive", "decision", "code"]].copy()
    view["p_true_positive"] = view["p_true_positive"].map(lambda value: f"{value:.3f}")
    view["code"] = view["code"].str.strip().str.slice(0, 60)
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
