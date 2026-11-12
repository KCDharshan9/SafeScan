"""Step 1 of the project: turn raw Bandit output into a labelled table.

What this script does
---------------------
1. Runs Bandit on the OWASP Benchmark Python test cases.
2. Parses Bandit's JSON output.
3. Joins each alert with the benchmark's ground truth.
4. Assigns a binary label: 1 = true positive, 0 = false positive.
5. Saves one row per alert to data/processed/alerts.csv.

Run it from the project root:

    python src/collect.py

You should see approximately:
    325 alerts
    130 true positives
    195 false positives

If you see very different numbers, stop and debug before moving on.
"""
from __future__ import annotations

import json
import re
# Runs a fixed Bandit collection command.
import subprocess  # nosec B404
import sys
from pathlib import Path

import pandas as pd

from config import (
    ALERTS_CSV,
    BANDIT_JSON,
    BENCHMARK_DIR,
    DATA_PROCESSED,
    GROUND_TRUTH_CSV,
    RULE_NAMES,
    RULE_TO_CATEGORY,
    TESTCODE_DIR,
)
from features import normalize_code


# ---------------------------------------------------------------------------
# 1. Run Bandit
# ---------------------------------------------------------------------------
def run_bandit() -> None:
    """Run Bandit and write its JSON report to data/processed/bandit_raw.json."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "bandit",
        "-r", "testcode",
        "-f", "json",
        "-o", str(BANDIT_JSON),
        "-q",  # quiet: do not print the progress bar
    ]
    print("Running:", " ".join(cmd))
    try:
        # The command is static and never shell-parsed.
        result = subprocess.run(  # nosec B603
            cmd,
            cwd=BENCHMARK_DIR,
            capture_output=True,
            text=True,
            timeout=180,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Bandit collection exceeded the 180-second limit.") from exc

    # Bandit exits with 1 when it finds issues. That is not an error for us.
    if result.returncode not in (0, 1):
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Bandit failed with exit code {result.returncode}")

    if not BANDIT_JSON.exists():
        raise RuntimeError("Bandit did not produce a JSON report.")


# ---------------------------------------------------------------------------
# 2. Load ground truth
# ---------------------------------------------------------------------------
def load_ground_truth() -> dict[str, dict]:
    """Return {test_name: {category, real, cwe}} from expectedresults-0.1.csv."""
    truth: dict[str, dict] = {}
    with GROUND_TRUTH_CSV.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            test_name, category, real, cwe = [part.strip() for part in line.split(",")]
            truth[test_name] = {
                "category": category,
                "real": real.lower() == "true",
                "cwe": cwe,
            }
    return truth


# ---------------------------------------------------------------------------
# 3. Small helpers
# ---------------------------------------------------------------------------
TEST_NAME_RE = re.compile(r"(BenchmarkTest\d+)")
FAMILY_RE = re.compile(r"/benchmark/([a-z]+-\d+)/")


def extract_test_name(filename: str) -> str | None:
    match = TEST_NAME_RE.search(filename)
    return match.group(1) if match else None


def extract_family(relative_path: str) -> str:
    """Group near-duplicate benchmark variants, e.g. 'weakrand-00'.

    We use this only for a grouped cross-validation check, never as a model
    feature. Using it as a feature would leak the vulnerability category.
    """
    path = BENCHMARK_DIR / relative_path.replace("\\", "/")
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"
    match = FAMILY_RE.search(source)
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------------------
# 4. Build the alert table
# ---------------------------------------------------------------------------
def build_alert_table(truth: dict[str, dict]) -> pd.DataFrame:
    with BANDIT_JSON.open(encoding="utf-8") as fh:
        report = json.load(fh)

    results = report.get("results", [])
    if not results:
        raise RuntimeError("Bandit found no alerts. Check the scan path.")

    alerts_per_file: dict[str, int] = {}
    for alert in results:
        name = extract_test_name(alert["filename"])
        if name:
            alerts_per_file[name] = alerts_per_file.get(name, 0) + 1

    rows: list[dict] = []
    for alert in results:
        test_name = extract_test_name(alert["filename"])
        if test_name is None or test_name not in truth:
            continue

        gt = truth[test_name]
        rule = alert["test_id"]

        # The label rule (see config.py for the full explanation).
        is_true_positive = int(
            gt["real"] and RULE_TO_CATEGORY.get(rule) == gt["category"]
        )

        relative_path = alert["filename"].replace("\\", "/")
        code = alert.get("code", "")

        rows.append(
            {
                "test_name": test_name,
                "source_path": relative_path,
                "family": extract_family(relative_path),
                "rule": rule,
                "rule_name": RULE_NAMES.get(rule, alert.get("test_name", "")),
                "severity": alert.get("issue_severity", ""),
                "confidence": alert.get("issue_confidence", ""),
                "line": int(alert.get("line_number", 0)),
                "code": code,
                "code_norm": normalize_code(code),
                "code_len": len(code),
                "n_alerts_in_file": alerts_per_file[test_name],
                "gt_category": gt["category"],
                "gt_real": gt["real"],
                "gt_cwe": gt["cwe"],
                "label": is_true_positive,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not TESTCODE_DIR.exists():
        raise SystemExit(
            f"Benchmark not found at {TESTCODE_DIR}.\n"
            "Download OWASP Benchmark Python and place it under "
            "data/raw/owasp-benchmark-python/."
        )
    if not GROUND_TRUTH_CSV.exists():
        raise SystemExit(f"Ground truth file not found: {GROUND_TRUTH_CSV}")

    run_bandit()
    truth = load_ground_truth()
    df = build_alert_table(truth)

    # Remove exact duplicate alerts (same file, line, rule).
    before = len(df)
    df = df.drop_duplicates(subset=["source_path", "line", "rule"]).reset_index(drop=True)
    removed = before - len(df)

    ALERTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALERTS_CSV, index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(df)} alerts to {ALERTS_CSV}")
    if removed:
        print(f"Removed {removed} duplicate alerts")
    print("\nLabel distribution:")
    print(df["label"].value_counts().rename({0: "FP (0)", 1: "TP (1)"}).to_string())
    print("\nAlerts by benchmark category and label:")
    print(
        df.groupby(["gt_category", "label"]).size()
        .unstack(fill_value=0)
        .rename(columns={0: "FP", 1: "TP"})
        .to_string()
    )
    print("\nAlerts by rule and label:")
    print(
        df.groupby(["rule", "label"]).size()
        .unstack(fill_value=0)
        .rename(columns={0: "FP", 1: "TP"})
        .to_string()
    )
    print(f"\nDistinct variant families: {df['family'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
