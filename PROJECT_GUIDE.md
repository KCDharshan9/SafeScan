# SafeScan: Complete Project Guide

## 1. The problem this project solves

Static application security testing (SAST) tools are designed to be cautious. That is useful because missing a real vulnerability can be costly, but it creates a practical problem: a scanner may report many findings that are harmless in their actual code context.

For example, Bandit can report an `exec()` call because executing dynamically built code can be dangerous. These two cases look similar to a rule-based scanner:

```python
# Likely dangerous: request input reaches exec()
value = request.args["expression"]
exec(value)
```

```python
# Potentially safer: code validates the value before execution
value = get_setting()
if not value.startswith("'"):
    raise ValueError("expected a literal")
exec(value)
```

The first example may be a genuine code-injection flaw. The second may be a false positive. The model in this project learns from labelled examples to place the first kind of alert higher in a human review queue.

The goal is not to delete findings automatically. The goal is to reduce analyst noise while retaining as many real findings as possible.

## 2. Main idea in one sentence

**Bandit finds candidate issues; the ML model estimates which candidates are most worth reviewing first.**

## 3. What happens when you scan code

The Triage page has two source options:

| Source | Typical use | Ground-truth labels? |
|---|---|---:|
| Upload file | Review one local Python file | No |
| Paste code | Quickly inspect a short snippet | No |

The workflow is intentionally explicit:

1. Select or provide source code.
2. Click **Run analysis**.
3. Bandit scans the source.
4. The app builds the model features and scores every alert.
5. Adjust the threshold, search, filter, select an alert, and inspect its context.

Changing the slider, table selection, or filters does not trigger another scan. This is important for two reasons: it makes the interface feel responsive, and it avoids scanning incomplete code while somebody is typing.

## 4. Understanding the score and threshold

Each finding receives `P(true positive)`, a number from `0.000` to `1.000`.

- A score near `1` means the alert resembles true positives from the training data.
- A score near `0` means it resembles false positives from the training data.
- It is not a legal, security, or production-risk guarantee.

The threshold turns that score into a workflow decision:

```text
score >= threshold: Review queue
score < threshold: Suppress as likely noise
```

At `0.30`, an alert scored `0.67` is sent to review and an alert scored `0.16` is suppressed. Raising the threshold reduces queue size but risks hiding true positives. Lowering it catches more possible bugs but sends more noise to reviewers.

The best threshold depends on your goal:

| Situation | Safer starting direction |
|---|---|
| A security-critical release | Lower threshold; prioritise recall |
| A noisy backlog needing human triage | Moderate threshold; inspect borderline items |
| A new codebase unlike the training data | Start low and validate model behaviour manually |

## 5. Data source and labels

The project uses the [OWASP Benchmark for Python](https://github.com/OWASP-Benchmark/BenchmarkPython), version 0.1. It contains labelled Python security test cases covering categories such as SQL injection, command injection, XXE, weak randomness, and deserialization.

The data preparation process runs Bandit on the test cases and joins each alert to benchmark labels. An alert counts as a true positive only when:

1. the test case contains a real benchmark vulnerability, and
2. the Bandit rule matches the vulnerability category.

Other alerts are treated as false positives for this evaluation. Uploaded code and pasted code do not have verified labels, so SafeScan does not present any as confirmed vulnerabilities.

The saved dataset contains 325 Bandit alerts: 130 labelled true positives and 195 labelled false positives.

## 6. Features used by the model

The model does not make its decision from the rule ID alone. It combines three kinds of evidence.

### Rule metadata

This includes Bandit rule ID, severity, confidence, number of findings in the file, and code length. A rule can have a different historical false-positive rate than another rule.

### Flagged code text

The flagged code is normalised and transformed using TF-IDF. This gives useful signals from code tokens without requiring a large neural model.

### AST context features

The code is parsed into a Python abstract syntax tree (AST). The extractor looks for simple contextual clues near the finding, including:

- a request or environment source reaching a sink;
- validation or sanitisation before a sink;
- literals, list arguments, or shell usage;
- direct assignment relationships;
- XML and deserialization-related safety signals.

The Triage detail panel names the context signals that fired, so a reviewer can understand the evidence behind the score. These explanations are clues, not a full proof of how the model arrived at every probability.

## 7. Model and training

The saved model is a scikit-learn Random Forest pipeline. It accepts the combined feature groups and predicts a probability for the positive class.

The repository also evaluates logistic regression, decision tree, naive Bayes, rule-prior, and majority baselines. Keeping alternative models matters: a strong-looking model is more credible when it is compared with simpler approaches, and the logistic model provides a useful sanity check.

Training and evaluation code lives in `src/train.py`; the saved model bundle is `models/rf_pipeline.joblib`.

## 8. Evaluation results

The project records multiple evaluation views rather than relying only on one accuracy number:

| Evaluation | Recorded result | Why it matters |
|---|---:|---|
| 5-fold stratified CV accuracy | 0.908 | General score across balanced folds |
| 5-fold stratified CV ROC-AUC | 0.966 | How well the model ranks positives above negatives |
| 20% holdout accuracy | 0.892 | Performance on data not used during tuning |
| Grouped CV accuracy | 0.834 | More realistic check for unseen vulnerability families |

Grouped validation is especially important. Security benchmarks can include similar variants of the same vulnerability. If near-duplicates appear in both training and testing, ordinary random splitting can look better than real-world generalisation. The grouped split asks the harder question: can the model still work on a related but unseen family?

## 9. Application pages

### Overview

An introduction to the detection, feature, score, and triage pipeline, a side-by-side context example, and a summary of recorded performance.

### Triage

The working analyst console. It has a manual scan action, threshold control, queue metrics, interactive table, filtering, CSV export, per-alert evidence, and source-code viewer.

### Model Insights

Evaluation details such as model comparisons, ROC/precision-recall curves, confusion matrix, threshold sweep, feature importance, ablation studies, grouped validation, and known limitations.

### Dataset & Method

The dataset card, label rule, feature engineering method, validation protocol, leakage controls, and primary sources.

## 10. Directory map

```text
app/
  streamlit_app.py       Application entry point and page registration
  common.py              Theme, shared UI, cached data/model access
  views/                 Four user-facing pages
data/
  raw/                   OWASP Benchmark source and expected labels
  processed/             Labelled alert table and raw Bandit output
models/                  Saved trained model pipeline
reports/                 Evaluation tables and reproducible visual assets
src/
  collect.py             Runs Bandit and builds the labelled alert dataset
  features.py            Builds text, metadata, and AST-context features
  train.py               Trains and evaluates candidate models
  predict.py             Runs Bandit plus model scoring on new code
tests/                   Regression checks for core inference behaviour
```

## 11. Installation and local use

Requirements are pinned in `requirements.txt`. On Windows, use the virtual environment's Python directly so the command does not depend on PATH configuration:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

To score a file without the UI:

```powershell
.\venv\Scripts\python.exe src\predict.py path\to\your_file.py --threshold 0.50
```

## 12. Reproducing the training artefacts

Run these commands in sequence after the benchmark data is present:

```powershell
.\venv\Scripts\python.exe src\collect.py
.\venv\Scripts\python.exe src\train.py
.\venv\Scripts\python.exe src\figures.py
```

`collect.py` creates `data/processed/alerts.csv`. `train.py` builds the model and writes evaluation outputs. `figures.py` regenerates the supporting analysis figures.

## 13. Important limitations

- The model was trained on a specific Python benchmark. Different frameworks, coding standards, libraries, and threat models may shift its performance.
- The AST context extractor follows direct assignments and simple guards. It may miss flows through objects, containers, callbacks, framework internals, or complex aliasing.
- A high score does not prove exploitability. A low score does not prove safety.
- The current interface accepts a single uploaded Python file or pasted code. It does not upload entire repositories.
- There is no user authentication, audit trail, policy engine, or production integration in this educational project.

## 14. Safe use recommendations

1. Keep Bandit’s raw output available; do not silently discard findings.
2. Treat the model score as a ranking aid.
3. Use lower thresholds for high-impact systems.
4. Measure results on your own labelled findings before changing team policy.
5. Retrain or recalibrate when your codebase differs materially from the OWASP Benchmark.

## 15. Primary references

- [OWASP Benchmark for Python repository](https://github.com/OWASP-Benchmark/BenchmarkPython)
- [Bandit documentation](https://bandit.readthedocs.io/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Streamlit documentation](https://docs.streamlit.io/)
