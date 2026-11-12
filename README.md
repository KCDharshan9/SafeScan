# SafeScan

An ML-assisted triage application for Python static-analysis findings.

Bandit identifies potentially risky code. This project adds a second layer that ranks those findings by their estimated likelihood of being a real vulnerability, helping an analyst spend time on the most valuable alerts first.

> This is a prioritisation tool, not an automatic security decision-maker. A suppressed finding can still be real and should be reviewed under the right operating policy.

## What you can do

- Upload one `.py` file or paste Python code.
- Run Bandit and score each finding with the saved Random Forest model.
- Choose a score threshold to create a smaller review queue.
- Search and filter findings, inspect the flagged code and context signals, then export the visible queue as CSV.
- Explore evaluation results, dataset construction, features, leakage controls, and model limits.

## Run locally on Windows

Use these commands from the project folder. They do not require activating the virtual environment, so Windows will always use the installed Streamlit package.

```powershell
cd SafeScan
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open the local URL shown by Streamlit. Start on **Triage**, choose a source, and select **Run analysis**. The app deliberately scans only after that action, so changing a threshold or a filter is immediate and never re-runs Bandit.

If you have already created `venv` and installed the requirements, only run the last command. Do not use `streamlit run ...` by itself. Windows only recognises that command after activation; `python -m streamlit` works either way.

## How it works

1. **Detect:** Bandit inspects Python source and emits security findings.
2. **Enrich:** the app adds rule metadata, normalised code text, and AST-derived context signals.
3. **Score:** the trained pipeline returns `P(true positive)` for each finding.
4. **Triage:** findings at or above the chosen threshold enter the review queue; lower-scoring findings are marked for suppression.

At the default `0.30` threshold, the saved benchmark evaluation hides roughly 80% of false positives while retaining roughly 96% of true positives. Those values describe this benchmark, not a guarantee for arbitrary code.

## Repository layout

```text
app/                 # Streamlit entry point, shared UI helpers, and pages
data/                # OWASP source data and labelled Bandit alerts
models/              # Saved scikit-learn pipeline
reports/             # Reproducible evaluation CSVs and figures
src/                 # Collection, features, training, prediction, and figures
tests/               # Focused regression tests
```

## Development commands

```powershell
# Rebuild the labelled alert table
.\venv\Scripts\python.exe src\collect.py

# Train and evaluate the model
.\venv\Scripts\python.exe src\train.py

# Regenerate analysis charts
.\venv\Scripts\python.exe src\figures.py

# Run focused checks
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

`streamlit` is not recognized

Use this command from the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

It runs Streamlit from the virtual environment directly and does not depend on your system PATH.

## References

- [OWASP Benchmark for Python](https://github.com/OWASP-Benchmark/BenchmarkPython)
- [Bandit documentation](https://bandit.readthedocs.io/)
- [Streamlit documentation](https://docs.streamlit.io/)
