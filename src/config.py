"""Project-wide paths and constants.

Keeping paths in one place means the training script, the Streamlit app, and
the feature extractor all read/write the same files.
"""
from pathlib import Path

# Project root = one level above this file (src/config.py -> project/)
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
BENCHMARK_DIR = DATA_RAW / "owasp-benchmark-python"
TESTCODE_DIR = BENCHMARK_DIR / "testcode"
GROUND_TRUTH_CSV = BENCHMARK_DIR / "expectedresults-0.1.csv"

DATA_PROCESSED = ROOT / "data" / "processed"
ALERTS_CSV = DATA_PROCESSED / "alerts.csv"
BANDIT_JSON = DATA_PROCESSED / "bandit_raw.json"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_pipeline.joblib"

# ---------------------------------------------------------------------------
# Label definition
# ---------------------------------------------------------------------------
# An alert is a TRUE POSITIVE only when:
#   1. the benchmark says the test case contains a real vulnerability, AND
#   2. the Bandit rule belongs to that vulnerability category.
#
# Every other alert is a FALSE POSITIVE. This mirrors how SAST benchmarks
# score tools: an alert only "counts" if it finds the intended vulnerability.
#
# Note: rules that are not in this map (e.g. B403 "import pickle",
# B406/B408 "import xml", B404 "import subprocess") are generic import
# warnings. They are not evidence of the specific injected vulnerability,
# so they are labelled FALSE POSITIVE.
# Human-readable names for the rules that appear in this benchmark. Bandit's
# JSON calls many blacklist checks simply "blacklist", which is not useful in
# a report or a UI table.
RULE_NAMES = {
    "B102": "exec used",
    "B301": "pickle deserialization",
    "B307": "eval used",
    "B311": "weak random number",
    "B314": "xml.etree parse",
    "B317": "xml.sax parse",
    "B318": "xml.dom.minidom parse",
    "B319": "xml.dom.pulldom parse",
    "B320": "lxml etree parse",
    "B403": "import pickle",
    "B404": "import subprocess",
    "B406": "import xml.sax",
    "B408": "import xml.dom.minidom",
    "B506": "yaml.load",
    "B602": "subprocess with shell=True",
    "B603": "subprocess without shell",
    "B608": "hardcoded SQL expression",
}

RULE_TO_CATEGORY = {
    # weak randomness
    "B311": "weakrand",
    # weak hashing
    "B324": "hash",
    # code injection (exec / eval)
    "B102": "codeinj",
    "B307": "codeinj",
    # unsafe deserialization
    "B301": "deserialization",
    "B506": "deserialization",
    # command injection
    "B602": "cmdi",
    "B603": "cmdi",
    "B604": "cmdi",
    "B605": "cmdi",
    "B606": "cmdi",
    "B607": "cmdi",
    "B609": "cmdi",
    # SQL injection
    "B608": "sqli",
    "B610": "sqli",
    "B611": "sqli",
    # XSS
    "B701": "xss",
    "B702": "xss",
    "B703": "xss",
    "B704": "xss",
    # XXE
    "B314": "xxe",
    "B318": "xxe",
    "B319": "xxe",
    "B320": "xxe",
}
