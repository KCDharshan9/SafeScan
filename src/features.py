"""Turn each alert into a feature row.

There are three kinds of features:

1. Metadata (already in alerts.csv):
   rule, severity, confidence, n_alerts_in_file, code_len

2. Text:
   code_norm -> TF-IDF later, inside the scikit-learn pipeline

3. Context (this file, using Python's `ast` module):
   Does the value reaching the sink come from user input?
   Is there a validation block before the sink?
   Is there a sanitizer?
   Is the sink argument a literal?
   ...

Why context matters
-------------------
The alert line is often identical for a true positive and a false positive.
Example from the benchmark:

    # FALSE POSITIVE
    bar = thing.doSomething(param)
    if not bar.startswith("'"): return
    exec(bar)                    # guarded, so safe

    # TRUE POSITIVE
    bar = config.get("keyB")     # derived from param
    exec(bar)                    # no guard, so real code injection

A bag-of-words model cannot tell these apart. The AST features can.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from config import BENCHMARK_DIR


def normalize_code(code: str) -> str:
    """Make code text less specific to this benchmark.

    - BenchmarkTest00025 -> BT
    - 12345              -> NUM
    - lowercase, collapse whitespace

    This stops TF-IDF from memorising test IDs and line numbers. The same
    transformation is used when training and when scoring new files.
    """
    code = str(code)
    code = re.sub(r"BenchmarkTest\d+", "BT", code)
    code = re.sub(r"\b\d+\b", "NUM", code)
    code = re.sub(r"\s+", " ", code)
    return code.strip().lower()

# ---------------------------------------------------------------------------
# Vocabulary of signals. These are intentionally simple and readable.
# ---------------------------------------------------------------------------
SOURCE_TOKENS = [
    "param",
    "request.",
    "request(",
    "input(",
    "sys.argv",
    "get_form_parameter",
    "cookies",
    "form[",
    "args.get",
    "wrapped",
    "unquote",
]

SANITIZER_TOKENS = [
    # Be specific. Plain "quote" also matches "unquote", and plain "escape"
    # also matches "escape_for_html" (output encoding, not a security guard).
    "basename",
    "shlex.quote",
    "html.escape",
    "secrets.",
    "safe_load",
    "defusedxml",
    "int(",
    "float(",
    "isinstance",
    "allowlist",
    "sanitize",
    "os.path.basename",
    "secure=",
    "sha256",
    "sha384",
    "sha512",
]

VALIDATION_TOKENS = ["startswith", "endswith", "isinstance", "==", "!=", " in ", "not ", "len("]

CONTEXT_COLUMNS = [
    "ctx_has_request",     # function reads user input before the sink
    "ctx_taint",           # backward slice reaches a source token
    "ctx_sanitizer",       # backward slice contains a sanitizer
    "ctx_validation",      # an if-guard on the sink variable returns early
    "ctx_last_literal",    # last assignment to the sink variable is a constant
    "ctx_n_assign_sink",   # how many times the sink variable is reassigned
    "ctx_depth",           # how deep the backward slice went
    "ctx_sink_literal",    # the sink argument itself is a constant
    "ctx_sink_name",       # the sink argument is a plain variable
    "ctx_has_try",         # sink is inside a try block
    "ctx_func_len",        # lines in the enclosing function
    "ctx_string_built",    # sink value is built with an f-string / + / .format
    "ctx_shell_true",      # code contains shell=True
    "ctx_list_args",       # subprocess called with a list argument
    "ctx_defusedxml",      # function uses defusedxml
]


def _empty_features() -> dict[str, int]:
    return {column: 0 for column in CONTEXT_COLUMNS}


def _find_enclosing_function(tree: ast.AST, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function whose line range contains `line`."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= line <= end:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _find_sink(func: ast.AST, line: int) -> ast.Call | ast.Assign | None:
    """Find the call or assignment that Bandit flagged.

    Most rules flag a Call (exec(bar), pickle.loads(...), subprocess.call(...)).
    B608 (SQL) flags an assignment such as:
        sql = f'SELECT ... {bar}'
    so we handle both.
    """
    candidates = []
    for node in ast.walk(func):
        if isinstance(node, (ast.Call, ast.Assign)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= line <= end:
                candidates.append(node)
    if not candidates:
        return None
    # Prefer the smallest (innermost) node, then the latest one.
    candidates.sort(key=lambda n: ((getattr(n, "end_lineno", n.lineno) - n.lineno), -n.lineno))
    return candidates[0]


def _names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_string_built(node: ast.AST) -> bool:
    """True for f-strings, string concatenation, % formatting, .format()."""
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in {"format", "replace", "join"}
    return False


def extract_context_features(source_path: Path | str, line: int) -> dict[str, int]:
    """Extract context features for one alert.

    Returns all-zero features if anything goes wrong (file missing, syntax
    error, no enclosing function). That is safe: the model can still use the
    metadata and text features.
    """
    features = _empty_features()

    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return features

    func = _find_enclosing_function(tree, line)
    if func is None:
        return features

    func_end = getattr(func, "end_lineno", line)
    features["ctx_func_len"] = max(func_end - func.lineno + 1, 0)

    sink = _find_sink(func, line)
    if sink is None:
        return features

    # ------------------------------------------------------------------
    # What is the "sink value"?
    # For a Call: its arguments.
    # For an Assign: the right-hand side.
    # ------------------------------------------------------------------
    if isinstance(sink, ast.Call):
        value_nodes = list(sink.args)
    else:
        value_nodes = [sink.value]

    arg_names: set[str] = set()
    for value in value_nodes:
        arg_names |= _names_in(value)

    features["ctx_sink_literal"] = int(
        any(isinstance(value, ast.Constant) for value in value_nodes)
    )
    features["ctx_sink_name"] = int(
        any(isinstance(value, ast.Name) for value in value_nodes)
    )
    features["ctx_string_built"] = int(
        any(_is_string_built(value) for value in value_nodes)
    )

    # ------------------------------------------------------------------
    # Collect every assignment in the function.
    # assigns[var] = [(line, value_node, rhs_source, names_in_rhs), ...]
    # ------------------------------------------------------------------
    assigns: dict[str, list[tuple[int, ast.AST, str, set[str]]]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            rhs = ast.get_source_segment(source, node.value) or ""
            rhs_names = _names_in(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns.setdefault(target.id, []).append(
                        (node.lineno, node.value, rhs, rhs_names)
                    )
    for var in assigns:
        assigns[var].sort(key=lambda item: item[0])

    features["ctx_n_assign_sink"] = sum(len(assigns.get(name, [])) for name in arg_names)

    # ------------------------------------------------------------------
    # Backward slice: follow the last assignment to each variable.
    # This is a small, deliberately simple version of taint analysis.
    # ------------------------------------------------------------------
    seen: set[str] = set()
    taint: set[str] = set()
    sanitizers: set[str] = set()
    depth = 0

    def walk_var(var: str, before: int, d: int = 0) -> None:
        nonlocal depth
        if d > 5 or var in seen or var in {"RESPONSE", "self"}:
            return
        seen.add(var)

        candidates = [item for item in assigns.get(var, []) if item[0] < before]
        if not candidates:
            return

        assignment_line, _value, rhs, rhs_names = candidates[-1]
        depth = max(depth, d + 1)
        low = rhs.lower()

        for token in SOURCE_TOKENS:
            if token in low:
                taint.add(token)
        for token in SANITIZER_TOKENS:
            if token in low:
                sanitizers.add(token)

        for name in rhs_names:
            walk_var(name, assignment_line, d + 1)

    for name in arg_names:
        walk_var(name, line)

    features["ctx_taint"] = int(bool(taint))
    features["ctx_sanitizer"] = int(bool(sanitizers))
    features["ctx_depth"] = depth

    # Last assignment to the sink variable: is it a constant?
    for name in arg_names:
        candidates = [item for item in assigns.get(name, []) if item[0] < line]
        if candidates and isinstance(candidates[-1][1], ast.Constant):
            features["ctx_last_literal"] = 1
            break

    # ------------------------------------------------------------------
    # Validation block: an `if` that mentions the sink variable and returns
    # or raises before the sink. Example:
    #     if not bar.startswith("'"): return
    # ------------------------------------------------------------------
    lines = source.splitlines()
    for node in ast.walk(func):
        if not isinstance(node, ast.If) or getattr(node, "lineno", None) is None:
            continue
        if node.lineno >= line:
            continue

        test = (ast.get_source_segment(source, node.test) or "").lower()
        if not any(name in test for name in arg_names):
            continue

        body = ""
        if node.body:
            start = node.body[0].lineno
            end = getattr(node.body[-1], "end_lineno", start)
            body = "\n".join(lines[start - 1:end]).lower()

        if any(token in test for token in VALIDATION_TOKENS) and (
            "return" in body or "raise" in body
        ):
            features["ctx_validation"] = 1
            break

    # Sink inside a try block?
    for node in ast.walk(func):
        if isinstance(node, ast.Try) and getattr(node, "lineno", 0) < line:
            features["ctx_has_try"] = 1
            break

    # ------------------------------------------------------------------
    # Cheap textual signals that are useful for specific rules.
    # ------------------------------------------------------------------
    pre_sink = "\n".join(lines[func.lineno - 1:line - 1]).lower()
    features["ctx_has_request"] = int(
        any(token in pre_sink for token in ["request.", "request(", "cookies", "form[", "args.get", "param"])
    )
    func_source = (ast.get_source_segment(source, func) or "").lower()
    features["ctx_shell_true"] = int("shell=true" in func_source)
    features["ctx_list_args"] = int(bool(re.search(r"subprocess\.\w+\(\s*\[", func_source)))
    features["ctx_defusedxml"] = int("defusedxml" in func_source)

    return features


def add_context_features(df: pd.DataFrame, benchmark_dir: Path = BENCHMARK_DIR) -> pd.DataFrame:
    """Add the context columns to an alert DataFrame.

    `df` must contain `source_path` (relative to the benchmark folder) and
    `line`. The returned DataFrame has all original columns plus CONTEXT_COLUMNS.
    """
    rows = [
        extract_context_features(benchmark_dir / str(path), int(line))
        for path, line in zip(df["source_path"], df["line"])
    ]
    context = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, context], axis=1)


# Columns the model expects. Keep this list in sync with train.py and the app.
FEATURE_COLUMNS = [
    "rule",
    "severity",
    "confidence",
    "n_alerts_in_file",
    "code_len",
    "code_norm",
] + CONTEXT_COLUMNS
