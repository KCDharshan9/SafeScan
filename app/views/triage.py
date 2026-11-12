"""SafeScan's interactive source-code triage workspace."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as ui  # noqa: E402


def highlight_line(code: str, line: int) -> int | None:
    for index, raw in enumerate(str(code).splitlines(), start=1):
        match = re.match(r"^\s*(\d+)\s", raw)
        if match and int(match.group(1)) == int(line):
            return index
    return None


def save_scan(loader, source_label: str, preview_text: str) -> None:
    try:
        with st.spinner("Scanning source and ranking findings..."):
            result = loader()
        st.session_state["triage_scan"] = {
            "data": result,
            "source_label": source_label,
            "preview_text": preview_text,
        }
        st.session_state.pop("triage_table", None)
        st.success("Scan complete. Review the prioritised findings below.")
    except (OSError, RuntimeError, ValueError) as exc:
        st.session_state.pop("triage_scan", None)
        st.error(f"The scan could not complete: {exc}")


ui.topbar("Triage")
ui.page_head(
    "Analyst workspace",
    "Review findings with clarity.",
    "Scan a Python file or paste source code. SafeScan ranks Bandit findings using contextual signals so you can focus your attention where it matters.",
)
st.markdown(
    '<div class="workspace-kicker">1. Provide source &nbsp; 2. Run analysis &nbsp; 3. Review the ranked queue</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Start a scan")
    st.caption("Your source stays in this local session. A scan runs only when you select Run analysis.")
    mode = st.segmented_control(
        "Source type",
        ["Upload file", "Paste code"],
        default="Upload file",
        key="triage_mode",
        width="stretch",
        label_visibility="collapsed",
    )
    if mode == "Upload file":
        with st.form("upload_scan_form", border=False):
            uploaded = st.file_uploader(
                "Python file",
                type=["py"],
                help="Select one Python file to analyse locally.",
            )
            submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")
        if submitted:
            if uploaded is None:
                st.warning("Select a Python file before running analysis.")
            else:
                content = uploaded.getvalue()
                filename = Path(uploaded.name).name
                save_scan(
                    lambda: ui.analyze_bytes(filename, content),
                    filename,
                    content.decode("utf-8", errors="ignore"),
                )
    else:
        with st.form("paste_scan_form", border=False):
            pasted = st.text_area(
                "Python code",
                height=190,
                placeholder="def handle_input(value):\n    return value",
                help="Paste a self-contained Python snippet to analyse.",
            )
            filename = st.text_input("File name", value="source.py")
            submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")
        if submitted:
            if not pasted.strip():
                st.warning("Paste Python code before running analysis.")
            else:
                filename = Path(filename.strip() or "source.py").name
                filename = filename if filename.endswith(".py") else f"{filename}.py"
                save_scan(
                    lambda: ui.analyze_bytes(filename, pasted.encode()), filename, pasted
                )

scan = st.session_state.get("triage_scan")
if not scan:
    st.info("Provide Python source above, then select Run analysis to create a ranked review queue.")
    ui.footer()
    st.stop()

df: pd.DataFrame = scan["data"].copy()
if df.empty:
    st.success("No Bandit findings were detected in this source.")
    st.caption("SafeScan found no security checks that require review in this file.")
    ui.footer()
    st.stop()

st.markdown("### Scan summary")
summary_left, summary_right = st.columns([2.2, 1], gap="large", vertical_alignment="bottom")
with summary_left:
    threshold = st.slider(
        "Review findings with a model score of at least",
        0.05,
        0.95,
        value=float(st.session_state.get("triage_threshold", ui.DEFAULT_THRESHOLD)),
        step=0.05,
        key="triage_threshold",
        help="Lower thresholds preserve more possible findings. Higher thresholds reduce likely noise.",
    )
with summary_right:
    st.metric("Active threshold", f"{threshold:.2f}")

df["kept"] = df["p_true_positive"] >= threshold
df["decision"] = df["kept"].map({True: "Review", False: "Deprioritise"})
raw_count = len(df)
review_count = int(df["kept"].sum())
deprioritised_count = raw_count - review_count
median_score = df.loc[df["kept"], "p_true_positive"].median() if review_count else None
ui.stat_grid([
    ("Findings detected", str(raw_count), f"Bandit alerts in {scan['source_label']}"),
    ("Review queue", str(review_count), f"{review_count / raw_count:.0%} meet the threshold"),
    ("Deprioritised", str(deprioritised_count), f"{deprioritised_count / raw_count:.0%} below the threshold"),
    ("Median review score", "Not available" if median_score is None else f"{median_score:.2f}", "among findings in the review queue"),
])

st.markdown("### Ranked findings")
filter_left, filter_mid, filter_right = st.columns([1.5, 1.25, 1], gap="medium")
with filter_left:
    search = st.text_input("Search", placeholder="Rule, file, or finding", key="triage_search")
with filter_mid:
    rules = st.multiselect(
        "Bandit rules", sorted(df["rule"].dropna().unique()), key="triage_rules", placeholder="All rules"
    )
with filter_right:
    view_mode = st.selectbox(
        "Queue", ["Review queue", "All findings", "Deprioritised"], key="triage_view"
    )

view = df.copy()
if view_mode == "Review queue":
    view = view[view["kept"]]
elif view_mode == "Deprioritised":
    view = view[~view["kept"]]
if rules:
    view = view[view["rule"].isin(rules)]
if search.strip():
    needle = search.strip().lower()
    searchable = view[["rule", "rule_name", "test_name"]].fillna("").astype(str)
    view = view[searchable.apply(lambda column: column.str.lower().str.contains(needle, regex=False)).any(axis=1)]
view = view.sort_values(["p_true_positive", "line"], ascending=[False, True]).reset_index(names="alert_index")

if view.empty:
    st.info("No findings match the current filters. Select All findings or clear a filter.")
else:
    display = view[["alert_index", "line", "rule", "rule_name", "severity", "confidence", "p_true_positive", "decision"]].copy()
    display["p_true_positive"] = display["p_true_positive"].round(3)
    event = st.dataframe(
        display,
        column_order=["line", "rule", "rule_name", "severity", "confidence", "p_true_positive", "decision"],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="triage_table",
        width="stretch",
        height=min(410, 70 + len(display) * 36),
        column_config={
            "line": st.column_config.NumberColumn("Line", width="small", format="%d"),
            "rule": st.column_config.TextColumn("Rule", width="small"),
            "rule_name": st.column_config.TextColumn("Finding", width="large"),
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "confidence": st.column_config.TextColumn("Bandit confidence", width="medium"),
            "p_true_positive": st.column_config.ProgressColumn("Model score", min_value=0.0, max_value=1.0, format="%.3f", width="medium"),
            "decision": st.column_config.TextColumn("Decision", width="medium"),
        },
    )
    selected_rows = event.selection.rows
    selected = view.iloc[selected_rows[0] if selected_rows else 0]
    st.download_button(
        "Download filtered queue as CSV",
        data=view.drop(columns=["alert_index"]).to_csv(index=False).encode("utf-8"),
        file_name="safescan-findings.csv",
        mime="text/csv",
    )

    st.markdown("### Finding detail")
    st.caption("Select a row to inspect it. The top finding is shown by default.")
    code_col, detail_col = st.columns([1.45, 1], gap="large")
    with code_col:
        st.markdown(f"**{selected['rule']} at line {int(selected['line'])}**  ")
        st.caption(selected["rule_name"])
        snippet = ui.format_snippet(selected["code"])
        st.markdown(
            ui.code_block(snippet, highlight=highlight_line(selected["code"], int(selected["line"]))),
            unsafe_allow_html=True,
        )
    with detail_col:
        st.metric("Model score", f"{selected['p_true_positive']:.3f}", selected["decision"])
        st.caption("Estimated similarity to true positive patterns in the training data.")
        st.markdown("**Context considered**")
        for reason in ui.explain_alert(selected):
            st.markdown(f"- {reason}")
        active = [item for item in ui.CONTEXT_COLUMNS if int(selected.get(item, 0))]
        st.caption(
            "Context signals: " + ", ".join(item.removeprefix("ctx_").replace("_", " ") for item in active)
            if active
            else "No AST context signal fired. The score relies on the rule and flagged code text."
        )

with st.expander("View scanned source"):
    st.code(scan["preview_text"], language="python", line_numbers=True)

ui.footer()
