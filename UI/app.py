from __future__ import annotations

import ast
import json
import re
import shutil
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEFAULT_RUN_ID = "2bzc4wvd"
DEFAULT_RUN_PATH = f"datexis-phd/ICD-prediction/{DEFAULT_RUN_ID}"
APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = APP_ROOT / "wandb_cache"
ICD_NAMES_DIR = APP_ROOT / "ICD_names"
APP_CONFIG_PATH = APP_ROOT.parent / "config.json"

TABLE_SAMPLE_GLOB = "sample_predictions"
TABLE_META_GLOB = "meta_verifier_instructions"


def _load_wandb_api_key() -> str | None:
    if "WANDB_API_KEY" in st.secrets:
        return st.secrets["WANDB_API_KEY"]
    if APP_CONFIG_PATH.exists():
        cfg = json.loads(APP_CONFIG_PATH.read_text())
        return cfg.get("WANDB_API_KEY")
    return None


def _parse_run_path(run_input: str) -> str:
    text = run_input.strip()
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    if text.count("/") == 2:
        return text
    if "/" not in text and text:
        return f"datexis-phd/ICD-prediction/{text}"
    raise ValueError(f"Unsupported run input: {run_input}")


def _run_id_from_path(run_path: str) -> str:
    return run_path.split("/")[-1]


def _safe_delete(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _enforce_cache_limit(max_runs: int = 5) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in CACHE_ROOT.iterdir() if p.is_dir() and p.name != ".wandb_cache"]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in run_dirs[max_runs:]:
        _safe_delete(stale)


def _download_run_files(run_path: str, dest: Path) -> None:
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B client import failed. Fix with: `pip install -r UI/requirements.txt` "
            "(or explicitly `pip install protobuf==6.32.1`)."
        ) from exc

    key = _load_wandb_api_key()
    if key and not st.session_state.get("_wandb_key_set"):
        # One-time setup to avoid surfacing key in logs/UI.
        import os

        os.environ["WANDB_API_KEY"] = key
        st.session_state["_wandb_key_set"] = True

    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    dest.mkdir(parents=True, exist_ok=True)

    table_files: dict[str, str] = {}
    for f in run.files():
        if f.name.startswith("media/table/") and TABLE_SAMPLE_GLOB in f.name:
            table_files["sample"] = f.name
        elif f.name.startswith("media/table/") and TABLE_META_GLOB in f.name:
            table_files["meta"] = f.name

    required = [
        "output.log",
        "config.yaml",
        "wandb-summary.json",
        table_files.get("sample", ""),
        table_files.get("meta", ""),
    ]
    for rel in required:
        if not rel:
            continue
        run.file(rel).download(root=str(dest), replace=True)

    history_rows = list(run.scan_history())
    if history_rows:
        pd.DataFrame(history_rows).to_csv(dest / "history_metrics.csv", index=False)

    # Optional: try to fetch instructions + code_stats artifacts used by this run.
    artifacts_dir = dest / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for art in run.logged_artifacts():
        if "instructions_db" in art.name or "code_stats" in art.name:
            target = artifacts_dir / art.name.replace(":", "_")
            target.mkdir(parents=True, exist_ok=True)
            art.download(root=str(target))


@st.cache_data(show_spinner=False)
def _load_table_json(path: str) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text())
    cols = raw.get("columns", [])
    rows = raw.get("data", [])
    return pd.DataFrame(rows, columns=cols)


def _find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} in {root}")
    return matches[0]


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            return [value]
    return [value]


def _parse_think_block(think_block: str) -> list[dict[str, Any]]:
    if not think_block:
        return []
    body = think_block.replace("<think>", "").replace("</think>", "")
    pattern = re.compile(r"\[t=(\d+)\]\s*Prediction:\s*(.*?)(?=\n\s*\[t=|\Z)", re.DOTALL)
    out = []
    for m in pattern.finditer(body):
        t = int(m.group(1))
        segment = m.group(2).strip()
        lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
        prediction = []
        bullets = []
        if lines:
            prediction = [c.strip() for c in lines[0].split(",") if c.strip()]
        for ln in lines[1:]:
            if ln.startswith("-"):
                bullets.append(ln[1:].strip())
        out.append({"iteration": t, "prediction": prediction, "instructions": bullets})
    return out


def _extract_note_summary(note: str, max_lines: int = 10) -> str:
    lines = [ln.strip() for ln in note.splitlines() if ln.strip()]
    if not lines:
        return "(no admission note)"
    return "\n".join(lines[:max_lines])


def _normalize_code(code: Any) -> str:
    txt = str(code).strip().upper().replace(".", "")
    return txt[:3] if txt else ""


def _norm_icd_csv_code(raw: str, digits_only: bool) -> str:
    s = str(raw).strip().upper().replace(".", "").replace(" ", "")
    if digits_only:
        s = "".join(ch for ch in s if ch.isdigit())
    return s


def _category_map_from_pairs(pairs: list[tuple[str, str]], digits_only: bool) -> dict[str, str]:
    """Map 3-character category key -> short description."""
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for code, desc in pairs:
        if not code or not desc:
            continue
        if digits_only:
            d = "".join(ch for ch in code if ch.isdigit())
            if len(d) < 3:
                continue
            k3 = d[:3]
            buckets[k3].append((d, desc))
            continue
        if not code[0].isalpha() or len(code) < 3:
            continue
        k3 = code[:3]
        buckets[k3].append((code, desc))

    out: dict[str, str] = {}
    for k3, cands in buckets.items():
        end9 = [(c, d) for c, d in cands if c.endswith("9")]
        pool = end9 if end9 else cands
        _, best_desc = min(pool, key=lambda x: (len(x[0]), x[0]))
        out[k3] = best_desc
    return out


def _resolve_icd_description(
    norm: str,
    exact_10: dict[str, str],
    cat_10: dict[str, str],
    exact_9: dict[str, str],
    cat_9: dict[str, str],
) -> str | None:
    if not norm:
        return None
    if norm[0].isalpha():
        if norm in exact_10:
            return exact_10[norm]
        return cat_10.get(norm)
    if norm.isdigit():
        if norm in exact_9:
            return exact_9[norm]
        return cat_9.get(norm)
    return None


@st.cache_data(show_spinner=False)
def _icd_bundle() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Picklable ICD maps: (exact_10, cat_10, exact_9, cat_9)."""
    exact_10: dict[str, str] = {}
    exact_9: dict[str, str] = {}
    pairs_10: list[tuple[str, str]] = []
    pairs_9: list[tuple[str, str]] = []

    if not ICD_NAMES_DIR.is_dir():
        return ({}, {}, {}, {})

    for path in sorted(ICD_NAMES_DIR.glob("*.csv")):
        nl = path.name.lower()
        if "icd_codes_10" in nl:
            kind = "10"
        elif "icd_codes_9" in nl:
            kind = "9"
        else:
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        col_by_upper = {str(c).strip().upper(): c for c in df.columns}
        if kind == "10":
            code_col = col_by_upper.get("CODE")
            desc_col = col_by_upper.get("SHORT DESCRIPTION")
        else:
            code_col = col_by_upper.get("DIAGNOSIS CODE")
            desc_col = col_by_upper.get("SHORT DESCRIPTION")
        if code_col is None or desc_col is None:
            continue

        for _, row in df.iterrows():
            desc = str(row[desc_col]).strip()
            if not desc:
                continue
            if kind == "10":
                cn = _norm_icd_csv_code(row[code_col], digits_only=False)
                if len(cn) < 3 or not cn[0].isalpha():
                    continue
                exact_10[cn] = desc
                pairs_10.append((cn, desc))
            else:
                cn = _norm_icd_csv_code(row[code_col], digits_only=True)
                if len(cn) < 3:
                    continue
                exact_9[cn] = desc
                pairs_9.append((cn, desc))

    cat_10 = _category_map_from_pairs(pairs_10, digits_only=False)
    cat_9 = _category_map_from_pairs(pairs_9, digits_only=True)

    return (exact_10, cat_10, exact_9, cat_9)


def _format_code_line(code: str, bundle: tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]) -> str:
    e10, c10, e9, c9 = bundle
    name = _resolve_icd_description(code, e10, c10, e9, c9)
    if name:
        return f"{escape(code)} — {escape(name)}"
    return escape(code)


def _build_iteration_records(row: pd.Series) -> list[dict[str, Any]]:
    parsed = _parse_think_block(str(row.get("think_block_final", "")))
    if parsed:
        parsed.sort(key=lambda x: x["iteration"])
        return parsed

    fallback = [_normalize_code(c) for c in _coerce_list(row.get("parsed_predictions"))]
    fallback = [c for c in fallback if c]
    if fallback:
        return [{"iteration": 0, "prediction": fallback, "instructions": []}]
    return []


def _box(title: str, body_html: str, bg: str, height: int) -> None:
    st.markdown(
        f"""
<div style="background:{bg}; border-radius:8px; padding:10px; margin-bottom:8px; color:#111111;">
  <div style="font-weight:700; margin-bottom:6px; color:#000000;">{escape(title)}</div>
  <div style="max-height:{height}px; overflow-y:auto; white-space:pre-wrap; line-height:1.35; color:#000000;">
    {body_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_codes_list(
    codes: list[str],
    bundle: tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]],
) -> str:
    if not codes:
        return "(none)"
    return "<br>".join(_format_code_line(c, bundle) for c in codes)


def _render_instruction_list(instructions: list[str]) -> str:
    if not instructions:
        return "(no instructions yet)"
    return "<br>".join(f"- {escape(i)}" for i in instructions)


def _render_patient_view(row: pd.Series, pane_scale: str, custom_height: int) -> None:
    st.subheader("Loop A: Single-View Replay")
    icd_bundle = _icd_bundle()
    gt_codes = sorted({_normalize_code(c) for c in _coerce_list(row.get("true_codes")) if _normalize_code(c)})
    halt_reason = row.get("halt_reason", "")
    iterations = int(row.get("iterations", 0) or 0)
    iter_records = _build_iteration_records(row)
    if not iter_records:
        st.warning("No iteration records found for this patient.")
        return

    max_t = max(r["iteration"] for r in iter_records)
    current_t = st.slider("Iteration", min_value=0, max_value=max_t, value=max_t, step=1)
    by_t = {r["iteration"]: r for r in iter_records}
    selected = by_t.get(current_t, iter_records[-1])

    pred_codes = sorted({_normalize_code(c) for c in selected["prediction"] if _normalize_code(c)})
    pred_set, gt_set = set(pred_codes), set(gt_codes)
    tp_codes = sorted(pred_set & gt_set)
    fp_codes = sorted(pred_set - gt_set)
    fn_codes = sorted(gt_set - pred_set)

    cumulative_instructions: list[str] = []
    for t in range(0, current_t + 1):
        rec = by_t.get(t)
        if rec:
            # newest on top in the instruction window
            cumulative_instructions = rec["instructions"] + cumulative_instructions

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations", iterations)
    c2.metric("Current t", current_t)
    c3.metric("Ground Truth", len(gt_codes))
    c4.metric("Verifier halt", str(halt_reason))

    st.markdown(
        """
<div style="border:1px solid var(--secondary-background-color); border-radius:8px; padding:8px 12px; margin:10px 0 14px 0; background:var(--secondary-background-color);">
  <div style="font-weight:700; margin-bottom:4px;">Flow</div>
  <div style="font-size:0.95rem; color:var(--text-color);">
    Admission Note
    <span style="padding:0 8px;">→</span>
    Prediction Buckets (TP/FP/FN)
    <span style="padding:0 8px;">→</span>
    Retrieved Instructions
    <span style="padding:0 8px;">→</span>
    Next Iteration
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    note_full = str(row.get("admission_note", ""))
    show_full_note = st.toggle("Show full admission note", value=False)
    note_text = note_full if show_full_note else _extract_note_summary(note_full, max_lines=10)

    scale_map = {
        "Small": {"note": 280, "instr": 180, "tp": 90, "fp": 100, "fn": 100, "pred": 90},
        "Medium": {"note": 360, "instr": 250, "tp": 110, "fp": 130, "fn": 130, "pred": 110},
        "Large": {"note": 460, "instr": 340, "tp": 130, "fp": 170, "fn": 170, "pred": 130},
    }
    if pane_scale == "Custom":
        h = max(120, custom_height)
        heights = {
            "note": h,
            "instr": int(h * 0.7),
            "tp": int(h * 0.32),
            "fp": int(h * 0.38),
            "fn": int(h * 0.38),
            "pred": int(h * 0.32),
        }
    else:
        heights = scale_map.get(pane_scale, scale_map["Medium"])

    left, mid, right = st.columns([1.1, 0.14, 1.0])
    with left:
        _box("Admission Note", escape(note_text), "#dce7f8", heights["note"])
        st.markdown(
            "<div style='text-align:center; font-size:1.25rem; margin:2px 0 4px 0; color:var(--text-color);'>↑</div>",
            unsafe_allow_html=True,
        )
        _box(
            "All Instructions (cumulative, newest first)",
            _render_instruction_list(cumulative_instructions),
            "#fbf3ba",
            heights["instr"],
        )
        st.markdown(
            "<div style='text-align:center; font-size:0.88rem; color:var(--text-color); margin-top:-2px;'>Instructions feed back into the next iteration</div>",
            unsafe_allow_html=True,
        )

    with mid:
        st.markdown(
            """
<div style="height: 100%; display:flex; flex-direction:column; justify-content:space-around; align-items:center; color:var(--text-color);">
  <div style="font-size:1.35rem;">→</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        _box("True Predictions (TP)", _render_codes_list(tp_codes, icd_bundle), "#b9f0c8", heights["tp"])
        st.markdown("<div style='text-align:center; color:var(--text-color); margin:-2px 0 2px 0;'>↓</div>", unsafe_allow_html=True)
        _box("False Predictions (FP)", _render_codes_list(fp_codes, icd_bundle), "#ffb5b5", heights["fp"])
        st.markdown("<div style='text-align:center; color:var(--text-color); margin:-2px 0 2px 0;'>↓</div>", unsafe_allow_html=True)
        _box("Missed Predictions (FN)", _render_codes_list(fn_codes, icd_bundle), "#ffb5b5", heights["fn"])
        st.markdown("<div style='text-align:center; color:var(--text-color); margin:-2px 0 2px 0;'>↓</div>", unsafe_allow_html=True)
        _box("Predicted at current t", _render_codes_list(pred_codes, icd_bundle), "#ececec", heights["pred"])


def _render_run_level(history_csv: Path, output_log: Path, summary: dict[str, Any]) -> None:
    st.subheader("Run-level charts and logs")
    if history_csv.exists():
        hist_df = pd.read_csv(history_csv)
        chart_cols = [
            c
            for c in [
                "iter/all/f1_micro",
                "iter/all/f1_macro",
                "retrieval_pct/semantic",
                "retrieval_pct/threshold_fpr",
                "retrieval_pct/threshold_fnr",
                "iter/all/parse_failures",
            ]
            if c in hist_df.columns
        ]
        if chart_cols:
            st.line_chart(hist_df[chart_cols], height=260)

    with st.expander("Run summary (wandb-summary.json)"):
        st.json(summary)

    with st.expander("Output log tail"):
        lines = output_log.read_text(errors="ignore").splitlines()
        st.text("\n".join(lines[-200:]))


def _ensure_default_cached() -> Path:
    run_dir = CACHE_ROOT / DEFAULT_RUN_ID
    if not run_dir.exists():
        with st.spinner("Downloading default run files from W&B..."):
            _download_run_files(DEFAULT_RUN_PATH, run_dir)
        _enforce_cache_limit(5)
    return run_dir


def _load_run(run_path: str) -> Path:
    run_id = _run_id_from_path(run_path)
    run_dir = CACHE_ROOT / run_id
    if not run_dir.exists():
        with st.spinner(f"Downloading run {run_path}..."):
            _download_run_files(run_path, run_dir)
    run_dir.touch()
    _enforce_cache_limit(5)
    return run_dir


def main() -> None:
    st.set_page_config(page_title="MERLIN Run Replay", layout="wide")
    st.title("MERLIN2 Run Replay UI (MVP)")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = _ensure_default_cached()

    st.sidebar.header("Run Loader")
    run_input = st.sidebar.text_input(
        "W&B run URL or path",
        value=f"https://wandb.ai/datexis-phd/ICD-prediction/runs/{DEFAULT_RUN_ID}",
    )
    if st.sidebar.button("Load run", use_container_width=True):
        try:
            run_path = _parse_run_path(run_input)
            run_dir = _load_run(run_path)
            st.session_state["active_run_dir"] = str(run_dir)
            st.sidebar.success(f"Loaded {run_path}")
        except Exception as exc:
            st.sidebar.error(f"Failed to load run: {exc}")

    if "active_run_dir" in st.session_state:
        run_dir = Path(st.session_state["active_run_dir"])
    else:
        st.session_state["active_run_dir"] = str(run_dir)

    sample_table = _find_file(run_dir, "*sample_predictions*.table.json")
    summary_path = _find_file(run_dir, "wandb-summary.json")
    output_log = _find_file(run_dir, "output.log")

    sample_df = _load_table_json(str(sample_table))
    summary = json.loads(summary_path.read_text())

    patient_count = len(sample_df)
    max_default = min(20, patient_count)
    visible_n = st.sidebar.slider("Visible patient pool", min_value=1, max_value=patient_count, value=max_default)
    index = st.sidebar.number_input("Patient index", min_value=1, max_value=visible_n, value=1, step=1)
    pane_scale = st.sidebar.selectbox("Pane size", options=["Small", "Medium", "Large", "Custom"], index=1)
    custom_height = st.sidebar.slider("Custom base pane height (px)", min_value=180, max_value=700, value=380, step=20)
    st.caption(f"Active run directory: `{run_dir}`")

    row = sample_df.iloc[int(index) - 1]
    _render_patient_view(row, pane_scale, custom_height)
    with st.expander("Run-level diagnostics", expanded=False):
        history_csv = run_dir / "history_metrics.csv"
        _render_run_level(history_csv, output_log, summary)


if __name__ == "__main__":
    main()
