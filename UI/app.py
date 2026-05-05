from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEFAULT_RUN_ID = "2bzc4wvd"
DEFAULT_RUN_PATH = f"datexis-phd/ICD-prediction/{DEFAULT_RUN_ID}"
APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = APP_ROOT / "wandb_cache"
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


def _render_patient_view(row: pd.Series) -> None:
    st.subheader("Loop A: Patient Iterations")

    gt_codes = _coerce_list(row.get("true_codes"))
    pred_codes = _coerce_list(row.get("parsed_predictions"))
    halt_reason = row.get("halt_reason", "")
    iterations = int(row.get("iterations", 0) or 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Iterations", iterations)
    c2.metric("Ground Truth Codes", len(gt_codes))
    c3.metric("Final Predicted Codes", len(pred_codes))
    st.caption(f"Verifier halt reason: `{halt_reason}`")

    with st.expander("Admission note", expanded=False):
        st.text(row.get("admission_note", ""))

    with st.expander("Ground truth vs final prediction", expanded=True):
        st.write("**Ground truth (3-digit):**", ", ".join(map(str, gt_codes)) or "(none)")
        st.write("**Predicted (3-digit):**", ", ".join(map(str, pred_codes)) or "(none)")

    parsed_iters = _parse_think_block(str(row.get("think_block_final", "")))
    if parsed_iters:
        for it in parsed_iters:
            with st.container(border=True):
                st.markdown(f"**Iteration t={it['iteration']}**")
                st.write("Prediction:", ", ".join(it["prediction"]) or "(none)")
                if it["instructions"]:
                    st.write("Retrieved instructions:")
                    for ins in it["instructions"]:
                        st.markdown(f"- {ins}")
                else:
                    st.write("Retrieved instructions: none")
    else:
        st.warning("Could not parse iteration trace from think block for this patient.")

    with st.expander("Raw generator response"):
        st.text(str(row.get("raw_response", "")))


def _render_loop_b(summary: dict[str, Any], meta_df: pd.DataFrame) -> None:
    st.subheader("Loop B: Meta-Verifier Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meta instructions rows", int(summary.get("meta_verifier_instructions", {}).get("nrows", 0)))
    c2.metric("Final iteration", int(summary.get("iteration", 0)))
    c3.metric("Final micro F1", f"{float(summary.get('f1_micro', 0.0)):.4f}")

    keep_cols = [
        "instruction_id",
        "type",
        "target_codes",
        "instruction_text",
        "efficacy_score",
        "source_hadm_ids",
    ]
    show_cols = [c for c in keep_cols if c in meta_df.columns]
    top_df = meta_df.copy()
    if "efficacy_score" in top_df.columns:
        top_df["efficacy_score"] = pd.to_numeric(top_df["efficacy_score"], errors="coerce").fillna(0.0)
        top_df = top_df.sort_values("efficacy_score", ascending=False)
    st.write("Top instruction rows")
    st.dataframe(top_df[show_cols].head(50), use_container_width=True)


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
    meta_table = _find_file(run_dir, "*meta_verifier_instructions*.table.json")
    summary_path = _find_file(run_dir, "wandb-summary.json")
    output_log = _find_file(run_dir, "output.log")

    sample_df = _load_table_json(str(sample_table))
    meta_df = _load_table_json(str(meta_table))
    summary = json.loads(summary_path.read_text())

    patient_count = len(sample_df)
    max_default = min(20, patient_count)
    visible_n = st.sidebar.slider("Visible patient pool", min_value=1, max_value=patient_count, value=max_default)
    index = st.sidebar.number_input("Patient index", min_value=1, max_value=visible_n, value=1, step=1)
    st.caption(f"Active run directory: `{run_dir}`")

    row = sample_df.iloc[int(index) - 1]
    _render_patient_view(row)

    st.divider()
    _render_loop_b(summary, meta_df)

    history_csv = run_dir / "history_metrics.csv"
    _render_run_level(history_csv, output_log, summary)


if __name__ == "__main__":
    main()
