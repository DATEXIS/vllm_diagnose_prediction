"""Loop A driver: data loading, pipeline construction, evaluation, persistence.

`main.py` invokes these helpers in order:

  1. `load_data`            — read patients parquet, derive `true_codes`.
  2. `download_artifacts`   — pull instructions + code-stats from wandb.
  3. `build_pipeline`       — construct MERLINPipeline, load existing instructions.
  4. `run_loop_a`           — execute Loop A across the cohort.
  5. `save_efficacy_scores` — persist score updates to disk for Loop B.
  6. `build_prediction_df`  — attach per-case outputs to the data frame.
  7. `evaluate_and_log`     — compute metrics and push them to wandb.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from src.data.data_loader import load_patients
from src.data.evaluate import evaluate_predictions, normalize_icd, safe_parse_true_labels
from src.merlin2.instruction_eval import compute_all as compute_instruction_eval
from src.merlin2.pipeline import MERLINPipeline, PipelineCaseResult
from src.merlin2.reporting import (
    compute_metrics_at_t,
    flatten_retrieval_events,
    format_retrieval_log,
)
from src.meta_verifier.schemas import Instruction
from src.meta_verifier.store import load_instructions, save_instructions
from src.utils import wandb_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- data prep

def load_data(config: dict) -> pd.DataFrame:
    target_col = config["data"].get("target_col", "ICD_CODES")
    return _ensure_columns(load_patients(config), target_col)


def _ensure_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Validate required columns and derive true_codes when ground truth is present."""
    if "admission_note" not in df.columns:
        raise KeyError("Patient file is missing 'admission_note' column.")
    df = df.copy()
    if "hadm_id" not in df.columns:
        df["hadm_id"] = df.index.astype(str)
    if target_col in df.columns:
        df["true_codes"] = df[target_col].apply(
            lambda v: [normalize_icd(c) for c in safe_parse_true_labels(v) if normalize_icd(c)]
        )
    return df


def download_artifacts(config: dict) -> None:
    """Pull instructions and code-stats parquets from wandb; wandb wins over local."""
    m2_cfg = config.get("merlin2", {})
    wandb_logger.download_instructions_artifact(
        artifact_name=m2_cfg.get("instructions_artifact_name", "instructions_db"),
        version=m2_cfg.get("instructions_artifact_version", "latest"),
        local_path=m2_cfg.get("instructions_path", "data/instructions.parquet"),
    )
    wandb_logger.download_code_stats_artifact(
        artifact_name=m2_cfg.get("code_stats_artifact_name", "code_stats"),
        version=m2_cfg.get("code_stats_artifact_version", "latest"),
        local_path=m2_cfg.get("code_stats_path", "data/code_stats.parquet"),
    )


# ---------------------------------------------------------------- pipeline + run

def build_pipeline(config: dict) -> Tuple[MERLINPipeline, list]:
    """Construct the pipeline and load existing instructions into the retriever."""
    instructions_path = config.get("merlin2", {}).get("instructions_path", "data/instructions.parquet")
    pipeline = MERLINPipeline(config)
    existing = load_instructions(instructions_path)
    if existing:
        pipeline.retriever.load_instructions(existing)
        logger.info(f"Loaded {len(existing)} existing instructions from {instructions_path}")
    return pipeline, existing


async def run_loop_a(
    pipeline: MERLINPipeline, df: pd.DataFrame
) -> List[PipelineCaseResult]:
    has_ground_truth = "true_codes" in df.columns
    ground_truth = df["true_codes"].tolist() if has_ground_truth else None

    def _on_wave_end(states, t):
        metrics = compute_metrics_at_t(t, states)
        if metrics:
            wandb_logger.log_per_iteration_metrics([metrics], offset=t)

    return await pipeline.run(
        admission_notes=df["admission_note"].tolist(),
        hadm_ids=df["hadm_id"].astype(str).tolist(),
        ground_truth_codes=ground_truth,
        on_wave_end=_on_wave_end if has_ground_truth else None,
    )


def save_efficacy_scores(
    pipeline: MERLINPipeline,
    config: dict,
    eval_tables: Optional[dict] = None,
) -> None:
    """Update efficacy scores from instruction_eval, persist to parquet, push to wandb.

    Efficacy is now set post-hoc from the per-instruction confusion matrix
    produced by instruction_eval.compute_all() rather than online delta-F1.

    Score assignment (from per_instruction table):
      efficacy_score = f1 if not NaN, else precision
      Instructions not retrieved in this run keep their current score.

    Only persistent (semantic / contrastive) instructions are saved —
    synthesised threshold warnings are never written to parquet.
    """
    m2_cfg = config.get("merlin2", {})
    instructions_path = m2_cfg.get("instructions_path", "data/instructions.parquet")
    persistent = pipeline.retriever._instructions
    if not persistent:
        return

    # Apply post-hoc efficacy scores from instruction_eval.
    if eval_tables:
        per_instr = eval_tables.get("per_instruction")
        if per_instr is not None and not per_instr.empty:
            _apply_eval_scores(persistent, per_instr)

    save_instructions(persistent, instructions_path)
    logger.info(
        "Saved efficacy scores for %d instructions to %s",
        len(persistent), instructions_path,
    )
    wandb_logger.log_instructions_artifact(
        artifact_name=m2_cfg.get("instructions_artifact_name", "instructions_db"),
        local_path=instructions_path,
    )
    wandb_logger.log_instruction_efficiency_tables(persistent)


def _apply_eval_scores(
    instructions: List[Instruction],
    per_instr_df: "pd.DataFrame",
) -> None:
    """Write F1 (or precision) from per_instr_df back into each instruction object."""
    import math
    score_map: dict = {}
    for row in per_instr_df.itertuples(index=False):
        f1 = getattr(row, "f1", float("nan"))
        prec = getattr(row, "precision", float("nan"))
        score = f1 if (not math.isnan(f1)) else prec
        if not math.isnan(score):
            score_map[row.instruction_id] = float(score)

    for instr in instructions:
        if instr.instruction_id in score_map:
            instr.efficacy_score = score_map[instr.instruction_id]
            logger.debug(
                "efficacy_score updated: id=%d → %.4f", instr.instruction_id, instr.efficacy_score
            )


# ---------------------------------------------------------------- predictions df + evaluation

def build_prediction_df(
    df: pd.DataFrame, results: List[PipelineCaseResult], config: dict
) -> pd.DataFrame:
    """Attach per-case pipeline outputs to the data frame."""
    df = df.copy()
    _exclude = (
        {"instruction_reasoning"}
        if not config["inference"].get("instruction_reasoning", True)
        else set()
    )
    df["predictions"] = [r.final_prediction.model_dump_json(exclude=_exclude) for r in results]
    df["raw_response"] = [r.final_raw_response for r in results]
    if config["inference"].get("thinking", False):
        df["thinking"] = [r.final_thinking for r in results]
    df["iterations"] = [r.iterations for r in results]
    df["halt_reason"] = [r.halt_reason for r in results]
    def _extract_coding_review(raw: str) -> str:
        import re
        m = re.search(r"<coding_review>(.*?)</coding_review>", raw, re.DOTALL)
        return m.group(1).strip() if m else raw

    df["coding_review"] = [
        _extract_coding_review(r.history.coding_reviews[-1]) if r.history.coding_reviews else ""
        for r in results
    ]
    df["retrieval_log"] = [format_retrieval_log(r) for r in results]
    return df


def evaluate_and_log(
    df: pd.DataFrame,
    results: List[PipelineCaseResult],
    config: dict,
    instructions: Optional[List[Instruction]] = None,
) -> Optional[tuple]:
    """Run evaluation and log all metrics to wandb.

    Returns (df_results, eval_tables) when ground truth is present,
    None otherwise. eval_tables is the dict from instruction_eval.compute_all()
    and is also passed to save_efficacy_scores() for the post-hoc score update.
    """
    target_col = config["data"].get("target_col", "ICD_CODES")
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not in data; skipping evaluation.")
        return None

    metrics, df_results = evaluate_predictions(df, target_col)
    ground_truth = df["true_codes"].tolist()

    wandb_logger.log_sample_table(df_results, config, n_samples=30)
    wandb_logger.log_no_valid_json_examples(df_results)
    wandb_logger.log_predictions_artifact(df_results)
    wandb_logger.log_metrics(metrics)
    wandb_logger.log_icd_counts(
        y_true=ground_truth,
        y_pred=df_results["parsed_predictions"].tolist(),
    )
    eval_tables = _log_retrieval_breakdown(results, ground_truth, instructions)
    _save_predictions_csv(df_results, config)
    return df_results, eval_tables


def _log_retrieval_breakdown(
    results: List[PipelineCaseResult],
    ground_truth: Optional[List[List[str]]] = None,
    instructions: Optional[List[Instruction]] = None,
) -> dict:
    """Log retrieval path breakdown and instruction confusion matrix to wandb.

    Returns the eval_tables dict (from compute_instruction_eval) so callers
    can pass it to save_efficacy_scores for the post-hoc score update.
    Returns an empty dict when ground truth or instructions are absent.
    """
    events_df = flatten_retrieval_events(results)
    if not events_df.empty:
        wandb_logger.log_retrieval_type_pcts(events_df)

    if ground_truth and instructions:
        tables = compute_instruction_eval(results, ground_truth, instructions)
        wandb_logger.log_instruction_confusion_matrix(tables)
        return tables
    return {}


def _save_predictions_csv(df_results: pd.DataFrame, config: dict) -> None:
    out_path = (
        config["data"]
        .get("patients_file", "predictions")
        .replace(".pq", "_predictions.csv")
        .replace(".parquet", "_predictions.csv")
    )
    if ".csv" not in out_path:
        out_path = "predictions.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")
