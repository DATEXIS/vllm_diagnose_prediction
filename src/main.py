"""MERLIN 2 entry point.

Runs one Loop-A pass over a sample of patients, evaluates against
ground truth, and (when enabled) runs Loop-B (Meta-Verifier) to mint new
instructions. Instructions are persisted to a parquet store between
runs; the user re-runs this script to launch additional phases.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import List

import pandas as pd
import yaml

from src.data.data_loader import load_patients
from src.data.evaluate import evaluate_predictions, normalize_icd, safe_parse_true_labels
from src.merlin2.pipeline import MERLINPipeline, PipelineCaseResult
from src.meta_verifier.code_stats import (
    load_code_stats,
    merge_new_codes,
    save_code_stats,
)
from src.meta_verifier.meta_verifier import MetaVerifier
from src.meta_verifier.store import append_instructions, load_instructions
from src.utils import wandb_logger

logger = logging.getLogger(__name__)


def setup_logging(config: dict) -> None:
    level = config.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Make sure the dataframe has the columns the pipeline expects."""
    if "admission_note" not in df.columns:
        raise KeyError("Patient file is missing 'admission_note' column.")
    if "hadm_id" not in df.columns:
        df = df.copy()
        df["hadm_id"] = df.index.astype(str)
    if "rareness_factor" not in df.columns:
        df = df.copy()
        df["rareness_factor"] = 1.0
    if target_col in df.columns:
        df = df.copy()
        df["true_codes"] = df[target_col].apply(
            lambda v: [normalize_icd(c) for c in safe_parse_true_labels(v) if normalize_icd(c)]
        )
    return df


async def main_async(config: dict) -> None:
    wandb_logger.init_wandb(config)
    wandb_logger.log_parameters(config)

    target_col = config["data"].get("target_col", "ICD_CODES")
    df = _ensure_columns(load_patients(config), target_col)

    # ----------------------------------------------- artifact roundtrip (download)
    m2_cfg = config.get("merlin2", {})
    instructions_path = m2_cfg.get("instructions_path", "data/instructions.parquet")
    instr_artifact_name = m2_cfg.get("instructions_artifact_name", "instructions_db")
    instr_artifact_version = m2_cfg.get("instructions_artifact_version", "latest")
    code_stats_path = m2_cfg.get("code_stats_path", "data/code_stats.parquet")
    stats_artifact_name = m2_cfg.get("code_stats_artifact_name", "code_stats")
    stats_artifact_version = m2_cfg.get("code_stats_artifact_version", "latest")

    # wandb wins: if either artifact exists, overwrite the local parquet so the
    # run starts from the project's canonical state.
    wandb_logger.download_instructions_artifact(
        artifact_name=instr_artifact_name,
        version=instr_artifact_version,
        local_path=instructions_path,
    )
    wandb_logger.download_code_stats_artifact(
        artifact_name=stats_artifact_name,
        version=stats_artifact_version,
        local_path=code_stats_path,
    )

    # ------------------------------------------------------------------ Loop A
    # MERLINPipeline._build_retriever loads cooccurrence + code_stats from
    # the (freshly-downloaded) parquet files. Instructions are loaded
    # explicitly below because they need an embedding-cache rebuild.
    pipeline = MERLINPipeline(config)

    existing = load_instructions(instructions_path)
    if existing:
        pipeline.retriever.load_instructions(existing)
        logger.info(f"Loaded {len(existing)} existing instructions from {instructions_path}")

    ground_truth = df["true_codes"].tolist() if "true_codes" in df.columns else None
    results: List[PipelineCaseResult] = await pipeline.run(
        admission_notes=df["admission_note"].tolist(),
        hadm_ids=df["hadm_id"].astype(str).tolist(),
        ground_truth_codes=ground_truth,
        rareness_factors=df["rareness_factor"].tolist(),
    )

    df = df.copy()
    df["predictions"] = [r.final_prediction.model_dump_json() for r in results]
    df["raw_response"] = [r.final_raw_response for r in results]
    df["iterations"] = [r.iterations for r in results]
    df["halt_reason"] = [r.halt_reason for r in results]
    # The non-think-block parts of the prompt (system + admission note +
    # JSON example) are constant across iterations, so we only log the
    # final iteration's think block — that's where the per-iteration
    # variation actually lives.
    df["think_block"] = [
        r.history.think_blocks[-1] if r.history.think_blocks else "" for r in results
    ]
    # Per-iteration retrieval detail: which instructions fired, via which
    # path, at what confidence.  Replaces the old `instructions` column
    # (which was a raw list of think-block strings — hard to read in wandb).
    df["retrieval_log"] = [_format_retrieval_log(r) for r in results]
    # Reason texts from the final prediction — useful for spotting false
    # reasoning that an instruction could correct (and for checking whether
    # semantic_reason retrieval is firing on them).
    df["prediction_reasons"] = [
        " | ".join(
            f"{d.icd_code}: {d.reason[:80]}"
            for d in r.final_prediction.diagnoses
        )
        for r in results
    ]

    # ------------------------------------------------------------------ Eval
    df_results = None
    if target_col in df.columns:
        metrics, df_results = evaluate_predictions(df, target_col)
        wandb_logger.log_metrics(metrics)
        wandb_logger.log_sample_table(df_results, n_samples=30)

        # Per-iteration F1 trace (training only — when ground truth is present)
        if ground_truth is not None:
            per_iter = _per_iteration_metrics(results, ground_truth)
            if per_iter:
                wandb_logger.log_per_iteration_metrics(per_iter)

        # ICD count metrics
        wandb_logger.log_icd_counts(
            y_true=df["true_codes"].tolist(),
            y_pred=df_results["parsed_predictions"].tolist(),
        )

        # Retrieval-event % per path type (line graph)
        events_df = _flatten_retrieval_events(results)
        if not events_df.empty:
            wandb_logger.log_retrieval_type_pcts(events_df)

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
    else:
        logger.warning(f"Target column '{target_col}' not in data; skipping evaluation.")

    # ------------------------------------------------------------ Loop B
    mv_cfg = config.get("meta_verifier", {})
    if mv_cfg.get("enabled", False) and df_results is not None:
        df_for_audit = df_results.copy()
        df_for_audit["pred_codes"] = df_for_audit["parsed_predictions"].apply(
            lambda lst: [normalize_icd(c) for c in lst if normalize_icd(c)]
        )
        df_for_audit["true_codes"] = df["true_codes"]
        if "discharge_note" not in df_for_audit.columns:
            df_for_audit["discharge_note"] = ""
        if "hadm_id" not in df_for_audit.columns:
            df_for_audit["hadm_id"] = df["hadm_id"].astype(str)

        meta_verifier = MetaVerifier(config)
        starting_id = (
            max((i.instruction_id for i in existing), default=0) + 1 if existing else 1
        )
        audit_result = await meta_verifier.audit(df_for_audit, starting_id)

        # ---- Path 1: semantic instructions (instructions_db artifact)
        if audit_result.instructions:
            wandb_logger.log_meta_verifier_instructions(audit_result.instructions)
            append_instructions(audit_result.instructions, instructions_path)
            logger.info(
                f"Appended {len(audit_result.instructions)} new instructions "
                f"to {instructions_path}"
            )
            wandb_logger.log_instructions_artifact(
                artifact_name=instr_artifact_name,
                local_path=instructions_path,
            )
        else:
            logger.info("Meta-Verifier produced 0 new instructions.")

        # ---- Path 2: code-stats threshold rules (code_stats artifact)
        # Frozen-rate semantics: codes already in the table stay put,
        # only genuinely-new codes are appended.
        existing_stats = load_code_stats(code_stats_path)
        truly_new_stats = merge_new_codes(existing_stats, audit_result.new_code_stats)
        if truly_new_stats:
            merged = dict(existing_stats)
            merged.update(truly_new_stats)
            save_code_stats(merged, code_stats_path)
            logger.info(
                f"Appended {len(truly_new_stats)} new code-stat row(s) to "
                f"{code_stats_path}"
            )
            wandb_logger.log_code_stats_artifact(
                artifact_name=stats_artifact_name,
                local_path=code_stats_path,
            )
        else:
            logger.info("Meta-Verifier produced 0 new code-stat rows.")

    wandb_logger.finish_wandb()


def _per_iteration_metrics(
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> List[dict]:
    """Compute all eval metrics per iteration t for two settings.

    Returns a list of dicts, one per iteration t, each with two keys:
      - "all":       metrics over every sample that has a prediction at t
      - "last_iter": metrics over only the samples whose final iteration is t
                     (i.e. they halted/finished after this wave)

    At t=0 both settings are identical — every sample is in its first (and
    potentially last) iteration, so the "last_iter" subset equals "all".

    Metric keys per setting: f1_micro, f1_macro, precision_micro,
    recall_micro, precision_macro, recall_macro.
    """
    from src.data.evaluate import calculate_metrics

    def _metrics_dict(m: dict) -> dict:
        return {
            "f1_micro": m["micro"]["f1"],
            "f1_macro": m["macro"]["f1"],
            "precision_micro": m["micro"]["precision"],
            "recall_micro": m["micro"]["recall"],
            "precision_macro": m["macro"]["precision"],
            "recall_macro": m["macro"]["recall"],
        }

    max_iters = max(r.iterations for r in results) if results else 0
    out = []
    for t in range(max_iters):
        entry: dict = {}

        # Setting 1: all samples that have a prediction at iteration t
        y_pred_all, y_true_all = [], []
        parse_failures_at_t = 0
        for r, truth in zip(results, ground_truth):
            if t < len(r.history.predictions):
                y_pred_all.append(
                    [normalize_icd(d.icd_code) for d in r.history.predictions[t].diagnoses
                     if normalize_icd(d.icd_code)]
                )
                y_true_all.append([normalize_icd(c) for c in truth if normalize_icd(c)])
                # Count parse failures: case halted at t due to parse failure
                if (
                    r.halt_reason == "parse_failure"
                    and len(r.history.predictions) - 1 == t
                ):
                    parse_failures_at_t += 1
        if y_pred_all:
            m = _metrics_dict(calculate_metrics(y_true_all, y_pred_all))
            m["parse_failures"] = parse_failures_at_t
            m["n_samples"] = len(y_pred_all)
            entry["all"] = m

        # Setting 2: samples whose final iteration is t
        # r.iterations == len(r.history.predictions), so the last index is
        # r.iterations - 1; a sample "finishes at t" when that equals t.
        y_pred_last, y_true_last = [], []
        for r, truth in zip(results, ground_truth):
            if len(r.history.predictions) - 1 == t:
                y_pred_last.append(
                    [normalize_icd(d.icd_code) for d in r.history.predictions[t].diagnoses
                     if normalize_icd(d.icd_code)]
                )
                y_true_last.append([normalize_icd(c) for c in truth if normalize_icd(c)])
        if y_pred_last:
            entry["last_iter"] = _metrics_dict(calculate_metrics(y_true_last, y_pred_last))

        if entry:
            out.append(entry)
    return out


def _flatten_retrieval_events(results: List[PipelineCaseResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t, events in enumerate(r.history.retrieval_events):
            for ev in events:
                rows.append(
                    {
                        "hadm_id": r.hadm_id,
                        "iteration": t,
                        "instruction_id": ev.instruction_id,
                        "path": ev.path,
                        "trigger_value": ev.trigger_value,
                        "efficacy_score": ev.efficacy_score,
                        "target_codes": ",".join(ev.target_codes),
                        "trigger_codes": ",".join(ev.trigger_codes),
                    }
                )
    return pd.DataFrame(rows)


def _format_retrieval_log(result: PipelineCaseResult) -> str:
    """Build a structured retrieval log for wandb inspection.

    Format per iteration:

        True labels: A, B, C
        T=1:
        Pred: X, Y, Z

        Missed codes (FNR):
        * M33  fnr=1.00  co-occurs-with: Z82, K86
        * N70  fnr=1.00  co-occurs-with: N18

        Rethink codes (FPR):
        * B18  fpr=1.00

        Similar to note (semantic):
        * [E11]  sim=0.85  "If the note mentions long-standing DM2..."

        Similar to reasoning (semantic_reason):
        * [N18]  sim=0.83  "When CKD is mentioned alongside..."
    """
    from src.merlin2.retriever import SEMANTIC, SEMANTIC_REASON, THRESHOLD_FPR, THRESHOLD_FNR

    true_codes = result.history.ground_truth_codes or []
    true_str = ", ".join(sorted(true_codes)) if true_codes else "—"

    blocks: List[str] = [f"True labels: {true_str}"]

    for t, (events, instrs) in enumerate(
        zip(result.history.retrieval_events, result.history.instructions_used)
    ):
        if not events:
            continue

        # prediction that triggered this retrieval = prediction from t-1
        pred_codes: List[str] = []
        if t > 0 and len(result.history.predictions) >= t:
            pred_codes = [
                d.icd_code for d in result.history.predictions[t - 1].diagnoses
            ]
        pred_str = ", ".join(pred_codes) if pred_codes else "—"

        ev_by_id = {ev.instruction_id: ev for ev in events}

        fnr_lines: List[str] = []
        fpr_lines: List[str] = []
        sem_lines: List[str] = []
        sem_reason_lines: List[str] = []

        for instr in instrs:
            ev = ev_by_id.get(instr.instruction_id)
            if ev is None:
                continue
            codes_tag = ", ".join(ev.target_codes) if ev.target_codes else "?"
            snippet = (instr.instruction_text or "")[:90].replace("\n", " ")

            if ev.path == THRESHOLD_FNR:
                cooccur = (
                    f"  co-occurs-with: {', '.join(sorted(ev.trigger_codes))}"
                    if ev.trigger_codes else ""
                )
                fnr_lines.append(f"* {codes_tag:<6}  fnr={ev.trigger_value:.2f}{cooccur}")
            elif ev.path == THRESHOLD_FPR:
                fpr_lines.append(f"* {codes_tag:<6}  fpr={ev.trigger_value:.2f}")
            elif ev.path == SEMANTIC:
                sem_lines.append(f"* [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\"")
            elif ev.path == SEMANTIC_REASON:
                sem_reason_lines.append(f"* [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\"")

        section: List[str] = [f"\nT={t}:", f"Pred: {pred_str}"]
        if fnr_lines:
            section.append("\nMissed codes (FNR):")
            section.extend(fnr_lines)
        if fpr_lines:
            section.append("\nRethink codes (FPR):")
            section.extend(fpr_lines)
        if sem_lines:
            section.append("\nSimilar to note (semantic):")
            section.extend(sem_lines)
        if sem_reason_lines:
            section.append("\nSimilar to reasoning (semantic_reason):")
            section.extend(sem_reason_lines)
        blocks.append("\n".join(section))

    return "\n".join(blocks) if len(blocks) > 1 else "(no retrievals)"


def main() -> None:
    parser = argparse.ArgumentParser(description="MERLIN 2 inference + Meta-Verifier")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
