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
from pathlib import Path
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
from src.meta_verifier.store import (
    append_instructions,
    load_instructions,
    persist_efficacy_updates,
)
from src.utils.rareness import compute_rareness_factors
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


def _merge_rareness_factors(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach rareness_factor from a sidecar parquet or compute from labels."""
    m2_cfg = config.get("merlin2", {})
    sidecar = m2_cfg.get("rareness_factors_path")
    if sidecar:
        p = Path(sidecar)
        if p.exists():
            rf = pd.read_parquet(p)
            if "hadm_id" not in rf.columns or "rareness_factor" not in rf.columns:
                raise KeyError(f"{sidecar} must have columns hadm_id, rareness_factor")
            df = df.merge(rf[["hadm_id", "rareness_factor"]], on="hadm_id", how="left")
            missing = df["rareness_factor"].isna().sum()
            if missing:
                raise ValueError(
                    f"{missing} cases have no rareness_factor after merge with {sidecar}"
                )
            logger.info(f"Merged rareness_factor from {sidecar}")
            return df
        logger.warning(
            "merlin2.rareness_factors_path=%s not found; falling back to "
            "compute_rareness_at_load or 1.0. Build with: "
            "python scripts/build_rareness_factors.py --config configs/experiment.yaml",
            sidecar,
        )

    if m2_cfg.get("compute_rareness_at_load", False) and "true_codes" in df.columns:
        factors = compute_rareness_factors(df["true_codes"].tolist())
        df = df.copy()
        df["rareness_factor"] = factors
        logger.info(
            "Computed rareness_factor at load (mean=%.3f, max=%.3f)",
            sum(factors) / len(factors),
            max(factors),
        )
        return df

    if "rareness_factor" not in df.columns:
        df = df.copy()
        df["rareness_factor"] = 1.0
        logger.warning(
            "rareness_factor column missing; defaulting to 1.0. "
            "Run scripts/build_rareness_factors.py for tail-weighted efficacy."
        )
    return df


def _ensure_columns(df: pd.DataFrame, target_col: str, config: dict) -> pd.DataFrame:
    """Make sure the dataframe has the columns the pipeline expects."""
    if "admission_note" not in df.columns:
        raise KeyError("Patient file is missing 'admission_note' column.")
    if "hadm_id" not in df.columns:
        df = df.copy()
        df["hadm_id"] = df.index.astype(str)
    if target_col in df.columns:
        df = df.copy()
        df["true_codes"] = df[target_col].apply(
            lambda v: [normalize_icd(c) for c in safe_parse_true_labels(v) if normalize_icd(c)]
        )
    return _merge_rareness_factors(df, config)


async def main_async(config: dict) -> None:
    wandb_logger.init_wandb(config)
    wandb_logger.log_parameters(config)

    target_col = config["data"].get("target_col", "ICD_CODES")
    df = _ensure_columns(load_patients(config), target_col, config)

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
    if not config["inference"].get("guided_decoding", "true"):
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

    # Persist Loop-A efficacy updates (training only; requires instruction store).
    if ground_truth is not None and pipeline.retriever.persistent_instructions:
        efficacy_by_id = {
            i.instruction_id: i.efficacy_score
            for i in pipeline.retriever.persistent_instructions
        }
        n_updated = persist_efficacy_updates(efficacy_by_id, instructions_path)
        if n_updated:
            wandb_logger.log_instructions_artifact(
                artifact_name=instr_artifact_name,
                local_path=instructions_path,
            )

    events_df = _flatten_retrieval_events(results)
    if not events_df.empty:
        events_path = m2_cfg.get(
            "retrieval_events_path",
            "data/retrieval_events_last.csv",
        )
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(events_path, index=False)
        logger.info(f"Wrote {len(events_df)} retrieval events to {events_path}")
        wandb_logger.log_file_artifact("retrieval_events", "retrieval_log", events_path)

    # ------------------------------------------------------------------ Eval
    df_results = None
    if target_col in df.columns:
        metrics, df_results = evaluate_predictions(df, target_col)
        wandb_logger.log_metrics(metrics)
        wandb_logger.log_sample_table(df_results)


        # Per-iteration F1 trace (training only — when ground truth is present)
        if ground_truth is not None:
            per_iter = _per_iteration_metrics(results, ground_truth)
            if per_iter:
                wandb_logger.log_per_iteration_metrics(per_iter)

        # Final run-level summary metrics — logged after per-iteration metrics
        # so wandb summary reflects the true final-prediction quality across
        # all samples (not just those that reached the last iteration wave).
        wandb_logger.log_metrics(metrics)

        # ICD count metrics
        wandb_logger.log_icd_counts(
            y_true=df["true_codes"].tolist(),
            y_pred=df_results["parsed_predictions"].tolist(),
        )

        # Retrieval-event % per path type (line graph)
        if not events_df.empty:
            wandb_logger.log_retrieval_type_pcts(events_df)
            _log_retrieval_path_summary(events_df)

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
        wandb_logger.log_file_artifact("predictions", "predictions", out_path)
    else:
        logger.warning(f"Target column '{target_col}' not in data; skipping evaluation.")

    # ------------------------------------------------------------ Loop B
    mv_cfg = config.get("meta_verifier", {})
    if mv_cfg.get("enabled", False):
        if df_results is None:
            logger.warning(
                "Meta-Verifier is enabled but skipped: evaluation results are "
                "unavailable (ground truth required). Check that '%s' column "
                "exists in the data file.",
                target_col,
            )
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


def _prf(true_codes: List[str], pred_codes: List[str]) -> tuple:
    """Return (precision, recall, f1) for a single sample."""
    t, p = set(true_codes), set(pred_codes)
    if not t and not p:
        return 1.0, 1.0, 1.0
    tp = len(t & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _log_retrieval_path_summary(events_df: pd.DataFrame) -> None:
    """Print path mix so you can see if semantic vs threshold retrieval fires."""
    from src.merlin2.retriever import THRESHOLD_FPR, THRESHOLD_FNR, is_semantic_path

    def _bucket(path: str) -> str:
        if path == THRESHOLD_FPR:
            return "threshold_fpr"
        if path == THRESHOLD_FNR:
            return "threshold_fnr"
        if is_semantic_path(path):
            return "semantic"
        return "other"

    events_df = events_df.copy()
    events_df["bucket"] = events_df["path"].map(_bucket)
    total = len(events_df)
    by_bucket = events_df["bucket"].value_counts()
    logger.info("Retrieval path mix (%d events):", total)
    for bucket, count in by_bucket.items():
        logger.info("  %s: %.1f%% (%d)", bucket, 100.0 * count / total, count)
    if total > 0:
        sem_pct = 100.0 * by_bucket.get("semantic", 0) / total
        thr_pct = 100.0 * (
            by_bucket.get("threshold_fpr", 0) + by_bucket.get("threshold_fnr", 0)
        ) / total
        if sem_pct < 5 and thr_pct > 50:
            logger.warning(
                "Semantic retrieval is nearly idle (%.1f%%) while threshold "
                "paths dominate (%.1f%%). Qdrant/hybrid is unlikely to help until "
                "the instruction DB grows or sim_* thresholds are lowered.",
                sem_pct,
                thr_pct,
            )
        elif sem_pct > 50 and thr_pct < 5:
            logger.info(
                "Semantic paths carry most retrieval (%.1f%%); threshold gates "
                "are secondary (%.1f%%).",
                sem_pct,
                thr_pct,
            )


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

    Layout:
        True labels: A, B, C

        T=0 (zero-shot): X1, Y1  R=0.50  P=0.67  F1=0.57
        T=1: X2, Y2, Z2          R=0.67  P=0.80  F1=0.73
        T=2 (final): X3, Y3, Z3  R=0.67  P=0.80  F1=0.73

        --- Retrieved Instructions ---
        T=1:
          Missed codes (FNR):
          * M33    fnr=1.00  co-occurs-with: Z82, K86
          Rethink codes (FPR):
          * B18    fpr=1.00
          Semantic Similarity:
          * [E11]  [note]  sim=0.85  "If the note mentions long-standing DM2..."

    Predictions and their per-iteration scores appear first so you can scan
    the outcome without wading through instruction text. Retrieved instructions
    follow in a separate block.
    """
    from src.merlin2.retriever import THRESHOLD_FPR, THRESHOLD_FNR, is_semantic_path

    true_codes = result.history.ground_truth_codes or []
    true_str = ", ".join(sorted(true_codes)) if true_codes else "—"

    lines: List[str] = [f"True labels: {true_str}", ""]

    # ---- Part 1: all predictions with per-iteration P/R/F1 ----------------
    n_preds = len(result.history.predictions)
    for t, pred_model in enumerate(result.history.predictions):
        pred_codes = [
            normalize_icd(d.icd_code)
            for d in pred_model.diagnoses
            if normalize_icd(d.icd_code)
        ]
        pred_str = ", ".join(pred_codes) if pred_codes else "—"

        labels = []
        if t == 0:
            labels.append("zero-shot")
        if t == n_preds - 1 and n_preds > 1:
            labels.append("final")
        label_suffix = f" ({', '.join(labels)})" if labels else ""

        if true_codes:
            prec, rec, f1 = _prf(true_codes, pred_codes)
            score_str = f"  R={rec:.2f}  P={prec:.2f}  F1={f1:.2f}"
        else:
            score_str = ""

        lines.append(f"T={t}{label_suffix}: {pred_str}{score_str}")

    # ---- Part 2: retrieved instructions, grouped by iteration --------------
    retrieval_pairs = list(
        zip(result.history.retrieval_events, result.history.instructions_used)
    )
    has_retrievals = any(events for events, _ in retrieval_pairs)

    if has_retrievals:
        lines.append("\n--- Retrieved Instructions ---")

        for t, (events, instrs) in enumerate(retrieval_pairs):
            if not events:
                continue

            ev_by_id = {ev.instruction_id: ev for ev in events}
            fnr_lines: List[str] = []
            fpr_lines: List[str] = []
            sem_lines: List[str] = []

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
                    fnr_lines.append(
                        f"  * {codes_tag:<6}  fnr={ev.trigger_value:.2f}{cooccur}"
                    )
                elif ev.path == THRESHOLD_FPR:
                    fpr_lines.append(
                        f"  * {codes_tag:<6}  fpr={ev.trigger_value:.2f}"
                    )
                elif is_semantic_path(ev.path):
                    section_tag = ev.path.removeprefix("sem_")
                    sem_lines.append(
                        f"  * [{codes_tag}]  [{section_tag}]"
                        f"  sim={ev.trigger_value:.2f}  \"{snippet}\""
                    )

            lines.append(f"\nT={t}:")
            if fnr_lines:
                lines.append("  Missed codes (FNR):")
                lines.extend(fnr_lines)
            if fpr_lines:
                lines.append("  Rethink codes (FPR):")
                lines.extend(fpr_lines)
            if sem_lines:
                lines.append("  Semantic Similarity:")
                lines.extend(sem_lines)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MERLIN 2 inference + Meta-Verifier")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
