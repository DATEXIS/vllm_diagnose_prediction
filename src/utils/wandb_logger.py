"""Wandb logging helpers.

Research-code style: no defensive guards. If WANDB_API_KEY is unset or
init fails, the pipeline crashes — that is the desired behavior.

Exception: transient HTTP 429 (rate-limit) errors from the wandb API are
retried with exponential backoff before crashing. A single burst of parallel
job starts can trigger 429s even with a valid key; a short wait resolves it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb
import yaml

from src.merlin2.retriever import RetrievalEvent
from src.meta_verifier.schemas import Instruction

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


# ----------------------------------------------------------------- init
def init_wandb(config: Dict[str, Any]) -> None:
    """Initialize a wandb run from the experiment config.

    Reads `wandb.project` / `wandb.entity` from the config. The API key
    must be set in the WANDB_API_KEY env var.

    Retries up to 5 times on HTTP 429 with exponential backoff (30 s, 60 s,
    120 s, 240 s). Any other exception propagates immediately.
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not set in environment.")
    wandb_cfg = config.get("wandb", {}) or {}
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            wandb.init(
                project=wandb_cfg.get("project", "ICD-prediction"),
                entity=wandb_cfg.get("entity"),
                name=config.get("run_name", config.get("job_name", "default")),
                config=config,
            )
            logger.info(f"Wandb run: {wandb.run.name}")
            return
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_attempts:
                wait = 30 * (2 ** (attempt - 1))  # 30 s, 60 s, 120 s, 240 s
                logger.warning(
                    f"Wandb init rate-limited (attempt {attempt}/{max_attempts}), "
                    f"retrying in {wait} s: {exc}"
                )
                time.sleep(wait)
            else:
                raise


def finish_wandb() -> None:
    wandb.finish()


# --------------------------------------------------------------- logging
def log_parameters(config: Dict[str, Any]) -> None:
    inf = config.get("inference", {})
    model = config.get("model", {})
    data = config.get("data", {})
    merlin2 = config.get("merlin2", {})
    wandb.config.update(
        {
            "model_name": model.get("name"),
            "max_model_len": model.get("max_model_len"),
            "temperature": inf.get("temperature"),
            "max_tokens": inf.get("max_tokens"),
            "concurrency": inf.get("concurrency"),
            "guided_decoding": inf.get("guided_decoding"),
            "sample_size": data.get("sample_size"),
            "merlin2.sim_note_threshold": merlin2.get("sim_note_threshold"),
            "merlin2.sim_icd_threshold": merlin2.get("sim_icd_threshold"),
            "merlin2.fpr_threshold": merlin2.get("fpr_threshold"),
            "merlin2.fnr_threshold": merlin2.get("fnr_threshold"),
            "merlin2.dedup_cluster_threshold": merlin2.get("dedup_cluster_threshold"),
            "merlin2.max_instructions_per_code": merlin2.get("max_instructions_per_code"),
            "merlin2.convergence_threshold": merlin2.get("convergence_threshold"),
            "merlin2.max_iterations": merlin2.get("max_iterations"),
            "merlin2.max_tokens_budget": merlin2.get("max_tokens_budget"),
            "merlin2.per_iteration_token_budget": merlin2.get("per_iteration_token_budget"),
            "merlin2.min_support": merlin2.get("min_support"),
        },
        allow_val_change=True,
    )


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log run-level summary metrics.

    Uses wandb.summary so these values always reflect the final state of the
    run (all samples' final predictions) and are not a time-series step that
    could be confused with per-iteration iter/* metrics.
    """
    wandb.summary.update(
        {
            "f1_micro": metrics["micro"]["f1"],
            "f1_macro": metrics["macro"]["f1"],
            "precision_micro": metrics["micro"]["precision"],
            "recall_micro": metrics["micro"]["recall"],
            "precision_macro": metrics["macro"]["precision"],
            "recall_macro": metrics["macro"]["recall"],
            "valid_json_pct": metrics.get("valid_json_pct", 0.0),
        }
    )


def log_per_iteration_metrics(per_iter: List[Dict[str, Any]]) -> None:
    """`per_iter` is a list of flat metric dicts, one per iteration t.

    Keys logged: iter/f1_micro, iter/f1_macro, iter/precision_micro, ...
    """
    for t, entry in enumerate(per_iter):
        log_dict: Dict[str, Any] = {"iteration": t}
        for k, v in entry.items():
            log_dict[f"iter/{k}"] = v
        wandb.log(log_dict)


def log_experiment_config(config: Dict[str, Any]) -> None:
    """Log the full experiment config as a rendered YAML block (once per run)."""
    yaml_text = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    wandb.log({"experiment_yaml": wandb.Html(f"<pre style='font-size:12px'>{yaml_text}</pre>")})


def log_sample_table(df: pd.DataFrame, config: dict, n_samples: int = 30) -> None:
    """Log a small sample table for debugging. Strings only; no nested objects.

    Drops verbose / redundant columns:
      - hadm_id / subject_id / discharge_note: identifiers or long text
      - predictions: raw ICDsModel JSON — full_diagnoses and parsed_predictions
        carry the same data in a more readable form and are always derived from
        r.final_prediction, so logging the raw JSON would be redundant.
      - admission_note: too long for a table cell; available in the data file.
      - ICD_CODES / true_labels (original target column): already normalised
        into true_codes by the pipeline.
    """
    log_df = df.drop(
        columns=['subject_id', 'discharge_note', 'true_codes'],
        errors="ignore",
    )

    if config['inference'].get('guided_decoding'):
        log_df = log_df.drop(columns=['predictions'], errors="ignore")

    log_df = log_df.copy()

    if "full_diagnoses" in log_df.columns:
        log_df["full_diagnoses"] = log_df["full_diagnoses"].apply(
            lambda fd: json.dumps(fd, indent=2, ensure_ascii=False)
        )

    sample = log_df.head(n_samples).map(str)
    wandb.log({"sample_predictions": wandb.Table(dataframe=sample)})


def log_retrieval_type_pcts(events_df: pd.DataFrame) -> None:
    """Log % of each retrieval path type per iteration as wandb line-graph metrics.

    Semantic paths (sem_*) are collapsed into a single 'semantic' bucket so
    the chart stays comparable across runs before/after section-based chunking.
    Per-section breakdown is available in the retrieval_log column of the
    sample table.
    One wandb.log call per iteration so they plot cleanly on the same axes.
    """
    if events_df.empty:
        return
    for iteration, grp in events_df.groupby("iteration"):
        total = len(grp)
        counts = grp["path"].value_counts()
        sem_count = sum(v for k, v in counts.items() if k.startswith("sem_"))
        sem_icd_count = sum(v for k, v in counts.items() if k.startswith("sem_icd"))
        sem_count = sem_count - sem_icd_count
        wandb.log(
            {
                "retrieval_pct/semantic_note": sem_count / total * 100,
                "retrieval_pct/semantic_icd": sem_icd_count / total * 100,
                "retrieval_pct/threshold_fpr": counts.get("threshold_fpr", 0) / total * 100,
                "retrieval_pct/threshold_fnr": counts.get("threshold_fnr", 0) / total * 100,
                "iteration": int(iteration),
            }
        )


def log_icd_counts(
    y_true: List[List[str]], y_pred: List[List[str]]
) -> None:
    """Log avg true ICD count and avg predicted ICD count for the run."""
    avg_true = sum(len(t) for t in y_true) / len(y_true) if y_true else 0.0
    avg_pred = sum(len(p) for p in y_pred) / len(y_pred) if y_pred else 0.0
    wandb.log({"icd_count/avg_true": avg_true, "icd_count/avg_pred": avg_pred})


# --------------------------------------------------- generic parquet artifact
def _download_parquet_artifact(
    artifact_name: str,
    version: str,
    artifact_type: str,
    local_path: str,
) -> bool:
    """Pull a single-parquet artifact into `local_path`, overwriting any local copy.

    Returns True if downloaded, False if the artifact does not yet exist
    AND the caller asked for `:latest` (first-run bootstrap). A pinned
    version that fails to resolve, or auth/network errors, propagate.

    Must be called inside an active wandb run.
    """
    full_name = f"{artifact_name}:{version}"
    try:
        artifact = wandb.use_artifact(full_name, type=artifact_type)
    except wandb.errors.CommError:
        if version != "latest":
            raise
        logger.info(
            f"No '{artifact_type}' artifact '{full_name}' found in this project "
            f"— starting with empty {artifact_type} store."
        )
        return False

    art_dir = Path(artifact.download())
    parquet_files = list(art_dir.glob("*.parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 parquet inside artifact {full_name}, "
            f"got {len(parquet_files)}: {parquet_files}"
        )
    dst = Path(local_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parquet_files[0], dst)
    logger.info(
        f"Downloaded {artifact_type} artifact {full_name} "
        f"({artifact.version}) -> {dst}"
    )
    return True


def _log_parquet_artifact(
    artifact_name: str,
    artifact_type: str,
    local_path: str,
) -> None:
    """Log `local_path` as a new version of a single-parquet artifact."""
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cannot log {artifact_type} artifact: {p} does not exist."
        )
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(p))
    wandb.log_artifact(artifact)
    logger.info(f"Logged {artifact_type} artifact '{artifact_name}' from {p}")


# --------------------------------------------------------- instructions artifact
def download_instructions_artifact(
    artifact_name: str, version: str, local_path: str,
) -> bool:
    return _download_parquet_artifact(artifact_name, version, "instructions", local_path)


def log_instructions_artifact(artifact_name: str, local_path: str) -> None:
    _log_parquet_artifact(artifact_name, "instructions", local_path)


# --------------------------------------------------------- code-stats artifact
def download_code_stats_artifact(
    artifact_name: str, version: str, local_path: str,
) -> bool:
    """First-run bootstrap if `:latest` and not yet logged. Otherwise raises."""
    return _download_parquet_artifact(artifact_name, version, "code_stats", local_path)


def log_code_stats_artifact(artifact_name: str, local_path: str) -> None:
    _log_parquet_artifact(artifact_name, "code_stats", local_path)


def log_predictions_artifact(df: pd.DataFrame, artifact_name: str = "predictions") -> None:
    """Upload the full predictions dataframe as a parquet artifact.

    Columns with non-serialisable objects (e.g. list-of-dicts full_diagnoses)
    are JSON-encoded to strings so pyarrow can write them cleanly.
    """
    tmp_path = Path("/tmp") / f"{artifact_name}.parquet"
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], (list, dict)):
                out[col] = out[col].apply(json.dumps)
    out.to_parquet(tmp_path, index=False)
    artifact = wandb.Artifact(name=artifact_name, type="predictions")
    artifact.add_file(str(tmp_path))
    wandb.log_artifact(artifact)
    logger.info(f"Logged predictions artifact '{artifact_name}' ({len(df)} rows) from {tmp_path}")


def log_instruction_efficiency_tables(instructions: List[Instruction]) -> None:
    """Log top-30 and bottom-30 instructions by efficacy_score as wandb Tables.

    Called after efficacy scores are updated (end of Loop A, training only).
    Skips if the store is empty. Only persistent instructions are passed in
    (synthesised threshold warnings are never in the store).
    """
    if not instructions:
        return

    def _row(i: Instruction) -> dict:
        return {
            "instruction_id": i.instruction_id,
            "type": i.type,
            "action": i.action,
            "section": i.section,
            "target_codes": ",".join(i.target_codes),
            "efficacy_score": round(i.efficacy_score, 6),
            "description": i.description[:200],
            "instruction_text": i.instruction_text[:200],
            "source_hadm_ids": ",".join(i.source_hadm_ids),
        }

    sorted_by_eff = sorted(instructions, key=lambda x: x.efficacy_score, reverse=True)
    top30 = [_row(i) for i in sorted_by_eff[:30]]
    bottom30 = [_row(i) for i in sorted_by_eff[-30:]]

    wandb.log({
        "instructions/top30_by_efficacy": wandb.Table(dataframe=pd.DataFrame(top30)),
        "instructions/bottom30_by_efficacy": wandb.Table(dataframe=pd.DataFrame(bottom30)),
    })
    logger.info(
        "Logged top-30 / bottom-30 instruction efficiency tables "
        "(%d total instructions).", len(instructions)
    )


def log_instruction_confusion_matrix(tables: dict) -> None:
    """Log instruction retrieval quality tables to wandb.

    Expects the dict returned by ``instruction_eval.compute_all()``:
        'per_instruction', 'by_type', 'by_action', 'by_section',
        'correctness' (skipped — too large for a wandb Table),
        'fn_events'   (skipped — too large for a wandb Table).

    Tables logged under the ``instruction_eval/`` namespace.

    Summary scalars (mean over instructions with n_retrieved ≥ 1) are pushed
    to wandb.summary so they appear in the run comparison view:
        instruction_eval/mean_precision
        instruction_eval/mean_recall
        instruction_eval/mean_f1
        instruction_eval/mean_adoption_rate
        instruction_eval/mean_change_rate
        instruction_eval/n_instructions_evaluated
    """
    table_keys = {
        "per_instruction": "instruction_eval/per_instruction",
        "by_type":         "instruction_eval/by_type",
        "by_action":       "instruction_eval/by_action",
        "by_section":      "instruction_eval/by_section",
    }
    log_dict: Dict[str, Any] = {}
    for key, wandb_key in table_keys.items():
        df = tables.get(key)
        if df is not None and not df.empty:
            log_dict[wandb_key] = wandb.Table(dataframe=df.round(4))
    if log_dict:
        wandb.log(log_dict)

    # Summary scalars from per_instruction
    per_instr = tables.get("per_instruction")
    if per_instr is not None and not per_instr.empty:
        active = per_instr[per_instr["n_retrieved"] >= 1]
        summary: Dict[str, Any] = {
            "instruction_eval/n_instructions_evaluated": int(len(active)),
        }
        for col in ("precision", "recall", "f1", "adoption_rate", "change_rate"):
            if col in active.columns:
                valid = active[col].dropna()
                if not valid.empty:
                    summary[f"instruction_eval/mean_{col}"] = float(valid.mean())
        wandb.summary.update(summary)

    logger.info(
        "Logged instruction confusion matrix (%d tables, %d instructions).",
        len(log_dict),
        len(per_instr) if per_instr is not None else 0,
    )


def log_meta_verifier_instructions(instructions: List[Instruction]) -> None:
    """Log a snapshot of the new instruction batch."""
    rows = [
        {
            "instruction_id": i.instruction_id,
            "type": i.type,
            "target_codes": ",".join(i.target_codes),
            "instruction_text": i.instruction_text[:200],
            "description": i.description[:200],
            "fpr_at_creation": i.fpr_at_creation,
            "fnr_at_creation": i.fnr_at_creation,
            "efficacy_score": i.efficacy_score,
            "source_hadm_ids": ",".join(i.source_hadm_ids),
            "has_embedding": i.semantic_embedding is not None,
        }
        for i in instructions[:100]
    ]
    wandb.log({"meta_verifier_instructions": wandb.Table(dataframe=pd.DataFrame(rows))})
