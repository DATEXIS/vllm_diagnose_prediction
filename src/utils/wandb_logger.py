"""Wandb logging helpers.

Research-code style: no defensive guards. If WANDB_API_KEY is unset or
init fails, the pipeline crashes — that is the desired behavior.

Exception: transient HTTP 429 (rate-limit) errors from the wandb API are
retried with exponential backoff before crashing. A single burst of parallel
job starts can trigger 429s even with a valid key; a short wait resolves it.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb

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
            "merlin2.sim_threshold": merlin2.get("sim_threshold"),
            "merlin2.fpr_threshold": merlin2.get("fpr_threshold"),
            "merlin2.fnr_threshold": merlin2.get("fnr_threshold"),
            "merlin2.convergence_threshold": merlin2.get("convergence_threshold"),
            "merlin2.max_iterations": merlin2.get("max_iterations"),
            "merlin2.max_tokens_budget": merlin2.get("max_tokens_budget"),
            "merlin2.per_iteration_token_budget": merlin2.get("per_iteration_token_budget"),
            "merlin2.learning_rate": merlin2.get("learning_rate"),
            "merlin2.min_support": merlin2.get("min_support"),
        },
        allow_val_change=True,
    )


def log_metrics(metrics: Dict[str, Any]) -> None:
    wandb.log(
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
    """`per_iter` is a list of dicts with 'all' and 'last_iter' sub-dicts, one per t.

    Keys logged:
        iter/all/f1_micro, iter/all/f1_macro, iter/all/precision_micro, ...
        iter/last_iter/f1_micro, ...
    """
    for t, entry in enumerate(per_iter):
        log_dict: Dict[str, Any] = {"iteration": t}
        for setting, m in entry.items():
            for k, v in m.items():
                log_dict[f"iter/{setting}/{k}"] = v
        wandb.log(log_dict)


def log_sample_table(df: pd.DataFrame, n_samples: int = 30) -> None:
    """Log a small sample table for debugging. Strings only; no nested objects."""
    log_df = df.drop(columns=['hadm_id', 'subject_id', 'discharge_note'], errors="ignore")
    sample = log_df.head(n_samples).map(str)
    wandb.log({"sample_predictions": wandb.Table(dataframe=sample)})


def log_retrieval_type_pcts(events_df: pd.DataFrame) -> None:
    """Log % of each retrieval path type per iteration as wandb line-graph metrics.

    Paths: semantic, threshold_fpr, threshold_fnr.
    One wandb.log call per iteration so they plot cleanly on the same axes.
    """
    if events_df.empty:
        return
    paths = ["semantic", "semantic_reason", "threshold_fpr", "threshold_fnr"]
    for iteration, grp in events_df.groupby("iteration"):
        total = len(grp)
        counts = grp["path"].value_counts()
        wandb.log(
            {f"retrieval_pct/{p}": counts.get(p, 0) / total * 100 for p in paths}
            | {"iteration": int(iteration)}
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
        for i in instructions
    ]
    wandb.log({"meta_verifier_instructions": wandb.Table(dataframe=pd.DataFrame(rows))})
