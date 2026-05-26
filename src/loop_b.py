"""Loop B driver: Meta-Verifier audit + new-instruction / code-stats persistence.

Runs only when ground truth is available and the Meta-Verifier is enabled
in the config. Two persistence paths:

  * `_persist_new_instructions`  — append fresh semantic instructions to
                                   `instructions.parquet` (Loop B output).
  * `_persist_new_code_stats`    — frozen-rate merge of new threshold-rate
                                   rows into `code_stats.parquet`.

Both upload the resulting parquet to wandb as a versioned artifact.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.evaluate import normalize_icd
from src.meta_verifier.code_stats import load_code_stats, merge_new_codes, save_code_stats
from src.meta_verifier.meta_verifier import MetaVerifier
from src.meta_verifier.store import append_instructions
from src.utils import wandb_logger

logger = logging.getLogger(__name__)


async def run_loop_b(
    df: pd.DataFrame,
    df_results: pd.DataFrame,
    existing: list,
    config: dict,
) -> None:
    """Run the Meta-Verifier audit and persist any new instructions and code stats."""
    m2_cfg = config.get("merlin2", {})
    instructions_path = m2_cfg.get("instructions_path", "data/instructions.parquet")
    code_stats_path = m2_cfg.get("code_stats_path", "data/code_stats.parquet")

    df_audit = _prepare_audit_df(df, df_results)
    starting_id = _next_instruction_id(existing)
    audit_result = await MetaVerifier(config).audit(df_audit, starting_id)

    _persist_new_instructions(
        audit_result,
        instructions_path,
        artifact_name=m2_cfg.get("instructions_artifact_name", "instructions_db"),
    )
    _persist_new_code_stats(
        audit_result,
        code_stats_path,
        artifact_name=m2_cfg.get("code_stats_artifact_name", "code_stats"),
    )


def _next_instruction_id(existing: list) -> int:
    return max((i.instruction_id for i in existing), default=0) + 1 if existing else 1


def _prepare_audit_df(df: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    """Build the audit dataframe the Meta-Verifier expects."""
    df_audit = df_results.copy()
    df_audit["pred_codes"] = df_audit["parsed_predictions"].apply(
        lambda lst: [normalize_icd(c) for c in lst if normalize_icd(c)]
    )
    df_audit["true_codes"] = df["true_codes"]
    if "discharge_note" not in df_audit.columns:
        df_audit["discharge_note"] = ""
    if "hadm_id" not in df_audit.columns:
        df_audit["hadm_id"] = df["hadm_id"].astype(str)
    return df_audit


def _persist_new_instructions(audit_result, instructions_path: str, artifact_name: str) -> None:
    if not audit_result.instructions:
        logger.info("Meta-Verifier produced 0 new instructions.")
        return
    wandb_logger.log_meta_verifier_instructions(audit_result.instructions)
    append_instructions(audit_result.instructions, instructions_path)
    logger.info(f"Appended {len(audit_result.instructions)} new instructions to {instructions_path}")
    wandb_logger.log_instructions_artifact(artifact_name=artifact_name, local_path=instructions_path)


def _persist_new_code_stats(audit_result, code_stats_path: str, artifact_name: str) -> None:
    """Append genuinely new code-stat rows (frozen-rate semantics: existing rows are untouched)."""
    existing_stats = load_code_stats(code_stats_path)
    truly_new = merge_new_codes(existing_stats, audit_result.new_code_stats)
    if not truly_new:
        logger.info("Meta-Verifier produced 0 new code-stat rows.")
        return
    save_code_stats({**existing_stats, **truly_new}, code_stats_path)
    logger.info(f"Appended {len(truly_new)} new code-stat row(s) to {code_stats_path}")
    wandb_logger.log_code_stats_artifact(artifact_name=artifact_name, local_path=code_stats_path)
