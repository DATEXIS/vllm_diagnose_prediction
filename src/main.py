"""MERLIN 2 entry point.

Top-level orchestration only: load data, run Loop A, evaluate, run Loop B.
Each phase lives in its own module so this file reads like a table of
contents:

  * src/loop_a.py  — Loop A: data, pipeline, run, evaluate, persist scores
  * src/loop_b.py  — Loop B: Meta-Verifier audit, persist new instructions
  * src/config.py  — config loading, logging setup
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import pandas as pd

from src.config import load_config, setup_logging
from src.loop_a import (
    build_pipeline,
    build_prediction_df,
    download_artifacts,
    evaluate_and_log,
    load_data,
    run_loop_a,
    save_efficacy_scores,
)
from src.loop_b import run_loop_b
from src.utils import wandb_logger

logger = logging.getLogger(__name__)


# ------------------------------------------------------------- entry point

async def main_async(config: dict) -> None:
    _init_wandb(config)

    df = load_data(config)
    download_artifacts(config)
    pipeline, existing = build_pipeline(config)

    results = await run_loop_a(pipeline, df)

    if _should_save_efficacy(df, config):
        save_efficacy_scores(pipeline, config)

    df = build_prediction_df(df, results, config)
    df_results = evaluate_and_log(df, results, config)

    await _maybe_run_loop_b(df, df_results, existing, config)

    wandb_logger.finish_wandb()


def _init_wandb(config: dict) -> None:
    wandb_logger.init_wandb(config)
    wandb_logger.log_parameters(config)
    wandb_logger.log_experiment_config(config)


def _should_save_efficacy(df: pd.DataFrame, config: dict) -> bool:
    """In-memory efficacy updates are persisted only when ground truth was available."""
    return "true_codes" in df.columns and config.get("merlin2", {}).get("update_efficacy", True)


async def _maybe_run_loop_b(df, df_results, existing, config: dict) -> None:
    mv_cfg = config.get("meta_verifier", {})
    if not mv_cfg.get("enabled", False):
        return
    if df_results is None:
        logger.warning(
            "Meta-Verifier is enabled but skipped: ground truth unavailable. "
            "Check that '%s' column exists in the data file.",
            config["data"].get("target_col", "ICD_CODES"),
        )
        return
    await run_loop_b(df, df_results, existing, config)


# ----------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="MERLIN 2 inference + Meta-Verifier")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
