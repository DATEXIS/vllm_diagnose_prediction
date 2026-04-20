import argparse
import asyncio
import logging
import yaml
import sys

from data_loader import load_patients
from prompter import build_prompts, get_schema
from inference import run_inference
from evaluate import evaluate_predictions
import wandb_logger

# Setup standard logging format
def setup_logging(config: dict):
    level = config.get('log_level', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
             return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

async def main_async(config: dict):
    # Load wandb config and initialize
    wandb_cfg = wandb_logger.load_wandb_config(config)
    wandb_initialized = wandb_logger.init_wandb(config, wandb_cfg)
    if wandb_initialized:
        wandb_logger.log_parameters(config)

    # 1. Load Data
    df = load_patients(config)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        if wandb_initialized:
            wandb_logger.finish_wandb()
        return

    # 2. Build Prompts
    prompts = build_prompts(df)

    # 3. Get Schema
    schema = get_schema() if config['inference'].get('guided_decoding', False) else None

    # 4. Run Inference
    predictions = await run_inference(config, prompts, schema)

    # 5. Evaluate
    target_col = config['data'].get('target_col', 'ICD_CODES')
    if target_col in df.columns:
        metrics, df_results = evaluate_predictions(df, target_col, predictions)

        if wandb_initialized:
            wandb_logger.log_metrics(metrics)
            wandb_logger.log_sample_table(df_results, predictions, metrics, n_samples=30)

        # Save results
        out_path = config['data'].get('patients_file', 'predictions').replace('.pq', '_predictions.csv').replace('.parquet', '_predictions.csv')
        if ".csv" not in out_path:
             out_path = "predictions.csv"

        df_results.to_csv(out_path, index=False)
        logger.info(f"Saved results and raw predictions to {out_path}")
    else:
        logger.warning(f"Target column '{target_col}' not found. Skipping evaluation.")

    if wandb_initialized:
        wandb_logger.finish_wandb()

def main():
    parser = argparse.ArgumentParser(description="vLLM Diagnose Prediction Client")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    asyncio.run(main_async(config))

if __name__ == "__main__":
    main()
