import argparse
import asyncio
import logging
import yaml
import sys

from data_loader import load_patients
from prompter import build_prompts, get_schema
from inference import run_inference
from evaluate import evaluate_predictions

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
    # 1. Load Data
    df = load_patients(config)
    if df.empty:
        logger.error("No data loaded. Exiting.")
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
        
        # Save results
        out_path = config['data'].get('patients_file', 'predictions').replace('.pq', '_predictions.csv').replace('.parquet', '_predictions.csv')
        if ".csv" not in out_path:
             out_path = "predictions.csv"
             
        df_results.to_csv(out_path, index=False)
        logger.info(f"Saved results and raw predictions to {out_path}")
    else:
        logger.warning(f"Target column '{target_col}' not found. Skipping evaluation.")

def main():
    parser = argparse.ArgumentParser(description="vLLM Diagnose Prediction Client")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    asyncio.run(main_async(config))

if __name__ == "__main__":
    main()
