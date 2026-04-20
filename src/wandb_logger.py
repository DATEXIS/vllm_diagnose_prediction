import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_wandb_initialized = False


def load_wandb_config(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Loads wandb configuration from main config or environment variable."""
    import os

    api_key = os.environ.get("WANDB_API_KEY")

    if not api_key and not config:
        logger.info("WANDB_API_KEY not set in environment and no config provided.")
        return None

    cfg = {"api_key": api_key} if api_key else {}

    if config and config.get("wandb"):
        cfg["project"] = config["wandb"].get("project", "ICD-prediction")
        cfg["entity"] = config["wandb"].get("entity")
    else:
        cfg["project"] = "ICD-prediction"

    if not api_key:
        logger.info("WANDB_API_KEY not set in environment. Wandb logging disabled.")
        return None

    return cfg


def init_wandb(config: dict, wandb_cfg: dict) -> bool:
    """Initialize wandb run."""
    global _wandb_initialized

    if not wandb_cfg:
        logger.info("Wandb logging disabled (no valid API key).")
        return False

    try:
        import wandb

        wandb.init(
            project=wandb_cfg.get('project', 'ICD-prediction'),
            entity=wandb_cfg.get('entity'),
            name=config.get('run_name', config.get('job_name', 'default')),
            config=config,
        )
        _wandb_initialized = True
        logger.info(f"Initialized wandb run: {wandb.run.name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")
        return False


def log_parameters(config: dict):
    """Log important parameters to wandb."""
    if not _wandb_initialized:
        return

    try:
        import wandb

        params = {
            "Model Name": config.get('model', {}).get('name'),
            "max_model_len": config.get('model', {}).get('max_model_len'),
            "temperature": config.get('inference', {}).get('temperature'),
            "max_tokens": config.get('inference', {}).get('max_tokens'),
            "concurrency": config.get('inference', {}).get('concurrency'),
            "guided_decoding": config.get('inference', {}).get('guided_decoding'),
            "sample_size": config.get('data', {}).get('sample_size'),
        }
        wandb.config.update(params, allow_val_change=True)
        logger.info("Logged parameters to wandb.")
    except Exception as e:
        logger.warning(f"Failed to log parameters to wandb: {e}")


def log_metrics(metrics: dict):
    """Log evaluation metrics to wandb."""
    if not _wandb_initialized:
        return

    try:
        import wandb

        wandb.log({
            "f1_micro": metrics['micro']['f1'],
            "f1_macro": metrics['macro']['f1'],
            "precision_micro": metrics['micro']['precision'],
            "recall_micro": metrics['micro']['recall'],
            "precision_macro": metrics['macro']['precision'],
            "recall_macro": metrics['macro']['recall'],
            "valid_json_pct": metrics.get('valid_json_pct', 0),
        })
        logger.info("Logged metrics to wandb.")
    except Exception as e:
        logger.warning(f"Failed to log metrics to wandb: {e}")


def _calculate_row_f1(true_codes: List[str], pred_codes: List[str]) -> float:
    """Calculate F1 for a single row."""
    if not true_codes and not pred_codes:
        return 1.0
    if not true_codes or not pred_codes:
        return 0.0

    true_set = set(true_codes)
    pred_set = set(pred_codes)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def log_sample_table(df: pd.DataFrame, predictions: List[str], metrics: dict, n_samples: int = 30):
    """Log detailed sample table to wandb."""
    if not _wandb_initialized:
        return

    try:
        import wandb

        from parsing_utils import safe_parse_json
        from evaluate import safe_parse_true_labels, normalize_icd

        target_col = df.columns[df.columns.str.lower().str.contains('icd')][0] if any(df.columns.str.lower().str.contains('icd')) else 'ICD_CODES'

        table_data = []
        for i, (_, row) in enumerate(df.head(n_samples).iterrows()):
            raw_response = predictions[i] if i < len(predictions) else ""
            parsed = safe_parse_json(raw_response) or {}
            pred_codes = [normalize_icd(c) for c in parsed] if isinstance(parsed, list) else []
            true_codes = [normalize_icd(c) for c in safe_parse_true_labels(row.get(target_col, []))]

            row_f1 = _calculate_row_f1(true_codes, pred_codes)

            subject_id = row.get('subject_id', '')
            hadm_id = row.get('hadm_id', '')
            admission_note = row.get('admission_note', '')[:500] if row.get('admission_note') else ''
            icd_codes = row.get(target_col, '')

            table_data.append({
                "subject_id": str(subject_id) if subject_id else "",
                "hadm_id": str(hadm_id) if hadm_id else "",
                "admission_note": admission_note,
                "ICD_CODES": str(icd_codes),
                "response": raw_response[:1000] if raw_response else "",
                "json": str(parsed)[:1000] if parsed else "",
                "predictions": str(pred_codes),
                "f1": round(row_f1, 4),
            })

        wandb.log({"sample_predictions": wandb.Table(dataframe=pd.DataFrame(table_data))})
        logger.info(f"Logged {len(table_data)} sample predictions to wandb.")
    except Exception as e:
        logger.warning(f"Failed to log sample table to wandb: {e}")


def finish_wandb():
    """Finish wandb run."""
    if not _wandb_initialized:
        return

    try:
        import wandb
        wandb.finish()
    except Exception as e:
        logger.warning(f"Failed to finish wandb: {e}")