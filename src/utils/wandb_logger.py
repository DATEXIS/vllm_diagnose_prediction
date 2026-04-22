import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml

from src.meta_verifier.schemas import Instruction

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

        from src.utils.parsing_utils import safe_parse_json
        from src.data.evaluate import safe_parse_true_labels, normalize_icd

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


def upload_experiment_results(
    df: pd.DataFrame,
    predictions: List[str],
    run_name: str,
    project: str = "ICD-prediction",
    entity: Optional[str] = None,
) -> bool:
    """Upload experiment results to wandb for later meta-verifier processing.
    
    Args:
        df: DataFrame with patient data, must contain: subject_id, hadm_id, admission_note, ICD_CODES
        predictions: List of model predictions (raw response strings)
        run_name: Name of the experiment run
        project: Wandb project name
        entity: Wandb entity/team name
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        import wandb
        from src.utils.parsing_utils import safe_parse_json
        from src.data.evaluate import safe_parse_true_labels, normalize_icd

        wandb.init(project=project, entity=entity, name=f"{run_name}_results", mode="offline")
        
        target_col = df.columns[df.columns.str.lower().str.contains('icd')][0] if any(df.columns.str.lower().str.contains('icd')) else 'ICD_CODES'
        
        table_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            raw_response = predictions[i] if i < len(predictions) else ""
            parsed = safe_parse_json(raw_response) or {}
            pred_codes = [normalize_icd(c) for c in parsed] if isinstance(parsed, list) else []
            true_codes = [normalize_icd(c) for c in safe_parse_true_labels(row.get(target_col, []))]
            
            row_f1 = _calculate_row_f1(true_codes, pred_codes)
            
            fp_codes = [c for c in pred_codes if c not in true_codes]
            fn_codes = [c for c in true_codes if c not in pred_codes]
            
            admission_note = row.get('admission_note', '') if row.get('admission_note') else ''
            
            table_data.append({
                "subject_id": str(row.get('subject_id', '')),
                "hadm_id": str(row.get('hadm_id', '')),
                "admission_note": admission_note,
                "ICD_CODES": str(row.get(target_col, '')),
                "response": raw_response,
                "pred_codes": pred_codes,
                "true_codes": true_codes,
                "fp_codes": fp_codes,
                "fn_codes": fn_codes,
                "f1": round(row_f1, 4),
            })
        
        result_df = pd.DataFrame(table_data)
        wandb.log({"experiment_results": wandb.Table(dataframe=result_df)})
        wandb.finish()
        
        logger.info(f"Uploaded experiment results for run '{run_name}' to wandb")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to upload experiment results to wandb: {e}")
        return False


def download_experiment_results(
    run_name: str,
    project: str = "ICD-prediction",
    entity: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download experiment results from wandb for meta-verifier processing.
    
    Args:
        run_name: Name of the experiment run to download
        project: Wandb project name
        entity: Wandb entity/team name
        
    Returns:
        DataFrame with experiment results or None if download fails
    """
    try:
        import wandb
        
        api = wandb.Api()
        
        filters = {"name": f"{run_name}_results"}
        runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
        
        if not runs:
            logger.warning(f"No run found with name '{run_name}_results'")
            return None
        
        run = runs[0]
        
        for file in run.files():
            if file.name == "experiment_results.table.json":
                file.download(replace=True)
                import json
                with open("experiment_results.table.json", "r") as f:
                    data = json.load(f)
                
                columns = [col["name"] for col in data["columns"]]
                rows = [[cell.get("v", cell.get("plain", "")) for cell in row["data"]] for row in data["data"]]
                
                result_df = pd.DataFrame(rows, columns=columns)
                
                import os
                os.remove("experiment_results.table.json")
                
                logger.info(f"Downloaded experiment results for run '{run_name}' ({len(result_df)} rows)")
                return result_df
        
        logger.warning(f"No experiment_results table found in run '{run_name}'")
        return None

    except Exception as e:
        logger.warning(f"Failed to download experiment results from wandb: {e}")
        return None


def upload_instructions(
    instructions: List[Instruction],
    run_name: str,
    project: str = "ICD-prediction",
    entity: Optional[str] = None,
) -> bool:
    """Upload generated instructions to wandb for later retrieval.

    Args:
        instructions: List of Instruction objects to upload
        run_name: Name of the experiment run
        project: Wandb project name
        entity: Wandb entity/team name

    Returns:
        True if upload successful, False otherwise
    """
    try:
        import wandb

        wandb.init(project=project, entity=entity, name=f"{run_name}_instructions", mode="offline")

        # Convert instructions to DataFrame for logging
        instruction_data = []
        for instr in instructions:
            instruction_data.append({
                "id": instr.id,
                "target_code": instr.target_code,
                "contrastive_rule": instr.contrastive_rule,
                "error_type": instr.error_type,
                "quote": instr.quote,
                "fpr": instr.fpr,
                "fnr": instr.fnr,
                "efficacy_score": instr.efficacy_score,
                "has_embedding": instr.semantic_embedding is not None,
            })

        instr_df = pd.DataFrame(instruction_data)
        wandb.log({"generated_instructions": wandb.Table(dataframe=instr_df)})

        # Also save the full instruction data as a JSON artifact
        import json
        from datetime import datetime

        # Prepare serializable instruction data
        serializable_instructions = []
        for instr in instructions:
            serializable_instructions.append({
                "id": instr.id,
                "target_code": instr.target_code,
                "contrastive_rule": instr.contrastive_rule,
                "error_type": instr.error_type,
                "quote": instr.quote,
                "fpr": instr.fpr,
                "fnr": instr.fnr,
                "efficacy_score": instr.efficacy_score,
                "semantic_embedding": instr.semantic_embedding,
                "created_at": instr.created_at.isoformat() if instr.created_at else None,
            })

        # Save to JSON file and upload
        json_path = f"instructions_{run_name}.json"
        with open(json_path, "w") as f:
            json.dump(serializable_instructions, f, indent=2)

        wandb.save(json_path)
        wandb.finish()

        logger.info(f"Uploaded {len(instructions)} instructions for run '{run_name}' to wandb")
        return True

    except Exception as e:
        logger.warning(f"Failed to upload instructions to wandb: {e}")
        return False


def download_instructions(
    run_name: str,
    project: str = "ICD-prediction",
    entity: Optional[str] = None,
) -> Optional[List[Instruction]]:
    """Download instructions from wandb for retrieval.

    Args:
        run_name: Name of the experiment run to download from
        project: Wandb project name
        entity: Wandb entity/team name

    Returns:
        List of Instruction objects or None if download fails
    """
    try:
        import wandb
        import json
        from datetime import datetime

        api = wandb.Api()

        # Search for runs with instructions
        filters = {"name": f"{run_name}_instructions"}
        runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

        if not runs:
            logger.warning(f"No run found with name '{run_name}_instructions'")
            return None

        run = runs[0]

        # Look for the JSON file with instructions
        json_filename = f"instructions_{run_name}.json"

        for file in run.files():
            if file.name == json_filename:
                file.download(replace=True)

                with open(json_filename, "r") as f:
                    data = json.load(f)

                instructions = []
                for item in data:
                    # Parse datetime if present
                    created_at = None
                    if item.get("created_at"):
                        created_at = datetime.fromisoformat(item["created_at"])

                    instructions.append(Instruction(
                        id=item["id"],
                        target_code=item["target_code"],
                        contrastive_rule=item["contrastive_rule"],
                        error_type=item.get("error_type", "unknown"),
                        quote=item.get("quote", ""),
                        fpr=item.get("fpr", 0.0),
                        fnr=item.get("fnr", 0.0),
                        efficacy_score=item.get("efficacy_score", 0.0),
                        semantic_embedding=item.get("semantic_embedding"),
                        created_at=created_at,
                    ))

                import os
                os.remove(json_filename)

                logger.info(f"Downloaded {len(instructions)} instructions from run '{run_name}'")
                return instructions

        logger.warning(f"No instructions file found in run '{run_name}'")
        return None

    except Exception as e:
        logger.warning(f"Failed to download instructions from wandb: {e}")
        return None