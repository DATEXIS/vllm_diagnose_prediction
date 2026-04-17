import json
import logging
import ast
from typing import List, Set, Any, Optional, Dict

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from prompter import ICDsModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)

import re

def safe_parse_json(text: str) -> List[str]:
    """Tries to parse the LLM output as JSON and extract the ICD codes with robust repair logic."""
    if not text:
        return []
    
    # Pre-processing: Strip whitespace and potentially find the JSON block
    original_text = text.strip()

    def try_parse(candidate: str) -> Optional[List[str]]:
        try:
            # First attempt: Pydantic's direct JSON validation
            validated = ICDsModel.model_validate_json(candidate)
            return [d.icd_code for d in validated.diagnoses if d.icd_code]
        except (ValidationError, ValueError, json.JSONDecodeError):
            try:
                # Second attempt: Repair common LLM mistakes (literal newlines/tabs)
                repaired = candidate.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                validated = ICDsModel.model_validate_json(repaired)
                return [d.icd_code for d in validated.diagnoses if d.icd_code]
            except (ValidationError, ValueError, json.JSONDecodeError):
                # Third attempt: Handle truncation
                # Find the last completed diagnosis object '}' and close the structure
                if '"diagnoses": [' in candidate:
                    try:
                        # Find the last completed item in the diagnoses list
                        # Search for the last '}' that is not at the very end
                        last_brace = candidate.rstrip().rfind('}')
                        if last_brace != -1:
                            truncated_repair = candidate[:last_brace+1] + ']}'
                            # Re-run repair on the truncated version
                            repaired_truncated = truncated_repair.replace('\n', '\\n').replace('\t', '\\t')
                            validated = ICDsModel.model_validate_json(repaired_truncated)
                            return [d.icd_code for d in validated.diagnoses if d.icd_code]
                    except Exception:
                        pass
                return None

    # Step 1: Try parsing the whole thing
    codes = try_parse(original_text)
    
    # Step 2: If failed, try regex extraction
    if codes is None:
        # Match as much as possible starting from the first {
        match = re.search(r'\{(.*)', original_text, re.DOTALL)
        if match:
             codes = try_parse(match.group())

    # Step 3: Handle the result
    if codes is not None:
        return codes
    
    # Final fallback for logging
    logger.warning(f"Failed to parse or validate JSON after all repair attempts. Preview (1000ch):\n{original_text[:1000]}")
    return []

def safe_parse_true_labels(val: Any) -> List[str]:
    """Parses ground truth labels if they are stored as strings, lists, or arrays."""
    # Check if it's already list-like (avoids ValueError in pd.isna)
    if isinstance(val, list):
        return [str(i) for i in val]
    
    # Handle NaN/None cases
    if pd.isna(val) if isinstance(val, (str, float, int)) else False:
        return []
    
    # Fallback check for other null-like objects
    if val is None:
        return []
    
    # If it's a string, try to evaluate it
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # Fallback: maybe it's comma separated
            return [l.strip() for l in str(val).split(',')]
    
    return []

def calculate_metrics(y_true: List[List[str]], y_pred: List[List[str]]):
    """Calculates evaluation metrics for multi-label classification."""
    
    # We need to binarize labels for sklearn
    all_labels = set()
    for true_list, pred_list in zip(y_true, y_pred):
        all_labels.update(true_list)
        all_labels.update(pred_list)
        
    all_labels_list = list(all_labels)
    
    # Binarize
    y_true_bin = []
    y_pred_bin = []
    
    for true_list, pred_list in zip(y_true, y_pred):
        true_row = [1 if label in true_list else 0 for label in all_labels_list]
        pred_row = [1 if label in pred_list else 0 for label in all_labels_list]
        
        y_true_bin.append(true_row)
        y_pred_bin.append(pred_row)
        
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='micro', zero_division=0
    )
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='macro', zero_division=0
    )

    results = {
        'micro': {'precision': p_micro, 'recall': r_micro, 'f1': f1_micro},
        'macro': {'precision': p_macro, 'recall': r_macro, 'f1': f1_macro}
    }
    
    return results

def evaluate_predictions(df: pd.DataFrame, target_col: str, predictions: List[str]):
    """Main evaluation pipeline."""
    logger.info("Starting evaluation...")
    
    y_pred_lists = [safe_parse_json(pred) for pred in predictions]
    y_true_lists = [safe_parse_true_labels(val) for val in df[target_col].tolist()]
    
    metrics = calculate_metrics(y_true_lists, y_pred_lists)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Micro F1: {metrics['micro']['f1']:.4f} (P: {metrics['micro']['precision']:.4f}, R: {metrics['micro']['recall']:.4f})")
    logger.info(f"Macro F1: {metrics['macro']['f1']:.4f} (P: {metrics['macro']['precision']:.4f}, R: {metrics['macro']['recall']:.4f})")
    
    # Store predictions back in DF
    df['raw_predictions'] = predictions
    df['parsed_predictions'] = y_pred_lists
    
    return metrics, df
