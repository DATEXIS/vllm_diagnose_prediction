import json
import logging
import ast
from typing import List, Set, Any, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from prompter import ICDsModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)

import re
from parsing_utils import safe_parse_json

def normalize_icd(code: Any) -> str:
    """
    Normalizes ICD codes for evaluation:
    1. Converts to string.
    2. Takes only the first 3 characters.
    3. Removes dots.
    Example: 'K86.0' -> 'K86', 'I10' -> 'I10'
    """
    if not code:
        return ""
    code_str = str(code).strip().upper()
    # Take first three characters and remove dots
    # Note: ICD-10 category is usually the first 3 chars (e.g. A00-B99)
    # We remove dots just in case they are in the first 3 (rare but possible in some formats)
    normalized = code_str.replace(".", "")
    return normalized[:3]

# safe_parse_json is now imported from parsing_utils.py


def safe_parse_true_labels(val: Any) -> List[str]:
    """Parses ground truth labels if they are stored as strings, lists, or arrays."""
    logger.debug(f"safe_parse_true_labels input type: {type(val)}, value: {repr(val)[:200]}")
    # Check if it's already list-like (avoids ValueError in pd.isna)
    if isinstance(val, list):
        return [str(i) for i in val]
    
    # Handle numpy arrays
    if isinstance(val, np.ndarray):
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
    
    valid_json_count = sum(1 for pred in y_pred_lists if pred)
    valid_json_pct = (valid_json_count / len(predictions)) * 100
    logger.info(f"Samples with valid JSON: {valid_json_count}/{len(predictions)} ({valid_json_pct:.1f}%)")
    
    # Debug: Log first few samples
    logger.debug(f"Sample predictions (raw): {y_pred_lists[:3]}")
    logger.debug(f"Sample ground truth (raw): {y_true_lists[:3]}")
    
    # Apply normalization to all codes
    y_pred_norm = [[normalize_icd(c) for c in clist if normalize_icd(c)] for clist in y_pred_lists]
    y_true_norm = [[normalize_icd(c) for c in clist if normalize_icd(c)] for clist in y_true_lists]
    
    logger.debug(f"Sample predictions (normalized): {y_pred_norm[:3]}")
    logger.debug(f"Sample ground truth (normalized): {y_true_norm[:3]}")
    
    metrics = calculate_metrics(y_true_norm, y_pred_norm)
    metrics['valid_json_pct'] = valid_json_pct
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Micro F1: {metrics['micro']['f1']:.4f} (P: {metrics['micro']['precision']:.4f}, R: {metrics['micro']['recall']:.4f})")
    logger.info(f"Macro F1: {metrics['macro']['f1']:.4f} (P: {metrics['macro']['precision']:.4f}, R: {metrics['macro']['recall']:.4f})")
    
    # Store predictions back in DF
    df['raw_predictions'] = predictions
    df['parsed_predictions'] = y_pred_lists
    
    return metrics, df
