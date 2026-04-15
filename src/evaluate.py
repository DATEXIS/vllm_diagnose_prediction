import json
import logging
import ast
from typing import List, Set

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

def safe_parse_json(text: str) -> List[str]:
    """Tries to parse the LLM output as JSON and extract the ICD codes."""
    if not text:
        return []
    try:
        data = json.loads(text)
        # Using the schema from prompter.py: {"diagnoses": [{"icd_code": "...", "reason": "..."}]}
        if "diagnoses" in data:
             return [d.get("icd_code") for d in data["diagnoses"] if d.get("icd_code")]
        return []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {text[:50]}...")
        return []

def safe_parse_true_labels(label_str: str) -> List[str]:
    """Parses ground truth labels if they are stored as strings of lists."""
    if pd.isna(label_str):
        return []
    if isinstance(label_str, list):
        return label_str
    try:
        return ast.literal_eval(label_str)
    except (ValueError, SyntaxError):
        # Fallback: maybe it's comma separated
        return [l.strip() for l in str(label_str).split(',')]

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
