"""Evaluation utilities for ICD-code predictions.

All metrics are computed at the 3-digit ICD level (see `normalize_icd`).
The headline metrics are F1 micro / macro; precision/recall are reported
in the same dict for both averages.
"""

import ast
import logging
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.prompter import ICDsModel

logger = logging.getLogger(__name__)


def normalize_icd(code: Any) -> str:
    """Normalize an ICD code to its 3-digit category.

    Examples:
        'K86.0' -> 'K86'
        'I10'   -> 'I10'
        ''      -> ''
    """
    if not code:
        return ""
    code_str = str(code).strip().upper().replace(".", "")
    return code_str[:3]


def safe_parse_true_labels(val: Any) -> List[str]:
    """Parse ground-truth labels stored as list, ndarray, or stringified list."""
    if isinstance(val, list):
        return [str(i) for i in val]
    if isinstance(val, np.ndarray):
        return [str(i) for i in val]
    if val is None:
        return []
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return [str(i) for i in parsed]
            return [str(parsed)]
        except (ValueError, SyntaxError):
            return [s.strip() for s in val.split(",")]
    return []


def calculate_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> dict:
    """Compute multi-label P/R/F1 (micro and macro) over a label union."""
    all_labels = set()
    for true_list, pred_list in zip(y_true, y_pred):
        all_labels.update(true_list)
        all_labels.update(pred_list)
    all_labels_list = sorted(all_labels)

    y_true_bin, y_pred_bin = [], []
    for true_list, pred_list in zip(y_true, y_pred):
        y_true_bin.append([1 if label in true_list else 0 for label in all_labels_list])
        y_pred_bin.append([1 if label in pred_list else 0 for label in all_labels_list])

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    return {
        "micro": {"precision": p_micro, "recall": r_micro, "f1": f1_micro},
        "macro": {"precision": p_macro, "recall": r_macro, "f1": f1_macro},
    }


def _parse_prediction_row(serialized: str) -> List[dict]:
    """Parse a single row of the `predictions` column into a list of diagnosis dicts.

    The pipeline writes each row as `ICDsModel.model_dump_json()`. We
    validate it back; malformed rows raise (fail-fast).
    """
    model = ICDsModel.model_validate_json(serialized)
    return [{"icd_code": d.icd_code, "reason": d.reason} for d in model.diagnoses]


def evaluate_predictions(df: pd.DataFrame, target_col: str):
    logger.info("Starting evaluation...")
    predictions = df["predictions"].tolist()
    if len(predictions) == 0:
        raise ValueError("No predictions found in dataframe.")

    full_diagnoses = [_parse_prediction_row(p) for p in predictions]
    y_pred_lists = [[d["icd_code"] for d in row] for row in full_diagnoses]
    y_true_lists = [safe_parse_true_labels(v) for v in df[target_col].tolist()]

    valid_json_count = sum(1 for pred in y_pred_lists if pred)
    valid_json_pct = (valid_json_count / len(predictions))
    logger.info(
        f"Samples with non-empty predictions: {valid_json_count}/{len(predictions)} "
        f"({valid_json_pct:.1f}%)"
    )
    logger.info(f"Sample predictions (raw): {y_pred_lists[:3]}")
    logger.info(f"Sample ground truth (raw): {y_true_lists[:3]}")

    y_pred_norm = [[normalize_icd(c) for c in lst if normalize_icd(c)] for lst in y_pred_lists]
    y_true_norm = [[normalize_icd(c) for c in lst if normalize_icd(c)] for lst in y_true_lists]

    logger.info(f"Sample predictions (normalized): {y_pred_norm[:3]}")
    logger.info(f"Sample ground truth (normalized): {y_true_norm[:3]}")

    metrics = calculate_metrics(y_true_norm, y_pred_norm)
    metrics["valid_json_pct"] = valid_json_pct

    logger.info(
        f"Micro F1: {metrics['micro']['f1']:.4f} "
        f"(P: {metrics['micro']['precision']:.4f}, R: {metrics['micro']['recall']:.4f})"
    )
    logger.info(
        f"Macro F1: {metrics['macro']['f1']:.4f} "
        f"(P: {metrics['macro']['precision']:.4f}, R: {metrics['macro']['recall']:.4f})"
    )

    df = df.copy()
    df["parsed_predictions"] = y_pred_lists
    df["full_diagnoses"] = full_diagnoses
    return metrics, df
