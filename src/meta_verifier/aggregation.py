"""Per-3-digit FPR/FNR aggregation + code-stats row computation.

The Meta-Verifier's second error-discovery path scans the audited batch
for codes that crossed `fpr_threshold` or `fnr_threshold` (with at least
`min_support` cases) and emits one CodeStat row per such code. FP wins
when a code crosses both thresholds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.data.evaluate import normalize_icd
from src.meta_verifier.code_stats import CodeStat, CodeStatsIndex


def aggregate_fpr_fnr(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Compute per-3-digit FPR, FNR, prediction-support, ground-truth-support."""
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}
    pred_n: Dict[str, int] = {}
    true_n: Dict[str, int] = {}
    for _, row in df.iterrows():
        pred = {normalize_icd(c) for c in row["pred_codes"] if normalize_icd(c)}
        true = {normalize_icd(c) for c in row["true_codes"] if normalize_icd(c)}
        for c in pred - true:
            fp[c] = fp.get(c, 0) + 1
        for c in true - pred:
            fn[c] = fn.get(c, 0) + 1
        for c in pred:
            pred_n[c] = pred_n.get(c, 0) + 1
        for c in true:
            true_n[c] = true_n.get(c, 0) + 1
    fpr = {c: fp.get(c, 0) / pred_n[c] for c in pred_n}
    fnr = {c: fn.get(c, 0) / true_n[c] for c in true_n}
    return fpr, fnr, pred_n, true_n


def compute_threshold_code_stats(
    df: pd.DataFrame,
    fpr_threshold: float,
    fnr_threshold: float,
    min_support: int,
) -> CodeStatsIndex:
    """Return CodeStat rows for codes that crossed FPR or FNR thresholds.

    A code can only be on one side at a time (FP or FN). If it crosses
    both, FP wins (it appears in the predictions either way, so the
    retriever needs the FP gate active).
    """
    fpr, fnr, pred_n, true_n = aggregate_fpr_fnr(df)
    out: CodeStatsIndex = {}

    for code, rate in fpr.items():
        if pred_n.get(code, 0) < min_support or rate < fpr_threshold:
            continue
        out[code] = CodeStat(
            code=code, fpr=float(rate), fnr=None,
            support_pred=int(pred_n.get(code, 0)),
            support_true=int(true_n.get(code, 0)),
        )

    for code, rate in fnr.items():
        if true_n.get(code, 0) < min_support or rate < fnr_threshold:
            continue
        if code in out:
            # FP gate already active for this code; FN flag is redundant.
            continue
        out[code] = CodeStat(
            code=code, fpr=None, fnr=float(rate),
            support_pred=int(pred_n.get(code, 0)),
            support_true=int(true_n.get(code, 0)),
        )

    return out
