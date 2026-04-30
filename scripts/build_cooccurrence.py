"""Build the 3-digit ICD co-occurrence matrix used by the FN retrieval path.

For each pair of 3-digit ICD codes (a, b) appearing together in a training
case's ground-truth label set, compute:

    joint_count(a, b)  - number of training cases containing both
    count(a)           - number of training cases containing a
    lift(a, b)         - N * joint_count(a,b) / (count(a) * count(b))

Only pairs with `joint_count >= min_joint_count` are emitted, and only
codes with `count(c) >= min_support` are kept (rare codes give noisy
lift). The output is stored both directions (a->b AND b->a) so retrieval
can do an O(1) lookup keyed on the predicted code.

Reads the training file path from `configs/experiment.yaml` (`data.patients_file`)
but ignores `sample_size` — the matrix is always built over the full
training set. Output path comes from `merlin2.cooccurrence_path`.

Usage:
    python scripts/build_cooccurrence.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import ast
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

import pandas as pd
import yaml
from utils import load_config

# from src.data.evaluate import normalize_icd, safe_parse_true_labels

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


def _load_full_train_codes(file_path: str, target_col: str) -> List[List[str]]:
    """Read the training parquet and return one list of 3-digit codes per case.

    `sample_size` is intentionally ignored here — the co-occurrence statistics
    must be computed over the full training set, not a sample.
    """
    if file_path.endswith(".pq") or file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not in {file_path}. Got: {list(df.columns)}"
        )
    cases: List[List[str]] = []
    for raw in df[target_col]:
        codes = {normalize_icd(c) for c in safe_parse_true_labels(raw)}
        codes.discard("")
        if codes:
            cases.append(sorted(codes))
    return cases


def build_cooccurrence(
    cases: List[List[str]],
    min_joint_count: int,
    min_support: int,
) -> pd.DataFrame:
    """Compute lift-based co-occurrence DataFrame from a list of code-sets.

    Returns a long-format DataFrame with columns:
      code_a, code_b, joint_count, lift, count_a, count_b
    Both directions (a,b) and (b,a) are emitted so retrieval can key on
    `code_a == predicted_code`.
    """
    n_cases = len(cases)
    if n_cases == 0:
        raise ValueError("No training cases with at least one valid 3-digit code.")

    code_count: Counter = Counter()
    pair_count: Dict[tuple, int] = defaultdict(int)
    for codes in cases:
        for c in codes:
            code_count[c] += 1
        # combinations() yields each unordered pair exactly once.
        for a, b in combinations(codes, 2):
            pair_count[(a, b)] += 1

    # Drop low-support codes — their marginals are too noisy for stable lift.
    valid_codes = {c for c, n in code_count.items() if n >= min_support}
    logger.info(
        f"Cases={n_cases}, unique 3-digit codes={len(code_count)}, "
        f"codes >= min_support({min_support})={len(valid_codes)}"
    )

    rows = []
    for (a, b), joint in pair_count.items():
        if joint < min_joint_count:
            continue
        if a not in valid_codes or b not in valid_codes:
            continue
        ca, cb = code_count[a], code_count[b]
        # lift = P(a,b) / (P(a)*P(b)) = N * joint / (ca * cb)
        lift = (n_cases * joint) / (ca * cb)
        rows.append(
            {"code_a": a, "code_b": b, "joint_count": joint, "lift": lift,
             "count_a": ca, "count_b": cb}
        )
        rows.append(
            {"code_a": b, "code_b": a, "joint_count": joint, "lift": lift,
             "count_a": cb, "count_b": ca}
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No co-occurrence pairs survived min_joint_count={min_joint_count} "
            f"and min_support={min_support}. Lower the thresholds."
        )
    df = df.sort_values(["code_a", "lift"], ascending=[True, False]).reset_index(drop=True)
    return df


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config()
    data_cfg = config.get("data", {})
    m2_cfg = config.get("merlin2", {})

    out_path = Path(m2_cfg.get("cooccurrence_path", "data/cooccurrence.parquet"))
    min_joint_count = int(m2_cfg.get("cooccurrence_min_joint_count", 3))
    min_support = int(m2_cfg.get("min_support", 3))

    file_path = data_cfg["patients_file"]
    target_col = data_cfg.get("target_col", "ICD_CODES")

    logger.info(f"Reading training labels from {file_path} (full set, no sampling)")
    cases = _load_full_train_codes(file_path, target_col)

    logger.info(
        f"Building co-occurrence: min_joint_count={min_joint_count}, "
        f"min_support={min_support}"
    )
    df = build_cooccurrence(cases, min_joint_count=min_joint_count, min_support=min_support)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} co-occurrence rows ({df['code_a'].nunique()} keys) to {out_path}")


if __name__ == "__main__":
    main()
