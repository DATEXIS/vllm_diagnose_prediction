"""Build per-admission-note rareness_factor for efficacy score weighting.

For each training case, computes a scalar rareness_factor ∈ (0, ∞) that
reflects how rare / difficult the ground-truth ICD codes are relative to the
rest of the training population.  Cases dominated by rare codes get a factor
> 1.0; common-code-heavy cases get < 1.0.  The mean over all training cases
is exactly 1.0, so the learning rate's magnitude is unchanged on average.

Algorithm
---------
1. Count 3-digit code frequencies over the full training set.
2. rareness(code) = N / count(code)
   Codes with count < min_support are capped at N / min_support so that
   singleton codes don't produce extreme outliers.
3. raw_factor(case) = agg(rareness(c) for c in case_codes)
   agg is one of: mean (default) | max | geom_mean
   Configured via merlin2.rareness_aggregation in experiment.yaml.
4. rareness_factor = raw_factor / mean(raw_factor over all training cases)

Cases with no valid codes get rareness_factor = 1.0 (neutral weight).

Reads config from configs/experiment.yaml.
Writes the rareness_factor column back into data.patients_file in-place
(overwrites an existing column if present).

Usage
-----
    python scripts/build_rareness_factors.py --config configs/experiment.yaml
"""

from __future__ import annotations

import ast
import logging
from collections import Counter
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers

def normalize_icd(code: Any) -> str:
    if not code:
        return ""
    return str(code).strip().upper().replace(".", "")[:3]


def safe_parse_true_labels(val: Any) -> List[str]:
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


def _aggregate(values: List[float], method: str) -> float:
    """Aggregate a list of per-code rareness values into one scalar.

    mean          — average rareness; ignores how many rare codes are present.
                    A single extremely rare code beats three moderately rare ones.
    sum           — multiple rare codes compound; scales fully with code count.
                    Common codes contribute little (rareness ≈ 1.0 post-norm).
    sqrt_mean     — mean * sqrt(n_codes).  Middle ground: rewards having several
                    rare codes without fully scaling with total code count.
                    Equivalent to sum / sqrt(n), so a 4× code count gives 2× factor.
    max           — dominated by the single rarest code; ignores all others.
    geom_mean     — smooth compromise between mean and max.
    """
    if not values:
        return 1.0
    if method == "mean":
        return float(np.mean(values))
    if method == "sum":
        return float(np.sum(values))
    if method == "sqrt_mean":
        return float(np.mean(values) * np.sqrt(len(values)))
    if method == "max":
        return float(np.max(values))
    if method == "geom_mean":
        return float(np.exp(np.mean(np.log(values))))
    raise ValueError(
        f"Unknown rareness_aggregation: {method!r}. "
        "Use mean | sum | sqrt_mean | max | geom_mean."
    )


# ------------------------------------------------------------------ core

def build_rareness_factors(
    df: pd.DataFrame,
    target_col: str,
    min_support: int,
    aggregation: str,
) -> pd.Series:
    """Return a Series of rareness_factor values aligned to df's index.

    Steps:
      1. Count 3-digit code frequencies across all rows (full file, no sampling).
      2. Compute per-code rareness = N / count, capped at N / min_support.
      3. Aggregate per case, then normalise to mean = 1.0.
    """
    # --- step 1: code counts over full file
    code_counts: Counter = Counter()
    all_case_codes: List[List[str]] = []
    for raw in df[target_col]:
        codes = [normalize_icd(c) for c in safe_parse_true_labels(raw)]
        codes = [c for c in codes if c]
        all_case_codes.append(codes)
        code_counts.update(codes)

    n_cases = len(df)
    n_unique = len(code_counts)
    logger.info(
        "Loaded %d cases, %d unique 3-digit codes", n_cases, n_unique
    )

    # --- step 2: per-code rareness (capped)
    cap = n_cases / max(min_support, 1)
    code_rareness: dict[str, float] = {
        code: min(n_cases / count, cap)
        for code, count in code_counts.items()
    }

    # --- step 3: per-case raw factor
    raw_factors: List[float] = []
    for codes in all_case_codes:
        values = [code_rareness[c] for c in codes if c in code_rareness]
        raw_factors.append(_aggregate(values, aggregation))

    raw_arr = np.asarray(raw_factors, dtype=np.float64)

    # --- step 4: normalise to mean = 1.0
    global_mean = float(raw_arr.mean())
    if global_mean == 0:
        raise ValueError("Global mean of raw rareness factors is 0 — check your data.")
    rareness_factors = raw_arr / global_mean

    logger.info(
        "rareness_factor stats — min=%.3f  mean=%.3f  max=%.3f  std=%.3f",
        rareness_factors.min(),
        rareness_factors.mean(),
        rareness_factors.max(),
        rareness_factors.std(),
    )

    return pd.Series(rareness_factors, index=df.index, name="rareness_factor")


# ------------------------------------------------------------------ entrypoint

def main() -> None:
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build per-case rareness_factor column.")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment YAML config (default: configs/experiment.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    m2_cfg = config.get("merlin2", {})

    file_path = data_cfg["patients_file"]
    target_col = data_cfg.get("target_col", "ICD_CODES")
    min_support = int(m2_cfg.get("min_support", 3))
    aggregation = m2_cfg.get("rareness_aggregation", "sum")

    logger.info(
        "Config: file=%s  target_col=%s  min_support=%d  aggregation=%s",
        file_path, target_col, min_support, aggregation,
    )

    p = Path(file_path)
    if p.suffix in (".pq", ".parquet"):
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        logger.error("Unsupported file format: %s", file_path)
        sys.exit(1)

    if target_col not in df.columns:
        logger.error(
            "Target column '%s' not found. Available columns: %s",
            target_col, list(df.columns),
        )
        sys.exit(1)

    if "rareness_factor" in df.columns:
        logger.info("Overwriting existing rareness_factor column.")

    df["rareness_factor"] = build_rareness_factors(
        df, target_col=target_col, min_support=min_support, aggregation=aggregation
    )

    if p.suffix in (".pq", ".parquet"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)

    logger.info("Wrote rareness_factor to %s", file_path)


if __name__ == "__main__":
    main()
