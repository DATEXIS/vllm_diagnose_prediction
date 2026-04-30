"""Cooccurrence index loader.

Reads the long-format parquet produced by `scripts/build_cooccurrence.py`
and reduces it to the per-key lookup the Retriever needs:

    {predicted_code: [(cooccurring_code, lift), ...]}

The returned dict is sorted by lift descending and capped at `top_k`
entries per key. A `lift_threshold` filter is also applied at load time —
only pairs at or above the threshold are kept. This means at retrieval
time the lookup is a plain dict get, no per-call filtering.

The matrix is treated as a frozen dataset property: once built, it is
not updated by Loop A or Loop B. If the training split changes, rebuild
with `scripts/build_cooccurrence.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CooccurrenceIndex = Dict[str, List[Tuple[str, float]]]


def load_cooccurrence_index(
    path: str | Path,
    lift_threshold: float,
    top_k: int,
) -> CooccurrenceIndex:
    """Load and reduce the parquet to a per-key dict.

    Returns an empty dict if the file does not exist (e.g. preprocessing
    has not been run yet on a fresh checkout). Caller decides whether
    that is acceptable — the Retriever simply won't fire FN warnings via
    the threshold path until the file appears.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(
            f"Cooccurrence file {p} not found. FN-via-cooccurrence retrieval "
            f"is disabled until you run scripts/build_cooccurrence.py."
        )
        return {}

    df = pd.read_parquet(p)
    required = {"code_a", "code_b", "lift"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Cooccurrence parquet {p} missing columns: {missing}")

    df = df[df["lift"] >= lift_threshold]
    if df.empty:
        logger.warning(
            f"Cooccurrence file {p} has zero pairs at lift >= {lift_threshold}. "
            f"FN-via-cooccurrence retrieval will not fire until the threshold is "
            f"lowered or the matrix is rebuilt."
        )
        return {}

    # Group by code_a, take top_k by lift descending.
    index: CooccurrenceIndex = {}
    for code_a, group in df.groupby("code_a"):
        ranked = group.sort_values("lift", ascending=False).head(top_k)
        index[str(code_a)] = [
            (str(row.code_b), float(row.lift)) for row in ranked.itertuples(index=False)
        ]
    logger.info(
        f"Loaded cooccurrence index from {p}: {len(index)} keys, "
        f"lift_threshold={lift_threshold}, top_k={top_k}"
    )
    return index


def expand_cooccurring(
    index: CooccurrenceIndex,
    predicted_codes: List[str],
) -> set[str]:
    """For a set of predicted codes, return the union of their top-K co-occurring codes.

    Excludes the predicted codes themselves — we only want codes that
    might be missing, not ones the model already named.
    """
    out: set[str] = set()
    predicted_set = set(predicted_codes)
    for code in predicted_set:
        for other, _lift in index.get(code, ()):
            if other not in predicted_set:
                out.add(other)
    return out
