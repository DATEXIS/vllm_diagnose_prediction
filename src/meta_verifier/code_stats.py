"""Per-3-digit-code threshold-warning stats.

A simpler companion to the instruction parquet: instead of materialising
FP_WARNING / FN_WARNING `Instruction` rows in the main store, we keep a
small per-code table and let the Retriever synthesise the warning text
on demand at retrieval time.

Schema (one row per 3-digit code that has crossed at least one threshold):
    code              str
    fpr               float | None    -- frozen at creation
    fnr               float | None    -- frozen at creation
    support_pred      int             -- n cases where the code was predicted
    support_true      int             -- n cases where the code was in ground truth

Frozen-rate semantics (per MERLIN2_SPEC.md §3): once a code is in the
table, its rates are NEVER overwritten. Loop B can ADD new codes; codes
already present are left alone. This keeps successful instructions
firing even after the model has reduced their error rate.

No efficacy column: per the design decision, threshold warnings have no
per-code efficacy score (they fire whenever the gate triggers, fixed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CodeStat:
    code: str
    fpr: Optional[float]
    fnr: Optional[float]
    support_pred: int
    support_true: int


CodeStatsIndex = Dict[str, CodeStat]


COLUMNS = ["code", "fpr", "fnr", "support_pred", "support_true"]


def load_code_stats(path: str | Path) -> CodeStatsIndex:
    """Load the code-stats parquet into a per-code dict. Empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        logger.info(f"Code-stats file {p} not found; threshold path is dormant.")
        return {}
    df = pd.read_parquet(p)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Code-stats parquet {p} missing columns: {missing}")
    out: CodeStatsIndex = {}
    for row in df.itertuples(index=False):
        out[str(row.code)] = CodeStat(
            code=str(row.code),
            fpr=None if row.fpr is None or (isinstance(row.fpr, float) and np.isnan(row.fpr)) else float(row.fpr),
            fnr=None if row.fnr is None or (isinstance(row.fnr, float) and np.isnan(row.fnr)) else float(row.fnr),
            support_pred=int(row.support_pred),
            support_true=int(row.support_true),
        )
    logger.info(f"Loaded {len(out)} code-stats rows from {p}")
    return out


def save_code_stats(stats: CodeStatsIndex, path: str | Path) -> None:
    """Whole-file rewrite. Cheap at research scale (a few thousand rows)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "code": s.code,
                "fpr": s.fpr,
                "fnr": s.fnr,
                "support_pred": s.support_pred,
                "support_true": s.support_true,
            }
            for s in stats.values()
        ],
        columns=COLUMNS,
    )
    df.to_parquet(p, index=False)
    logger.info(f"Wrote {len(df)} code-stats rows to {p}")


def merge_new_codes(
    existing: CodeStatsIndex,
    candidates: CodeStatsIndex,
) -> Dict[str, CodeStat]:
    """Return only the candidates whose code is NOT already in `existing`.

    Codes already in `existing` are left untouched (frozen-rate semantics
    per MERLIN2_SPEC §3). Caller can then `existing.update(new_codes)` and
    save.
    """
    new = {c: s for c, s in candidates.items() if c not in existing}
    overlap = len(candidates) - len(new)
    if overlap:
        logger.info(
            f"merge_new_codes: skipping {overlap} candidate(s) already in code_stats "
            f"(frozen-rate semantics)."
        )
    return new
