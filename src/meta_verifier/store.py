"""Parquet-backed storage for MERLIN 2 instructions.

Schema mirrors `Instruction` (see schemas.py). Embeddings live as a
list-of-float column. Whole-file rewrite on save; that is fine for the
research scale (thousands of rows). For Loop A reads, instructions are
materialized in memory and embeddings stacked into a single numpy
matrix for brute-force cosine retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.meta_verifier.schemas import Instruction

logger = logging.getLogger(__name__)


def load_instructions(path: str | Path) -> List[Instruction]:
    """Load instructions from parquet. Returns empty list if file does not exist."""
    p = Path(path)
    if not p.exists():
        logger.info(f"Instruction store at {p} does not exist; starting empty.")
        return []
    df = pd.read_parquet(p)
    return [_row_to_instruction(row) for _, row in df.iterrows()]


def save_instructions(instructions: List[Instruction], path: str | Path) -> None:
    """Write all instructions to parquet, overwriting any existing file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_instruction_to_row(i) for i in instructions])
    df.to_parquet(p, index=False)
    logger.info(f"Wrote {len(instructions)} instructions to {p}")


def append_instructions(new: Iterable[Instruction], path: str | Path) -> List[Instruction]:
    """Load existing, append new (assigning IDs if missing), save, return all."""
    existing = load_instructions(path)
    next_id = (max((i.instruction_id for i in existing), default=0) + 1) if existing else 1
    appended = list(existing)
    for instr in new:
        if instr.instruction_id is None or instr.instruction_id == 0:
            instr = instr.model_copy(update={"instruction_id": next_id})
            next_id += 1
        else:
            next_id = max(next_id, instr.instruction_id + 1)
        appended.append(instr)
    save_instructions(appended, path)
    return appended


def stack_embeddings(instructions: List[Instruction]) -> tuple[np.ndarray, List[int]]:
    """Stack non-null embeddings into a (N, D) matrix and return the
    instruction-list indices they correspond to. Used by the Retriever
    for brute-force cosine search.
    """
    rows: List[List[float]] = []
    indices: List[int] = []
    for idx, instr in enumerate(instructions):
        if instr.semantic_embedding is not None:
            rows.append(instr.semantic_embedding)
            indices.append(idx)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), indices


def _instruction_to_row(instr: Instruction) -> dict:
    return {
        "instruction_id": instr.instruction_id,
        "type": instr.type,
        "instruction_text": instr.instruction_text,
        "description": instr.description,
        "target_codes": list(instr.target_codes),
        "source_hadm_ids": list(instr.source_hadm_ids),
        "fpr_at_creation": instr.fpr_at_creation,
        "fnr_at_creation": instr.fnr_at_creation,
        "efficacy_score": instr.efficacy_score,
        "semantic_embedding": instr.semantic_embedding,
        "created_at": instr.created_at,
    }


def _row_to_instruction(row: pd.Series) -> Instruction:
    embedding = row.get("semantic_embedding")
    if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
        embedding_list: Optional[List[float]] = None
    else:
        embedding_list = [float(x) for x in embedding]

    fpr = row.get("fpr_at_creation")
    fnr = row.get("fnr_at_creation")
    return Instruction(
        instruction_id=int(row["instruction_id"]),
        type=row["type"],
        instruction_text=row["instruction_text"],
        description=row.get("description", "") or "",
        target_codes=list(row.get("target_codes", []) or []),
        source_hadm_ids=list(row.get("source_hadm_ids", []) or []),
        fpr_at_creation=None if fpr is None or (isinstance(fpr, float) and np.isnan(fpr)) else float(fpr),
        fnr_at_creation=None if fnr is None or (isinstance(fnr, float) and np.isnan(fnr)) else float(fnr),
        efficacy_score=float(row.get("efficacy_score", 0.0) or 0.0),
        semantic_embedding=embedding_list,
        created_at=row["created_at"] if "created_at" in row else None,
    )
