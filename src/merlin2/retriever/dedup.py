"""Within-iteration deduplication of semantic retrieval events.

Two passes, both applied AFTER the three retrieval paths have fired and
BEFORE the per-type count caps and budget selection:

  * `cluster_deduplicate_semantic_events` — collapse near-duplicate
    instructions (instruction-to-instruction cosine sim >= threshold) to
    the single highest-`efficacy_score` representative per cluster.

  * `per_code_cap_semantic_events` — at most `max_instructions_per_code`
    semantic instructions per target ICD code. Instructions targeting
    multiple codes count against each of them.

Both expect `events` already sorted by `efficacy_score` descending so a
greedy first-wins scan keeps the highest-quality representative.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .events import RetrievalEvent
from src.meta_verifier.schemas import Instruction


def cluster_deduplicate_semantic_events(
    events: List[RetrievalEvent],
    instructions_by_id: Dict[int, Instruction],
    dedup_cluster_threshold: float,
) -> List[RetrievalEvent]:
    """Drop semantic instructions too similar to an already-selected one.

    Instructions whose embedding is missing or zero-norm are kept
    unconditionally — they cannot be compared and we don't want to silently
    drop them. Threshold instructions are never passed here.
    """
    selected_events: List[RetrievalEvent] = []
    selected_embs: List[np.ndarray] = []
    selected_norms: List[float] = []

    for ev in events:
        instr = instructions_by_id[ev.instruction_id]
        emb, norm = _normalised_embedding(instr)
        if emb is None:
            selected_events.append(ev)
            continue

        if selected_embs and _max_cosine(selected_embs, selected_norms, emb, norm) >= dedup_cluster_threshold:
            continue

        selected_events.append(ev)
        selected_embs.append(emb)
        selected_norms.append(norm)

    return selected_events


def per_code_cap_semantic_events(
    events: List[RetrievalEvent],
    max_instructions_per_code: int,
) -> List[RetrievalEvent]:
    """Keep at most `max_instructions_per_code` semantic events per target code.

    Instructions with no `target_codes` are always admitted.
    """
    code_counts: Dict[str, int] = {}
    selected: List[RetrievalEvent] = []

    for ev in events:
        if not ev.target_codes:
            selected.append(ev)
            continue
        if any(code_counts.get(c, 0) >= max_instructions_per_code for c in ev.target_codes):
            continue
        selected.append(ev)
        for c in ev.target_codes:
            code_counts[c] = code_counts.get(c, 0) + 1

    return selected


def _normalised_embedding(instr: Instruction):
    if instr.semantic_embedding is None:
        return None, 0.0
    emb = np.asarray(instr.semantic_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm == 0:
        return None, 0.0
    return emb, norm


def _max_cosine(
    selected_embs: List[np.ndarray],
    selected_norms: List[float],
    emb: np.ndarray,
    norm: float,
) -> float:
    sel_mat = np.stack(selected_embs)
    sel_norms = np.asarray(selected_norms, dtype=np.float32)
    dots = sel_mat @ emb
    denom = sel_norms * norm
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = np.where(denom > 0, dots / denom, 0.0)
    return float(np.max(sims))
