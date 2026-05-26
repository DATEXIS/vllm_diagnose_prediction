"""Runtime-synthesised threshold-warning instructions.

Threshold warnings (FP / FN) are NOT persisted in the instruction store.
They are synthesised on demand from the per-code stats table:

  * `synthetic_instruction_id(kind, code)` → deterministic int ID derived
    from md5("<kind>_<code>"), high bit set so it cannot collide with the
    small sequential IDs assigned to persisted (semantic) instructions.

  * `build_threshold_text(...)` → the prose that gets injected into the
    Generator's <coding_review> block.

  * `SyntheticInstructionFactory` → caches FP warnings (their text is stable),
    builds FN warnings fresh per case so concurrent batches don't share
    mutable trigger-code text.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from src.meta_verifier.code_stats import CodeStat
from src.meta_verifier.schemas import Instruction, InstructionType


def synthetic_instruction_id(kind: str, code: str) -> int:
    """Deterministic instruction_id for a runtime-synthesised threshold warning.

    md5("<kind>_<code>") truncated to 32 bits, with the high bit set so the
    value is always >= 2^31 — a clean ID range that cannot collide with the
    small sequential int IDs assigned to persisted (semantic) instructions.
    """
    h = hashlib.md5(f"{kind}_{code}".encode()).hexdigest()[:8]
    return int(h, 16) | (1 << 31)


def build_threshold_text(
    kind: str,
    code: str,
    stat: CodeStat,
    trigger_codes: Optional[List[str]] = None,
) -> str:
    """Render the warning text injected into the Generator's <think> block.

    For FN warnings, `trigger_codes` is the list of predicted codes whose
    co-occurrence with `code` caused this warning to fire. Naming them lets
    the model understand *why* the warning is relevant to this case.
    """
    if kind == "fp":
        pct = f"{stat.fpr:.0%}"
        n = stat.support_pred
        return (
            f"{code} — false positive in {pct} of cases where it was predicted across {n} cases."
        )
    if kind == "fn":
        cooccur_clause = ""
        if trigger_codes:
            codes_str = ", ".join(sorted(trigger_codes))
            cooccur_clause = f"frequently co-occurs with my prediction {codes_str} and "
        return (
            f"{code} {cooccur_clause} is missed {stat.fnr:.0%} of the time"
            f" when it should be assigned (n={stat.support_true})"
        )
    raise ValueError(f"Unknown threshold-warning kind: {kind!r}")


class SyntheticInstructionFactory:
    """Builds threshold-warning Instruction objects, caching FP warnings.

    FP warning text depends only on `(code, stat)` so the object can be
    cached and re-used across iterations / cases. FN warning text depends
    on the per-case `trigger_codes`, so a fresh object is built every call;
    the cache entry is still updated so dedup-by-ID keeps working.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, Instruction] = {}

    @property
    def cache(self) -> Dict[int, Instruction]:
        """Read-only view of the synthesised-instruction cache (for the Retriever's `instructions` property)."""
        return self._cache

    def reset(self) -> None:
        """Drop the cache — used when the underlying code_stats change."""
        self._cache = {}

    def get_or_create(
        self,
        kind: str,
        code: str,
        stat: CodeStat,
        trigger_codes: Optional[List[str]] = None,
    ) -> Instruction:
        instr_id = synthetic_instruction_id(kind, code)

        if kind == "fp":
            cached = self._cache.get(instr_id)
            if cached is not None:
                return cached
            instr = _build_synthetic_instruction(
                instr_id, InstructionType.FP_WARNING, kind, code, stat,
            )
            self._cache[instr_id] = instr
            return instr

        # kind == "fn": always build a fresh object with case-specific text.
        instr = _build_synthetic_instruction(
            instr_id, InstructionType.FN_WARNING, kind, code, stat, trigger_codes,
        )
        # Update cache so the deterministic ID is findable for dedup.
        # Carry-over uses CaseState.instructions_used (not this cache) so
        # storing the latest case's version here is harmless.
        self._cache[instr_id] = instr
        return instr


def _build_synthetic_instruction(
    instr_id: int,
    type_: str,
    kind: str,
    code: str,
    stat: CodeStat,
    trigger_codes: Optional[List[str]] = None,
) -> Instruction:
    return Instruction(
        instruction_id=instr_id,
        type=type_,
        instruction_text=build_threshold_text(kind, code, stat, trigger_codes),
        description="",
        target_codes=[code],
        source_hadm_ids=[],
        fpr_at_creation=stat.fpr,
        fnr_at_creation=stat.fnr,
        efficacy_score=0.0,
        semantic_embedding=None,
    )
