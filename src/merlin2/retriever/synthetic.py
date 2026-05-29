"""Runtime-synthesised FP threshold-warning instructions.

FP warnings are NOT persisted in the instruction store. They are synthesised
on demand from the per-code stats table:

  * `synthetic_instruction_id(kind, code)` → deterministic int ID derived
    from md5("<kind>_<code>"), high bit set so it cannot collide with the
    small sequential IDs assigned to persisted (semantic) instructions.

  * `build_threshold_text(...)` → the prose that gets injected into the
    Generator's <coding_review> block.

  * `SyntheticInstructionFactory` → caches FP warnings (their text is stable
    across cases since it depends only on code + stat, not per-case context).
"""

from __future__ import annotations

import hashlib
from typing import Dict

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


def build_threshold_text(code: str, stat: CodeStat) -> str:
    """Render the FP warning text injected into the Generator's <coding_review> block."""
    pct = f"{stat.fpr:.0%}"
    n = stat.support_pred
    return f"{code} — false positive in {pct} of cases where it was predicted across {n} cases."


class SyntheticInstructionFactory:
    """Builds and caches FP-warning Instruction objects.

    FP warning text depends only on `(code, stat)` so the object can be
    cached and re-used across iterations / cases.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, Instruction] = {}

    @property
    def cache(self) -> Dict[int, Instruction]:
        """Read-only view of the cache (for the Retriever's `instructions` property)."""
        return self._cache

    def reset(self) -> None:
        """Drop the cache — used when the underlying code_stats change."""
        self._cache = {}

    def get_or_create(self, code: str, stat: CodeStat) -> Instruction:
        instr_id = synthetic_instruction_id("fp", code)
        cached = self._cache.get(instr_id)
        if cached is not None:
            return cached
        instr = Instruction(
            instruction_id=instr_id,
            type=InstructionType.FP_WARNING,
            action="remove",
            instruction_text=build_threshold_text(code, stat),
            description="",
            target_codes=[code],
            source_hadm_ids=[],
            fpr_at_creation=stat.fpr,
            fnr_at_creation=None,
            efficacy_score=0.0,
            semantic_embedding=None,
        )
        self._cache[instr_id] = instr
        return instr
