"""Retrieval result dataclasses.

The Retriever produces one `RetrievalEvent` per fired instruction, plus a
top-level `RetrievalResult` that the Pipeline consumes (the instructions
to inject into the prompt, the events for later threshold tuning, and a
budget-skip counter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.meta_verifier.schemas import Instruction


@dataclass
class RetrievalEvent:
    """A single retrieval hit, recorded for later threshold tuning."""
    instruction_id: int
    path: str                      # sem_* (section/ICD) | threshold_fpr | threshold_fnr
    trigger_value: float           # cosine score, or fpr/fnr value
    efficacy_score: float
    target_codes: List[str] = field(default_factory=list)   # codes this instruction targets
    trigger_codes: List[str] = field(default_factory=list)  # FNR only: predicted codes that fired this


@dataclass
class RetrievalResult:
    instructions: List[Instruction] = field(default_factory=list)
    events: List[RetrievalEvent] = field(default_factory=list)
    skipped_for_budget: int = 0
