"""Per-case state carried through the MERLIN 2 inference loop.

`CaseState` is the mutable record the Pipeline grows wave-by-wave for a
single case. `PipelineCaseResult` is the finalised view returned to
callers once the case halts.

Small set-theoretic helpers live here too so the orchestration file
doesn't carry tiny one-liners.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from src.data.evaluate import normalize_icd
from src.merlin2.retriever import RetrievalEvent
from src.meta_verifier.schemas import Instruction
from src.prompter import ICDsModel


# --------------------------------------------------------------- state

@dataclass
class CaseState:
    hadm_id: str
    admission_note: str
    ground_truth_codes: Optional[List[str]]   # None at test time
    # Pre-parsed, pre-filtered admission note sections (set once at run() init).
    note_sections: Dict[str, str] = field(default_factory=dict)

    # Per-iteration history (grows by one each wave the case participates in)
    predictions: List[ICDsModel] = field(default_factory=list)
    parse_failed_at: List[bool] = field(default_factory=list)  # parallel to predictions
    raw_responses: List[str] = field(default_factory=list)
    thinking_responses: List[str] = field(default_factory=list)  # reasoning_content per iteration
    prompts: List[str] = field(default_factory=list)
    coding_reviews: List[str] = field(default_factory=list)
    retrieval_events: List[List[RetrievalEvent]] = field(default_factory=list)
    instruction_ids_used: List[List[int]] = field(default_factory=list)
    # Instruction objects captured at retrieval time — stored so carry-over
    # reproduces the exact text seen by the model, not the shared mutable cache.
    instructions_used: List[List[Instruction]] = field(default_factory=list)

    halted: bool = False
    halt_reason: str = ""
    seen_instruction_ids: set = field(default_factory=set)


@dataclass
class PipelineCaseResult:
    hadm_id: str
    final_prediction: ICDsModel
    final_raw_response: str
    iterations: int
    halt_reason: str
    history: CaseState  # full per-iteration record
    final_thinking: str = ""  # reasoning_content from the final iteration (empty when thinking=False)


# --------------------------------------------------------------- helpers

def three_digit_codes(model: ICDsModel) -> List[str]:
    return [normalize_icd(d.icd_code) for d in model.diagnoses if normalize_icd(d.icd_code)]


def case_f1(true_codes: Sequence[str], pred_codes: Sequence[str]) -> float:
    t, p = set(true_codes), set(pred_codes)
    if not t and not p:
        return 1.0
    if not t or not p:
        return 0.0
    tp = len(t & p)
    precision = tp / (tp + len(p - t)) if tp + len(p - t) else 0.0
    recall = tp / (tp + len(t - p)) if tp + len(t - p) else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def finalize_case(s: CaseState) -> PipelineCaseResult:
    # Use the last iteration that produced a non-empty prediction.  A parse
    # failure appends ICDsModel(diagnoses=[]), so we skip those when picking
    # the final result.  If every iteration failed (degenerate), fall back to
    # the true last entry rather than crashing.
    idx = len(s.predictions) - 1
    while idx > 0 and not s.predictions[idx].diagnoses:
        idx -= 1

    if idx != len(s.predictions) - 1:
        logger.warning(
            "[PIPELINE] Last iteration for %s had no valid prediction — "
            "falling back to iteration %d.",
            s.hadm_id, idx,
        )

    final_thinking = s.thinking_responses[idx] if s.thinking_responses else ""
    return PipelineCaseResult(
        hadm_id=s.hadm_id,
        final_prediction=s.predictions[idx],
        final_raw_response=s.raw_responses[idx],
        iterations=len(s.predictions),
        halt_reason=s.halt_reason,
        history=s,
        final_thinking=final_thinking,
    )
