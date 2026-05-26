"""Structured DEBUG log emitted per case per wave.

Renders something like:

    [hadm_id | t=N | true: A, B, C]
      Pred:   X, Y, Z
      carry:  N from prior iterations

      FNR – missed codes:
        M33  fnr=1.00  co-occurs-with: Z82, K86

      FPR – rethink codes:
        B18  fpr=1.00

      Semantic – similar to note:
        #42  [E11]  sim=0.85  "If the note mentions long-standing..."

      (skipped N over budget)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.merlin2.retriever import SEM_ICD, THRESHOLD_FNR, THRESHOLD_FPR, is_semantic_path
from src.merlin2.retriever_events import RetrievalResult
from src.meta_verifier.schemas import Instruction

logger = logging.getLogger(__name__)

InstructionHistory = List[Tuple[List[str], List[Instruction]]]


def log_wave_inputs(
    hadm_id: str,
    iteration: int,
    instruction_history: InstructionHistory,
    retrieval: RetrievalResult,
    ground_truth_codes: Optional[List[str]] = None,
) -> None:
    """Emit a structured DEBUG log for one case at one wave."""
    lines = _build_log_lines(hadm_id, iteration, instruction_history, retrieval, ground_truth_codes)
    logger.debug("\n".join(lines))


def _build_log_lines(
    hadm_id: str,
    iteration: int,
    instruction_history: InstructionHistory,
    retrieval: RetrievalResult,
    ground_truth_codes: Optional[List[str]],
) -> List[str]:
    true_str = ", ".join(sorted(ground_truth_codes)) if ground_truth_codes else "—"
    pred_codes = instruction_history[-1][0] if instruction_history else []
    pred_str = ", ".join(pred_codes) if pred_codes else "(none)"
    carry_count = sum(len(instrs) for _, instrs in instruction_history[:-1]) if instruction_history else 0

    lines = [
        f"[{hadm_id} | t={iteration} | true: {true_str}]",
        f"  Pred:   {pred_str}",
        f"  carry:  {carry_count} from prior iterations",
    ]
    lines.extend(_render_retrieval_groups(retrieval))

    if not retrieval.instructions:
        lines.append("  (no instructions retrieved)")
    if retrieval.skipped_for_budget:
        lines.append(f"  ({retrieval.skipped_for_budget} skipped over budget)")
    return lines


def _render_retrieval_groups(retrieval: RetrievalResult) -> List[str]:
    fnr_lines, fpr_lines, sem_by_section, sem_icd_lines = _classify_events(retrieval)

    lines: List[str] = []
    if fnr_lines:
        lines.append("  FNR – missed codes:")
        lines.extend(fnr_lines)
    if fpr_lines:
        lines.append("  FPR – rethink codes:")
        lines.extend(fpr_lines)
    for section_path, entries in sem_by_section.items():
        lines.append(f"  Semantic [{section_path}]:")
        lines.extend(entries)
    if sem_icd_lines:
        lines.append("  Semantic [sem_icd] – similar to reasoning:")
        lines.extend(sem_icd_lines)
    return lines


def _classify_events(retrieval: RetrievalResult):
    ev_by_id = {ev.instruction_id: ev for ev in retrieval.events}
    fnr_lines: List[str] = []
    fpr_lines: List[str] = []
    sem_by_section: Dict[str, List[str]] = {}
    sem_icd_lines: List[str] = []

    for instr in retrieval.instructions:
        ev = ev_by_id.get(instr.instruction_id)
        if ev is None:
            continue
        codes_tag = ", ".join(ev.target_codes) if ev.target_codes else "?"
        snippet = (instr.instruction_text or "")[:70].replace("\n", " ")

        if ev.path == THRESHOLD_FNR:
            cooccur = (
                f"  co-occurs-with: {', '.join(sorted(ev.trigger_codes))}"
                if ev.trigger_codes else ""
            )
            fnr_lines.append(f"    {codes_tag:<6}  fnr={ev.trigger_value:.2f}{cooccur}")
        elif ev.path == THRESHOLD_FPR:
            fpr_lines.append(f"    {codes_tag:<6}  fpr={ev.trigger_value:.2f}")
        elif ev.path == SEM_ICD:
            sem_icd_lines.append(
                f"    #{instr.instruction_id}  [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\""
            )
        elif is_semantic_path(ev.path):
            sem_by_section.setdefault(ev.path, []).append(
                f"    #{instr.instruction_id}  [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\""
            )
    return fnr_lines, fpr_lines, sem_by_section, sem_icd_lines
