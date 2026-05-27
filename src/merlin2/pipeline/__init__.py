"""MERLIN 2 Loop A orchestrator.

Lockstep waves: all live samples complete iteration t before any starts
t+1. After each wave: (a) record retrieval events, (b) update efficacy
scores from delta-F1 * rareness_factor (training only), (c) ask the
Verifier which cases halt.

The first wave is zero-shot (no instructions, no <think> block). From t=1
onward retrieval is active. Phase-level orchestration lives in main.py.

This file is intentionally thin. Heavy lifting lives in sibling modules:

  * state      — CaseState / PipelineCaseResult + small helpers
  * builders   — Generator / Retriever / Verifier construction
  * embedding  — batched note + reason embedding
  * efficacy   — efficacy-score updates
  * logging    — DEBUG per-wave structured log
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.merlin2.generator import Generator, GenerateRequest, GenerateResult
from .builders import build_generator, build_retriever, build_verifier
from .efficacy import update_efficacy_scores
from .embedding import BatchedEmbeddings, embed_cases
from .logging import log_wave_inputs
from .state import (
    CaseState, PipelineCaseResult, finalize_case, three_digit_codes,
)
from src.merlin2.retriever import RetrievalResult, Retriever
from src.merlin2.verifier import HaltReason, Verifier
from src.utils.admission_note_parser import filter_sections, parse_sections
from src.data.evaluate import normalize_icd
from src.meta_verifier.schemas import Instruction

# Re-export for backward compatibility (reporting.py and tests still import these here).
__all__ = ["CaseState", "MERLINPipeline", "PipelineCaseResult"]

logger = logging.getLogger(__name__)


class MERLINPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        generator: Optional[Generator] = None,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.config = config
        m2_cfg = config.get("merlin2", {})
        self.learning_rate: float = m2_cfg.get("learning_rate", 1.2)
        self._update_efficacy_enabled: bool = m2_cfg.get("update_efficacy", True)

        self.generator = generator if generator is not None else build_generator(config)
        self.retriever = retriever if retriever is not None else build_retriever(config)
        self.verifier  = verifier  if verifier  is not None else build_verifier(config)

    # ---------------------------------------------------------------- run (Loop A)

    async def run(
        self,
        admission_notes: List[str],
        hadm_ids: Optional[List[str]] = None,
        ground_truth_codes: Optional[List[List[str]]] = None,
        rareness_factors: Optional[List[float]] = None,
    ) -> List[PipelineCaseResult]:
        states = self._init_states(admission_notes, hadm_ids, ground_truth_codes, rareness_factors)
        self._parse_note_sections(states)

        logger.info(f"[PIPELINE] Wave 0 (zero-shot): {len(states)} cases")
        await self._run_wave(states, iteration=0)

        if not self.retriever.instructions:
            return self._halt_all_with_empty_db(states)

        for t in range(1, self.verifier.max_iterations):
            live = [s for s in states if not s.halted]
            if not live:
                break
            logger.info(f"[PIPELINE] Wave {t}: {len(live)} live cases")
            await self._refinement_wave(live, iteration=t)

        self._halt_remaining(states, HaltReason.MAX_ITERATIONS_REACHED)
        return [finalize_case(s) for s in states]

    # ---------------------------------------------------------------- init / halt

    def _init_states(
        self,
        admission_notes,
        hadm_ids,
        ground_truth_codes,
        rareness_factors,
    ) -> List[CaseState]:
        n = len(admission_notes)
        hadm_ids = hadm_ids or [str(i) for i in range(n)]
        rareness_factors = rareness_factors or [1.0] * n
        _validate_run_lengths(n, hadm_ids, rareness_factors, ground_truth_codes)
        return [
            CaseState(
                hadm_id=hadm_ids[i],
                admission_note=admission_notes[i],
                ground_truth_codes=(
                    [normalize_icd(c) for c in ground_truth_codes[i]]
                    if ground_truth_codes is not None else None
                ),
                rareness_factor=float(rareness_factors[i]),
            )
            for i in range(n)
        ]

    def _parse_note_sections(self, states: List[CaseState]) -> None:
        """Parse admission note sections once per case for batch embedding efficiency."""
        if not self.retriever.section_names:
            return
        for s in states:
            parsed = parse_sections(s.admission_note, self.retriever.section_names)
            s.note_sections = filter_sections(parsed, self.retriever.ignore_phrases)
        avg = sum(len(s.note_sections) for s in states) / max(len(states), 1)
        logger.debug("[PIPELINE] Section parsing: avg %.1f non-empty sections per case", avg)

    def _halt_all_with_empty_db(self, states: List[CaseState]) -> List[PipelineCaseResult]:
        self._halt_remaining(states, HaltReason.EMPTY_DB)
        return [finalize_case(s) for s in states]

    @staticmethod
    def _halt_remaining(states: List[CaseState], reason: str) -> None:
        for s in states:
            if not s.halted:
                s.halted = True
                s.halt_reason = reason

    # ---------------------------------------------------------------- refinement

    async def _refinement_wave(self, live: List[CaseState], iteration: int) -> None:
        """Pre-fetch retrievals so empty-retrieval cases short-circuit, then run the wave."""
        retrievals = self._prefetch_retrieval(live)
        self._halt_empty_retrievals(live, retrievals)

        still_live = [(s, r) for s, r in zip(live, retrievals) if not s.halted]
        if not still_live:
            return
        live_states, live_retrievals = zip(*still_live)
        await self._run_wave(list(live_states), iteration=iteration, pre_retrieval=list(live_retrievals))
        self._check_verifier_halts(list(live_states), iteration=iteration)

    def _halt_empty_retrievals(
        self, states: List[CaseState], retrievals: List[RetrievalResult]
    ) -> None:
        for s, retrieval in zip(states, retrievals):
            if not retrieval.events:
                if len(three_digit_codes(s.predictions[-1])) < self.verifier.min_prediction_size:
                    continue  # under-predicting — let it keep iterating
                s.halted = True
                s.halt_reason = HaltReason.NO_NEW_INSTRUCTIONS

    def _check_verifier_halts(self, states: List[CaseState], iteration: int) -> None:
        for s in states:
            halt, reason = self.verifier.should_halt(
                iteration=iteration,
                current_predictions=three_digit_codes(s.predictions[-1]),
                previous_predictions=three_digit_codes(s.predictions[-2]),
                instructions_retrieved=len(s.retrieval_events[-1]),
            )
            if halt:
                s.halted = True
                s.halt_reason = reason

    # ---------------------------------------------------------------- wave

    async def _run_wave(
        self,
        states: List[CaseState],
        iteration: int,
        pre_retrieval: Optional[List[RetrievalResult]] = None,
    ) -> None:
        retrievals = self._retrievals_for_wave(states, iteration, pre_retrieval)
        histories = [self._build_instruction_history(s, r) for s, r in zip(states, retrievals)]
        requests = [GenerateRequest(s.admission_note, h) for s, h in zip(states, histories)]

        if logger.isEnabledFor(logging.DEBUG):
            for s, retrieval, history in zip(states, retrievals, histories):
                log_wave_inputs(s.hadm_id, iteration, history, retrieval, s.ground_truth_codes)

        gen_results = await self.generator.generate_batch(requests)
        for s, retrieval, gen_res in zip(states, retrievals, gen_results):
            self._record_wave_result(s, retrieval, gen_res, iteration)

    def _retrievals_for_wave(
        self,
        states: List[CaseState],
        iteration: int,
        pre_retrieval: Optional[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        if iteration == 0:
            return [RetrievalResult() for _ in states]
        if pre_retrieval is not None:
            return pre_retrieval
        return self._prefetch_retrieval(states)

    @staticmethod
    def _build_instruction_history(
        s: CaseState, retrieval: RetrievalResult,
    ) -> List[Tuple[List[str], List[Instruction]]]:
        """Build the per-iteration (predicted_codes, instructions) history for one case.

        Each entry pairs the codes predicted at iteration t with the
        instructions retrieved in response to that prediction (shown to the
        model at t+1). The new entry appends the latest prediction plus
        fresh retrieval at the end.
        """
        history: List[Tuple[List[str], List[Instruction]]] = []
        # instructions_used[0] is always [] (zero-shot), so skip index 0.
        for t in range(len(s.instructions_used) - 1):
            history.append((three_digit_codes(s.predictions[t]), s.instructions_used[t + 1]))
        if s.predictions:
            history.append((three_digit_codes(s.predictions[-1]), retrieval.instructions))
        return history

    def _record_wave_result(
        self,
        s: CaseState,
        retrieval: RetrievalResult,
        gen_res: GenerateResult,
        iteration: int,
    ) -> None:
        s.predictions.append(gen_res.prediction)
        s.raw_responses.append(gen_res.raw_response)
        s.thinking_responses.append(gen_res.thinking_content)
        s.prompts.append(gen_res.prompt)
        s.coding_reviews.append(gen_res.coding_review)
        s.retrieval_events.append(retrieval.events)
        new_ids = [ev.instruction_id for ev in retrieval.events]
        s.instruction_ids_used.append(new_ids)
        s.instructions_used.append(list(retrieval.instructions))
        s.seen_instruction_ids.update(new_ids)

        if gen_res.parse_failed:
            # Soft-land: treat as an empty prediction and continue rather than
            # halting immediately. The empty output (0 codes) triggers the
            # min_prediction_size guard in the Verifier so the case keeps
            # iterating. max_iterations remains the hard ceiling.
            logger.warning(
                "[PIPELINE] Parse failure for %s at iteration %d — "
                "treating as empty prediction, will retry next wave.",
                s.hadm_id, iteration,
            )
            return

        if not gen_res.prediction.diagnoses:
            # Valid JSON but empty diagnoses list — same soft-land as parse failure.
            # The zero-prediction escalation message in instruction_feedback will
            # fire next wave because last_codes will be [].
            logger.warning(
                "[PIPELINE] Empty diagnoses list for %s at iteration %d — "
                "will retry next wave with zero-prediction escalation.",
                s.hadm_id, iteration,
            )
            return

        if s.ground_truth_codes is not None and self._update_efficacy_enabled:
            update_efficacy_scores(s, retrieval, gen_res, iteration, self.learning_rate)

    # ---------------------------------------------------------------- pre-fetch

    def _prefetch_retrieval(self, states: List[CaseState]) -> List[RetrievalResult]:
        """Run the Retriever for all live cases with a single pooled embedding call."""
        batch = embed_cases(states)
        return [self._retrieve_for_case(s, i, batch) for i, s in enumerate(states)]

    def _retrieve_for_case(
        self, s: CaseState, case_idx: int, batch: BatchedEmbeddings,
    ) -> RetrievalResult:
        return self.retriever.retrieve(
            admission_note=s.admission_note,
            previous_predicted_codes=three_digit_codes(s.predictions[-1]),
            already_retrieved_ids=s.seen_instruction_ids,
            previous_reasons=batch.reasons(case_idx),
            note_sections=s.note_sections or None,
            section_embeddings=batch.section_embeddings(case_idx),
            reason_embeddings=batch.reason_embeddings(case_idx),
        )

    # ---------------------------------------------------------------- misc helpers

    def _lookup_instructions(self, ids):
        if not ids:
            return []
        by_id = {i.instruction_id: i for i in self.retriever.instructions}
        return [by_id[i] for i in ids if i in by_id]


# --------------------------------------------------------------- validation

def _validate_run_lengths(
    n: int,
    hadm_ids: List[str],
    rareness_factors: List[float],
    ground_truth_codes: Optional[List[List[str]]],
) -> None:
    if ground_truth_codes is not None and len(ground_truth_codes) != n:
        raise ValueError("ground_truth_codes length mismatch")
    if len(hadm_ids) != n or len(rareness_factors) != n:
        raise ValueError("hadm_ids / rareness_factors length mismatch")
