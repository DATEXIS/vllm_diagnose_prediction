"""Per-iteration efficacy-score updates.

Only persistent (semantic / contrastive) instructions accumulate reward.
Synthesised threshold warnings are skipped — they have no efficacy
tracking by design (their `efficacy_score` stays at 0.0).

The reward is `learning_rate * delta_F1 * rareness_factor` where:
  * `delta_F1` is the F1 improvement of this iteration's prediction over
    the previous one (against ground truth).
  * `rareness_factor` is precomputed per sample to weight rare-code wins
    more heavily than common-code wins.
"""

from __future__ import annotations

from src.merlin2.generator import GenerateResult
from src.merlin2.pipeline_state import CaseState, case_f1, three_digit_codes
from src.merlin2.retriever_events import RetrievalResult
from src.meta_verifier.schemas import InstructionType


def update_efficacy_scores(
    state: CaseState,
    retrieval: RetrievalResult,
    gen_res: GenerateResult,
    iteration: int,
    learning_rate: float,
) -> None:
    """Compute delta-F1 and update efficacy scores for freshly retrieved instructions."""
    pred_codes = three_digit_codes(gen_res.prediction)
    f1 = case_f1(state.ground_truth_codes, pred_codes)
    prev_f1 = state.iteration_f1[-1] if state.iteration_f1 else 0.0
    state.iteration_f1.append(f1)

    if iteration == 0 or not retrieval.events:
        return

    update = learning_rate * (f1 - prev_f1) * state.rareness_factor
    for instr in retrieval.instructions:
        if instr.type in (InstructionType.FP_WARNING, InstructionType.FN_WARNING):
            continue
        instr.efficacy_score = float(instr.efficacy_score + update)
