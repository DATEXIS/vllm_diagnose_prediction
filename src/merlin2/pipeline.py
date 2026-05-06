"""MERLIN 2 Loop A orchestrator.

Lockstep waves: all live samples complete iteration t before any starts
iteration t+1. After each wave we (a) record per-case retrieval events,
(b) update each freshly-retrieved instruction's efficacy score from
delta-F1 * rareness_factor (training only), and (c) ask the Verifier
which cases halt. Halted cases are filtered out of the next wave.

The very first wave is zero-shot: no retrieved instructions, no
pre-filled <think> block. From t=1 onward retrieval is active.

Phase-level orchestration (multiple Loop-A passes, Loop-B in between)
lives in main.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.data.evaluate import normalize_icd
from src.merlin2.generator import Generator, GenerateRequest, GenerateResult
from src.merlin2.retriever import RetrievalEvent, RetrievalResult, Retriever
from src.utils.embeddings import encode_texts
from src.merlin2.verifier import HaltReason, Verifier
from src.meta_verifier.schemas import Instruction, InstructionType
from src.prompter import ICDsModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- per-case state
@dataclass
class CaseState:
    hadm_id: str
    admission_note: str
    ground_truth_codes: Optional[List[str]]   # None at test time
    rareness_factor: float                    # default 1.0; tunes efficacy reward

    # Per-iteration history (length grows by one each wave the case participates in)
    predictions: List[ICDsModel] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    think_blocks: List[str] = field(default_factory=list)
    retrieval_events: List[List[RetrievalEvent]] = field(default_factory=list)
    instruction_ids_used: List[List[int]] = field(default_factory=list)
    # Instruction objects as returned at retrieval time — stored so carry-over
    # can reproduce the exact text the model saw rather than re-deriving it
    # from the shared (mutable) retriever cache.
    instructions_used: List[List[Instruction]] = field(default_factory=list)
    iteration_f1: List[float] = field(default_factory=list)

    cumulative_think_tokens: int = 0
    halted: bool = False
    halt_reason: str = ""
    seen_instruction_ids: set = field(default_factory=set)


# --------------------------------------------------------------- public result
@dataclass
class PipelineCaseResult:
    hadm_id: str
    final_prediction: ICDsModel
    final_raw_response: str
    iterations: int
    halt_reason: str
    history: CaseState  # full per-iteration record


# --------------------------------------------------------------- helpers
def _three_digit_codes(model: ICDsModel) -> List[str]:
    return [normalize_icd(d.icd_code) for d in model.diagnoses if normalize_icd(d.icd_code)]


def _f1(true_codes: Sequence[str], pred_codes: Sequence[str]) -> float:
    t, p = set(true_codes), set(pred_codes)
    if not t and not p:
        return 1.0
    if not t or not p:
        return 0.0
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------- Pipeline
class MERLINPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        generator: Optional[Generator] = None,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.config = config
        merlin2_cfg = config.get("merlin2", {})

        self.learning_rate: float = merlin2_cfg.get("learning_rate", 1.2)

        self.generator = generator if generator is not None else self._build_generator()
        self.retriever = retriever if retriever is not None else self._build_retriever()
        self.verifier = verifier if verifier is not None else self._build_verifier()

    # ---------- component construction from config (test-injectable) ----
    def _build_generator(self) -> Generator:
        model_cfg = self.config.get("model", {})
        inference_cfg = self.config.get("inference", {})
        job_name = self.config.get("job_name", "default")
        namespace = self.config.get("k8s", {}).get("namespace", "default")
        return Generator(
            api_base=model_cfg.get(
                "api_base",
                f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1",
            ),
            model=model_cfg.get("name", "Qwen/Qwen3-8B"),
            temperature=inference_cfg.get("temperature", 0.0),
            max_tokens=inference_cfg.get("max_tokens", 1024),
            config=self.config,
        )

    def _build_retriever(self) -> Retriever:
        merlin2_cfg = self.config.get("merlin2", {})
        from src.meta_verifier.code_stats import load_code_stats
        from src.utils.cooccurrence import load_cooccurrence_index

        cooccurrence_index = load_cooccurrence_index(
            path=merlin2_cfg.get("cooccurrence_path", "data/cooccurrence.parquet"),
            lift_threshold=float(merlin2_cfg.get("cooccurrence_threshold", 3.0)),
            top_k=int(merlin2_cfg.get("cooccurrence_top_k", 20)),
        )
        code_stats = load_code_stats(
            merlin2_cfg.get("code_stats_path", "data/code_stats.parquet"),
        )
        return Retriever(
            sim_threshold=merlin2_cfg.get("sim_threshold", 0.8),
            fpr_threshold=merlin2_cfg.get("fpr_threshold", 0.5),
            fnr_threshold=merlin2_cfg.get("fnr_threshold", 0.5),
            max_tokens_budget=merlin2_cfg.get("max_tokens_budget", 2500),
            per_iteration_token_budget=merlin2_cfg.get("per_iteration_token_budget"),
            threshold_budget_fraction=merlin2_cfg.get("threshold_budget_fraction", 0.5),
            cooccurrence_index=cooccurrence_index,
            code_stats=code_stats,
        )

    def _build_verifier(self) -> Verifier:
        merlin2_cfg = self.config.get("merlin2", {})
        return Verifier(
            max_iterations=merlin2_cfg.get("max_iterations", 5),
            max_tokens_budget=merlin2_cfg.get("max_tokens_budget", 2500),
            convergence_threshold=merlin2_cfg.get("convergence_threshold", 0.9),
        )

    # ----------------------------------------------------------- run
    async def run(
        self,
        admission_notes: List[str],
        hadm_ids: Optional[List[str]] = None,
        ground_truth_codes: Optional[List[List[str]]] = None,
        rareness_factors: Optional[List[float]] = None,
    ) -> List[PipelineCaseResult]:
        n = len(admission_notes)
        if hadm_ids is None:
            hadm_ids = [str(i) for i in range(n)]
        if rareness_factors is None:
            rareness_factors = [1.0] * n
        if ground_truth_codes is not None and len(ground_truth_codes) != n:
            raise ValueError("ground_truth_codes length mismatch")
        if len(hadm_ids) != n or len(rareness_factors) != n:
            raise ValueError("hadm_ids / rareness_factors length mismatch")

        states: List[CaseState] = [
            CaseState(
                hadm_id=hadm_ids[i],
                admission_note=admission_notes[i],
                ground_truth_codes=(
                    [normalize_icd(c) for c in ground_truth_codes[i]]
                    if ground_truth_codes is not None
                    else None
                ),
                rareness_factor=float(rareness_factors[i]),
            )
            for i in range(n)
        ]

        empty_db = len(self.retriever.instructions) == 0

        # ---- t = 0 : zero-shot wave -----------------------------------
        logger.info(f"[PIPELINE] Wave 0 (zero-shot): {n} cases")
        await self._run_wave(states, iteration=0)

        if empty_db:
            for s in states:
                if s.halted:
                    continue  # don't overwrite a parse_failure halt from t=0
                s.halted = True
                s.halt_reason = HaltReason.EMPTY_DB
            return [self._finalize(s) for s in states]

        # ---- t >= 1 : refinement waves --------------------------------
        for t in range(1, self.verifier.max_iterations):
            live = [s for s in states if not s.halted]
            if not live:
                break
            logger.info(f"[PIPELINE] Wave {t}: {len(live)} live cases")

            # Pre-fetch retrieval results so we can short-circuit cases
            # that get zero new instructions: their prompt would be
            # identical to t-1's, so the generator call would be wasted.
            retrieval_per_case = self._prefetch_retrieval(live)
            for s, retrieval in zip(live, retrieval_per_case):
                if not retrieval.events:
                    s.halted = True
                    s.halt_reason = HaltReason.NO_NEW_INSTRUCTIONS

            still_live = [s for s in live if not s.halted]
            still_live_retrieval = [
                r for s, r in zip(live, retrieval_per_case) if not s.halted
            ]
            if still_live:
                await self._run_wave(
                    still_live,
                    iteration=t,
                    pre_retrieval=still_live_retrieval,
                )

                for s in still_live:
                    halt, reason = self.verifier.should_halt(
                        iteration=t,
                        current_predictions=_three_digit_codes(s.predictions[-1]),
                        previous_predictions=_three_digit_codes(s.predictions[-2]),
                        instructions_retrieved=len(s.retrieval_events[-1]),
                        cumulative_think_tokens=s.cumulative_think_tokens,
                    )
                    if halt:
                        s.halted = True
                        s.halt_reason = reason

        # any case that exhausts max_iterations without halting earlier:
        for s in states:
            if not s.halted:
                s.halted = True
                s.halt_reason = HaltReason.MAX_ITERATIONS_REACHED

        return [self._finalize(s) for s in states]

    # ------------------------------------------------------ pre-fetch
    def _prefetch_retrieval(self, states: List[CaseState]) -> List[RetrievalResult]:
        """Run the Retriever for all live cases (used at t>=1 to detect
        zero-new-instructions before paying for a generator call).

        All admission notes and reason texts are encoded in a single batch
        call so the embedding model (GPU or CPU) is fully utilised.
        """
        # Flatten all texts into one list for a single encode_texts() call.
        all_texts: List[str] = []
        note_indices: List[int] = []
        reason_slices: List[tuple] = []
        per_case_reasons: List[List[str]] = []

        for s in states:
            note_indices.append(len(all_texts))
            all_texts.append(s.admission_note)
            reasons = [d.reason for d in s.predictions[-1].diagnoses if d.reason.strip()]
            per_case_reasons.append(reasons)
            start = len(all_texts)
            all_texts.extend(reasons)
            reason_slices.append((start, start + len(reasons)))

        all_embeddings = encode_texts(all_texts) if all_texts else []

        out: List[RetrievalResult] = []
        for i, s in enumerate(states):
            r_start, r_end = reason_slices[i]
            out.append(
                self.retriever.retrieve(
                    admission_note=s.admission_note,
                    previous_predicted_codes=_three_digit_codes(s.predictions[-1]),
                    already_retrieved_ids=s.seen_instruction_ids,
                    previous_reasons=per_case_reasons[i],
                    note_embedding=all_embeddings[note_indices[i]],
                    reason_embeddings=all_embeddings[r_start:r_end],
                )
            )
        return out

    # ----------------------------------------------------------- wave
    async def _run_wave(
        self,
        states: List[CaseState],
        iteration: int,
        pre_retrieval: Optional[List[RetrievalResult]] = None,
    ) -> None:
        # 1) build retrieval results (skipped at t=0; reused at t>=1)
        if iteration == 0:
            per_case_retrieval: List[RetrievalResult] = [RetrievalResult() for _ in states]
        elif pre_retrieval is not None:
            per_case_retrieval = pre_retrieval
        else:
            per_case_retrieval = self._prefetch_retrieval(states)

        # 2) build per-iteration instruction history for each case.
        #
        # Each history entry is (predicted_codes_at_t, instructions_retrieved_AFTER_t).
        # The think block reads: "I predicted X, then considered these instructions."
        #
        # s.instructions_used[t] = instructions shown TO the model at iteration t,
        # which were retrieved based on the prediction from t-1:
        #   s.instructions_used[0] = []        (zero-shot, nothing shown)
        #   s.instructions_used[1] = retrieved based on pred_0, shown at t=1
        #   s.instructions_used[2] = retrieved based on pred_1, shown at t=2
        #
        # So the correct pairing is predictions[t] with instructions_used[t+1].
        # The previous code paired predictions[t] with instructions_used[t], which
        # produced two bugs: (a) a useless first entry (pred_0, []) and (b) the
        # last two entries both showing the same prediction codes.
        #
        # Carry-over entries use `s.instructions_used` — the instruction objects
        # captured at retrieval time — rather than re-deriving them from the
        # shared retriever cache. The cache may have been mutated by another
        # case in the same batch (FN warning text is case-specific), so reading
        # from the cache would show the wrong trigger codes for carry-over.
        instruction_histories: List[List[Tuple[List[str], List[Instruction]]]] = []
        for s, retrieval in zip(states, per_case_retrieval):
            history: List[Tuple[List[str], List[Instruction]]] = []
            # Carry-over: pair prediction[t] with instructions shown at t+1
            # (those instructions were retrieved based on prediction[t]).
            # Skip index 0: instructions_used[0] is always [] (zero-shot).
            for t in range(len(s.instructions_used) - 1):
                pred_codes = _three_digit_codes(s.predictions[t])
                history.append((pred_codes, s.instructions_used[t + 1]))
            # New entry: latest prediction + freshly retrieved instructions
            if s.predictions:
                history.append((_three_digit_codes(s.predictions[-1]), retrieval.instructions))
            instruction_histories.append(history)

        # 3) build generate requests
        requests: List[GenerateRequest] = []
        for s, history in zip(states, instruction_histories):
            requests.append(
                GenerateRequest(
                    admission_note=s.admission_note,
                    instruction_history=history,
                )
            )

        if logger.isEnabledFor(logging.DEBUG):
            for s, retrieval, history in zip(states, per_case_retrieval, instruction_histories):
                self._log_wave_inputs(
                    s.hadm_id, iteration, history, retrieval, s.ground_truth_codes
                )

        # 4) one batched LLM call
        results: List[GenerateResult] = await self.generator.generate_batch(requests)

        # 5) record per-case state and update efficacy
        for s, retrieval, gen_res in zip(states, per_case_retrieval, results):
            s.predictions.append(gen_res.prediction)
            s.raw_responses.append(gen_res.raw_response)
            s.prompts.append(gen_res.prompt)
            s.think_blocks.append(gen_res.think_block)
            s.retrieval_events.append(retrieval.events)
            new_ids = [ev.instruction_id for ev in retrieval.events]
            s.instruction_ids_used.append(new_ids)
            s.instructions_used.append(list(retrieval.instructions))  # snapshot at retrieval time
            s.seen_instruction_ids.update(new_ids)
            tokens = sum(self.retriever._estimate_tokens(i) for i in retrieval.instructions)
            s.cumulative_think_tokens += tokens

            # Malformed LLM response -> halt the case immediately. Further
            # iterations on the same prompt would hit the same failure.
            if gen_res.parse_failed:
                s.halted = True
                s.halt_reason = HaltReason.PARSE_FAILURE
                continue

            if s.ground_truth_codes is not None:
                pred_codes = _three_digit_codes(gen_res.prediction)
                f1 = _f1(s.ground_truth_codes, pred_codes)
                prev_f1 = s.iteration_f1[-1] if s.iteration_f1 else 0.0
                s.iteration_f1.append(f1)
                if iteration > 0 and retrieval.events:
                    delta = f1 - prev_f1
                    update = self.learning_rate * delta * s.rareness_factor
                    for instr in retrieval.instructions:
                        # Threshold warnings (FP/FN) are synthesised at
                        # runtime from frozen code_stats — they have no
                        # efficacy tracking by design. Skipping the
                        # update prevents the cached synthetic
                        # instruction's score from drifting iteration
                        # over iteration.
                        if instr.type in (
                            InstructionType.FP_WARNING,
                            InstructionType.FN_WARNING,
                        ):
                            continue
                        instr.efficacy_score = float(instr.efficacy_score + update)

    # ----------------------------------------------------------- wave logging
    @staticmethod
    def _log_wave_inputs(
        hadm_id: str,
        iteration: int,
        instruction_history: "List[Tuple[List[str], List[Instruction]]]",
        retrieval: "RetrievalResult",
        ground_truth_codes: Optional[List[str]] = None,
    ) -> None:
        """Structured DEBUG log per case per wave.

        Format:
            [hadm_id | t=N | true: A, B, C]
              Pred:   X, Y, Z
              carry:  N from prior iterations

              FNR – missed codes:
                M33  fnr=1.00  co-occurs-with: Z82, K86
                N70  fnr=1.00  co-occurs-with: N18

              FPR – rethink codes:
                B18  fpr=1.00

              Semantic – similar to note:
                #42  [E11]  sim=0.85  "If the note mentions long-standing..."

              Semantic-reason – similar to reasoning:
                #81  [N18]  sim=0.83  "When CKD is mentioned alongside..."

              (skipped N over budget)
        """
        from src.merlin2.retriever import SEMANTIC, SEMANTIC_REASON, THRESHOLD_FPR, THRESHOLD_FNR

        true_str = ", ".join(sorted(ground_truth_codes)) if ground_truth_codes else "—"
        pred_codes = instruction_history[-1][0] if instruction_history else []
        pred_str = ", ".join(pred_codes) if pred_codes else "(none)"
        carry_count = sum(len(instrs) for _, instrs in instruction_history[:-1]) if instruction_history else 0

        ev_by_id = {ev.instruction_id: ev for ev in retrieval.events}

        fnr_lines: List[str] = []
        fpr_lines: List[str] = []
        sem_lines: List[str] = []
        sem_reason_lines: List[str] = []

        for instr in retrieval.instructions:
            ev = ev_by_id.get(instr.instruction_id)
            if ev is None:
                continue
            codes_tag = ", ".join(ev.target_codes) if ev.target_codes else "?"

            if ev.path == THRESHOLD_FNR:
                cooccur = f"  co-occurs-with: {', '.join(sorted(ev.trigger_codes))}" if ev.trigger_codes else ""
                fnr_lines.append(f"    {codes_tag:<6}  fnr={ev.trigger_value:.2f}{cooccur}")

            elif ev.path == THRESHOLD_FPR:
                fpr_lines.append(f"    {codes_tag:<6}  fpr={ev.trigger_value:.2f}")

            elif ev.path == SEMANTIC:
                snippet = (instr.instruction_text or "")[:70].replace("\n", " ")
                sem_lines.append(f"    #{instr.instruction_id}  [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\"")

            elif ev.path == SEMANTIC_REASON:
                snippet = (instr.instruction_text or "")[:70].replace("\n", " ")
                sem_reason_lines.append(f"    #{instr.instruction_id}  [{codes_tag}]  sim={ev.trigger_value:.2f}  \"{snippet}\"")

        lines: List[str] = [
            f"[{hadm_id} | t={iteration} | true: {true_str}]",
            f"  Pred:   {pred_str}",
            f"  carry:  {carry_count} from prior iterations",
        ]

        if fnr_lines:
            lines.append("  FNR – missed codes:")
            lines.extend(fnr_lines)
        if fpr_lines:
            lines.append("  FPR – rethink codes:")
            lines.extend(fpr_lines)
        if sem_lines:
            lines.append("  Semantic – similar to note:")
            lines.extend(sem_lines)
        if sem_reason_lines:
            lines.append("  Semantic-reason – similar to reasoning:")
            lines.extend(sem_reason_lines)
        if not retrieval.instructions:
            lines.append("  (no instructions retrieved)")
        if retrieval.skipped_for_budget:
            lines.append(f"  ({retrieval.skipped_for_budget} skipped over budget)")

        logger.debug("\n".join(lines))

    # ----------------------------------------------------------- finalize
    def _finalize(self, s: CaseState) -> PipelineCaseResult:
        return PipelineCaseResult(
            hadm_id=s.hadm_id,
            final_prediction=s.predictions[-1],
            final_raw_response=s.raw_responses[-1],
            iterations=len(s.predictions),
            halt_reason=s.halt_reason,
            history=s,
        )

    def _lookup_instructions(self, ids: Sequence[int]) -> List[Instruction]:
        if not ids:
            return []
        by_id = {i.instruction_id: i for i in self.retriever.instructions}
        return [by_id[i] for i in ids if i in by_id]
