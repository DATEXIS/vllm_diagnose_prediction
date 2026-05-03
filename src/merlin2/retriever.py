"""MERLIN 2 Hybrid-Retriever.

Two retrieval paths, OR'd together:

  * Semantic path  — embed the admission note, retrieve instructions
                     from the persistent store whose `description`
                     embedding has cosine similarity >= sim_threshold.

  * Threshold path — synthesised at runtime from the per-code stats
                     table (no persistent Instruction rows). Inactive at
                     t=0. Two distinct gates:
      - FP gate: for each code in the previous prediction, look up
                 `code_stats[code].fpr`. If >= fpr_threshold, synthesise
                 an FP warning Instruction and emit it.
      - FN gate: expand the previous prediction via the co-occurrence
                 index (codes with high lift to the predicted ones,
                 top-K capped at load time). For each cooccurring code,
                 look up `code_stats[code].fnr`. If >= fnr_threshold,
                 synthesise an FN warning Instruction and emit it.
    The asymmetry is deliberate: an FP warning is about a code the model
    DID predict, an FN warning is about a code the model SHOULD HAVE
    predicted — gating both on the same predicted set would make the FN
    branch unreachable for missed codes.

Synthesised threshold warnings get deterministic instruction IDs (md5
hash of "fp_<code>" / "fn_<code>") so they survive iteration-to-iteration
deduplication and the Pipeline's instruction-by-id lookup. They have no
semantic_embedding (they don't fire via the semantic path) and no
efficacy tracking — `efficacy_score` is always 0.0 (per design).

Within a single case, instructions retrieved at earlier iterations are
suppressed (the Pipeline maintains the per-case "already retrieved" set
via the `already_retrieved_ids` argument).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from src.meta_verifier.code_stats import CodeStat, CodeStatsIndex
from src.meta_verifier.schemas import Instruction, InstructionType
from src.utils.cooccurrence import CooccurrenceIndex, expand_cooccurring
from src.utils.embeddings import encode_single_text

logger = logging.getLogger(__name__)


SEMANTIC = "semantic"
THRESHOLD_FPR = "threshold_fpr"
THRESHOLD_FNR = "threshold_fnr"


def synthetic_instruction_id(kind: str, code: str) -> int:
    """Deterministic instruction_id for a runtime-synthesised threshold warning.

    md5("<kind>_<code>") truncated to 32 bits, with the high bit set so
    the value is always >= 2^31 — a clean ID range that cannot collide
    with the small sequential int IDs assigned to persisted (semantic)
    instructions.
    """
    h = hashlib.md5(f"{kind}_{code}".encode()).hexdigest()[:8]
    return int(h, 16) | (1 << 31)


def _build_threshold_text(kind: str, code: str, stat: CodeStat) -> str:
    """Render the warning text injected into the Generator's <think> block."""
    if kind == "fp":
        return (
            f"You predicted {code}, but {code} has historical FPR "
            f"{stat.fpr:.2f} (n={stat.support_pred}). Re-examine whether the "
            f"admission note actually supports this code; demote if cues are weak."
        )
    if kind == "fn":
        return (
            f"Code {code} is missed in {stat.fnr:.0%} of cases where it should "
            f"have been assigned (n={stat.support_true}). Look for cues "
            f"consistent with {code} in the admission note before finalizing."
        )
    raise ValueError(f"Unknown threshold-warning kind: {kind!r}")


@dataclass
class RetrievalEvent:
    """A single retrieval hit, recorded for later threshold tuning."""
    instruction_id: int
    path: str                # SEMANTIC | THRESHOLD_FPR | THRESHOLD_FNR
    trigger_value: float     # cosine score, or fpr/fnr_at_creation
    efficacy_score: float


@dataclass
class RetrievalResult:
    instructions: List[Instruction] = field(default_factory=list)
    events: List[RetrievalEvent] = field(default_factory=list)
    skipped_for_budget: int = 0


class Retriever:
    """Threshold-based hybrid retriever over a fixed instruction list."""

    def __init__(
        self,
        sim_threshold: float = 0.8,
        fpr_threshold: float = 0.5,
        fnr_threshold: float = 0.5,
        max_tokens_budget: int = 2500,
        threshold_budget_fraction: float = 0.5,
        cooccurrence_index: Optional[CooccurrenceIndex] = None,
        code_stats: Optional[CodeStatsIndex] = None,
    ):
        self.sim_threshold = sim_threshold
        self.fpr_threshold = fpr_threshold
        self.fnr_threshold = fnr_threshold
        self.max_tokens_budget = max_tokens_budget
        self.threshold_budget_fraction = threshold_budget_fraction
        # Cooccurrence index: {predicted_code -> [(other_code, lift), ...]}.
        # Threshold + top-K already applied at load time. An empty dict
        # disables the FN-via-cooccurrence path entirely.
        self._cooccurrence_index: CooccurrenceIndex = cooccurrence_index or {}
        # Per-code threshold stats (frozen rates from Loop B). Empty dict
        # disables the threshold path entirely.
        self._code_stats: CodeStatsIndex = code_stats or {}
        # Persistent instructions, loaded from the parquet store.
        self._instructions: List[Instruction] = []
        # Synthesised threshold-warning instructions, cached so the
        # pipeline's instruction-by-id lookup (used for prompt carry-over)
        # finds them. Keyed by deterministic-hash instruction_id.
        self._synthetic_cache: Dict[int, Instruction] = {}

        # Cached embedding matrix and parallel index list, rebuilt lazily.
        # Built only over self._instructions (persisted) — synthesised
        # warnings have no embedding and never participate in the
        # semantic path.
        self._emb_matrix: Optional[np.ndarray] = None
        self._emb_indices: List[int] = []
        self._emb_norms: Optional[np.ndarray] = None

    def load_cooccurrence_index(self, index: CooccurrenceIndex) -> None:
        """Replace the cooccurrence index. Empty dict disables the FN path."""
        self._cooccurrence_index = index or {}

    def load_code_stats(self, stats: CodeStatsIndex) -> None:
        """Replace the per-code stats lookup. Empty dict disables the threshold path."""
        self._code_stats = stats or {}
        # Cached synthesised instructions are stale once stats change.
        self._synthetic_cache = {}

    # ------------------------------------------------------------------ load
    def load_instructions(self, instructions: Iterable[Instruction]) -> None:
        self._instructions = list(instructions)
        self._invalidate_cache()
        logger.info(f"Retriever loaded {len(self._instructions)} instructions")

    def add_instruction(self, instruction: Instruction) -> None:
        self._instructions.append(instruction)
        self._invalidate_cache()

    @property
    def instructions(self) -> List[Instruction]:
        # Persistent + synthesised, in that order. The pipeline's
        # `_lookup_instructions` uses this for prompt carry-over and must
        # see synthesised threshold warnings too. Persisted callers
        # (e.g. main.py saving to parquet) should NOT save this list
        # directly — they only handle the new instructions returned from
        # the Meta-Verifier audit, never the retriever's view.
        return list(self._instructions) + list(self._synthetic_cache.values())

    def _invalidate_cache(self) -> None:
        self._emb_matrix = None
        self._emb_indices = []
        self._emb_norms = None

    def _build_embedding_cache(self) -> None:
        rows: List[List[float]] = []
        idxs: List[int] = []
        for i, instr in enumerate(self._instructions):
            if instr.semantic_embedding is not None:
                rows.append(instr.semantic_embedding)
                idxs.append(i)
        if not rows:
            self._emb_matrix = np.zeros((0, 0), dtype=np.float32)
            self._emb_indices = []
            self._emb_norms = np.zeros((0,), dtype=np.float32)
            return
        mat = np.asarray(rows, dtype=np.float32)
        self._emb_matrix = mat
        self._emb_indices = idxs
        self._emb_norms = np.linalg.norm(mat, axis=1)

    # ------------------------------------------------------------ retrieval
    def retrieve(
        self,
        admission_note: str,
        previous_predicted_codes: Optional[List[str]],
        already_retrieved_ids: Optional[Set[int]] = None,
    ) -> RetrievalResult:
        """Retrieve instructions for one case at one iteration.

        Args:
            admission_note: the raw admission note (used by the semantic path).
            previous_predicted_codes: the 3-digit codes predicted at the
                previous iteration. None or [] -> threshold path is skipped
                (this is the t=0 case).
            already_retrieved_ids: instructions already used in earlier
                iterations of this same case; they are suppressed.

        Returns:
            A RetrievalResult containing the (deduped, budget-capped)
            instruction list and a per-hit log of how each was triggered.
        """
        if already_retrieved_ids is None:
            already_retrieved_ids = set()

        if self._emb_matrix is None:
            self._build_embedding_cache()

        # ---- collect triggers per instruction (one event each) ---------
        triggered: dict[int, RetrievalEvent] = {}

        # Semantic path
        if self._emb_matrix is not None and self._emb_matrix.size > 0:
            note_emb = np.asarray(encode_single_text(admission_note), dtype=np.float32)
            note_norm = float(np.linalg.norm(note_emb))
            if note_norm > 0:
                # cosine sims to all stored instructions with embeddings
                dots = self._emb_matrix @ note_emb
                denom = self._emb_norms * note_norm
                # Avoid div-by-zero for instructions with zero-norm embeddings
                with np.errstate(invalid="ignore", divide="ignore"):
                    sims = np.where(denom > 0, dots / denom, 0.0)
                hits = np.where(sims >= self.sim_threshold)[0]
                for h in hits:
                    instr_idx = self._emb_indices[int(h)]
                    instr = self._instructions[instr_idx]
                    if instr.instruction_id in already_retrieved_ids:
                        continue
                    triggered[instr.instruction_id] = RetrievalEvent(
                        instruction_id=instr.instruction_id,
                        path=SEMANTIC,
                        trigger_value=float(sims[h]),
                        efficacy_score=instr.efficacy_score,
                    )

        # Threshold path (only for t>=1). Synthesised at runtime from
        # code_stats — no persistent Instruction rows.
        #   FP gate: code in predicted_set with code_stats[code].fpr >= thr
        #   FN gate: code in cooccurring_set with code_stats[code].fnr >= thr
        if previous_predicted_codes and self._code_stats:
            predicted_set = set(previous_predicted_codes)
            cooccurring_set = expand_cooccurring(
                self._cooccurrence_index, list(predicted_set)
            )

            # FP gate
            for code in predicted_set:
                stat = self._code_stats.get(code)
                if stat is None or stat.fpr is None or stat.fpr < self.fpr_threshold:
                    continue
                instr = self._get_or_create_synthetic("fp", code, stat)
                if instr.instruction_id in already_retrieved_ids:
                    continue
                if instr.instruction_id in triggered:
                    continue  # already grabbed via semantic path
                triggered[instr.instruction_id] = RetrievalEvent(
                    instruction_id=instr.instruction_id,
                    path=THRESHOLD_FPR,
                    trigger_value=stat.fpr,
                    efficacy_score=0.0,
                )

            # FN gate
            for code in cooccurring_set:
                stat = self._code_stats.get(code)
                if stat is None or stat.fnr is None or stat.fnr < self.fnr_threshold:
                    continue
                instr = self._get_or_create_synthetic("fn", code, stat)
                if instr.instruction_id in already_retrieved_ids:
                    continue
                if instr.instruction_id in triggered:
                    continue
                triggered[instr.instruction_id] = RetrievalEvent(
                    instruction_id=instr.instruction_id,
                    path=THRESHOLD_FNR,
                    trigger_value=stat.fnr,
                    efficacy_score=0.0,
                )

        # ---- prioritize and apply split token budget -------------------
        # Problem: threshold instructions always have efficacy_score=0.0
        # (no tracking by design), while semantic instructions accumulate
        # positive efficacy over training runs. A purely efficacy-ordered
        # queue would fill the budget with semantics and never admit
        # FPR/FNR warnings.
        #
        # Solution: split the budget.
        #   - Threshold slot: threshold_budget_fraction * max_tokens_budget
        #     FPR/FNR instructions sorted by trigger_value desc (most-
        #     violated codes first). Any unused threshold tokens spill
        #     into the semantic slot.
        #   - Semantic slot: the remainder, sorted by efficacy desc.
        #
        # `instructions` property includes synthesised threshold warnings
        # so they are looked up alongside persisted ones.
        id_to_instr = {i.instruction_id: i for i in self.instructions}

        threshold_events = sorted(
            [ev for ev in triggered.values() if ev.path in (THRESHOLD_FPR, THRESHOLD_FNR)],
            key=lambda ev: (-ev.trigger_value, ev.instruction_id),
        )
        semantic_events = sorted(
            [ev for ev in triggered.values() if ev.path == SEMANTIC],
            key=lambda ev: (-ev.efficacy_score, ev.instruction_id),
        )

        selected_instructions: List[Instruction] = []
        selected_events: List[RetrievalEvent] = []
        total_tokens = 0
        skipped = 0

        # Fill threshold slot first
        threshold_cap = int(self.max_tokens_budget * self.threshold_budget_fraction)
        threshold_tokens = 0
        for ev in threshold_events:
            instr = id_to_instr[ev.instruction_id]
            est = self._estimate_tokens(instr)
            if threshold_tokens + est > threshold_cap:
                skipped += 1
                continue
            selected_instructions.append(instr)
            selected_events.append(ev)
            threshold_tokens += est
            total_tokens += est

        # Fill remaining budget with semantic instructions
        for ev in semantic_events:
            instr = id_to_instr[ev.instruction_id]
            est = self._estimate_tokens(instr)
            if total_tokens + est > self.max_tokens_budget:
                skipped += 1
                continue
            selected_instructions.append(instr)
            selected_events.append(ev)
            total_tokens += est

        if skipped:
            logger.debug(
                "Budget cap hit: %d instruction(s) skipped "
                "(threshold_used=%d/%d, total_used=%d/%d)",
                skipped, threshold_tokens, threshold_cap,
                total_tokens, self.max_tokens_budget,
            )

        return RetrievalResult(
            instructions=selected_instructions,
            events=selected_events,
            skipped_for_budget=skipped,
        )

    # ----------------------------------------------- synthesised warnings
    def _get_or_create_synthetic(
        self, kind: str, code: str, stat: CodeStat
    ) -> Instruction:
        """Build (or fetch from cache) a runtime threshold-warning Instruction.

        Cached so the same instruction_id is reused across iterations of
        the same case — critical for the dedup / carry-over machinery.
        Synthesised instructions never carry an embedding (threshold-only)
        and never accumulate efficacy (per design).
        """
        instr_id = synthetic_instruction_id(kind, code)
        cached = self._synthetic_cache.get(instr_id)
        if cached is not None:
            return cached
        instr_type = (
            InstructionType.FP_WARNING if kind == "fp" else InstructionType.FN_WARNING
        )
        instr = Instruction(
            instruction_id=instr_id,
            type=instr_type,
            instruction_text=_build_threshold_text(kind, code, stat),
            description="",
            target_codes=[code],
            source_hadm_ids=[],
            fpr_at_creation=stat.fpr,
            fnr_at_creation=stat.fnr,
            efficacy_score=0.0,
            semantic_embedding=None,
        )
        self._synthetic_cache[instr_id] = instr
        return instr

    @staticmethod
    def _estimate_tokens(instruction: Instruction) -> int:
        # Rough word-count heuristic; the budget is a soft cap, not a contract.
        text = (instruction.instruction_text or "") + " " + (instruction.description or "")
        words = len(text.split())
        return max(1, int(words * 1.3))
