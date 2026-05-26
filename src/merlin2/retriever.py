"""MERLIN 2 Hybrid-Retriever (orchestration).

Three retrieval paths, OR'd together at the per-instruction level:

  * Semantic path        — embed each admission-note section, retrieve
                           instructions whose stored embedding has cosine
                           similarity >= sim_note_threshold.

  * Semantic-reason path — embed each reason text from the previous
                           iteration's prediction; uses sim_icd_threshold.
                           Fires at t>=1 only.

  * Threshold path       — synthesised at runtime from the per-code stats
                           table (no persistent rows). Inactive at t=0.
                           FP gate: code in predicted_set, fpr >= thr.
                           FN gate: code cooccurs with the T=0 prediction,
                                    fnr >= thr (excludes seeds where fpr
                                    >= fpr_threshold to avoid contradictory
                                    add-then-drop suggestions).

After the three paths fire, semantic events are cluster-deduplicated and
per-code-capped (see `retriever_dedup`). The remaining events are then
admitted under per-type caps (FP, FN, semantic) and an optional total cap.

Constants, dataclasses, dedup helpers and synthetic-warning building live in
sibling modules and are re-exported from here for backward compatibility:

    from src.merlin2.retriever import SEM_ICD, THRESHOLD_FPR, ...
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

# Re-export public symbols so external callers (tests, reporting, pipeline)
# can keep importing them from `src.merlin2.retriever`.
from src.merlin2.retriever_paths import (  # noqa: F401
    SEM_ALLERGIES, SEM_COMPLAINT, SEM_EXAM, SEM_FAMILY, SEM_ICD,
    SEM_ILLNESS, SEM_MED_HIST, SEM_MEDICATION, SEM_NOTE, SEM_SOCIAL,
    THRESHOLD_FNR, THRESHOLD_FPR,
    is_semantic_path, section_to_path,
)
from src.merlin2.retriever_events import RetrievalEvent, RetrievalResult  # noqa: F401
from src.merlin2.retriever_synthetic import (  # noqa: F401
    SyntheticInstructionFactory, build_threshold_text, synthetic_instruction_id,
)
from src.merlin2.retriever_dedup import (
    cluster_deduplicate_semantic_events, per_code_cap_semantic_events,
)
from src.meta_verifier.code_stats import CodeStat, CodeStatsIndex
from src.meta_verifier.schemas import Instruction, InstructionType
from src.utils.cooccurrence import CooccurrenceIndex, expand_cooccurring_with_parents
from src.utils.embeddings import encode_single_text

logger = logging.getLogger(__name__)


class Retriever:
    """Threshold-based hybrid retriever over a fixed instruction list."""

    def __init__(
        self,
        sim_note_threshold: float = 0.8,
        sim_icd_threshold: float = 0.8,
        fpr_threshold: float = 0.5,
        fnr_threshold: float = 0.5,
        dedup_cluster_threshold: float = 1.0,
        max_instructions_per_code: Optional[int] = None,
        max_fp_warnings: Optional[int] = None,
        max_fn_warnings: Optional[int] = None,
        max_sem_instructions: Optional[int] = None,
        max_instructions_total: Optional[int] = None,
        cooccurrence_index: Optional[CooccurrenceIndex] = None,
        code_stats: Optional[CodeStatsIndex] = None,
        section_names: Optional[List[str]] = None,
        ignore_phrases: Optional[List[str]] = None,
    ):
        self.sim_note_threshold = sim_note_threshold
        self.sim_icd_threshold = sim_icd_threshold
        self.fpr_threshold = fpr_threshold
        self.fnr_threshold = fnr_threshold
        self.dedup_cluster_threshold = dedup_cluster_threshold
        self.max_instructions_per_code = max_instructions_per_code
        self.max_fp_warnings = max_fp_warnings
        self.max_fn_warnings = max_fn_warnings
        self.max_sem_instructions = max_sem_instructions
        self.max_instructions_total = max_instructions_total
        self._cooccurrence_index: CooccurrenceIndex = cooccurrence_index or {}
        self._code_stats: CodeStatsIndex = code_stats or {}
        self.section_names: List[str] = list(section_names or [])
        self.ignore_phrases: List[str] = list(ignore_phrases or [])

        self._instructions: List[Instruction] = []
        self._synthetic_factory = SyntheticInstructionFactory()

        # Cached embedding matrix, rebuilt lazily.
        self._emb_matrix: Optional[np.ndarray] = None
        self._emb_indices: List[int] = []
        self._emb_norms: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ config
    def load_cooccurrence_index(self, index: CooccurrenceIndex) -> None:
        """Replace the cooccurrence index. Empty dict disables the FN path."""
        self._cooccurrence_index = index or {}

    def load_code_stats(self, stats: CodeStatsIndex) -> None:
        """Replace the per-code stats lookup. Empty dict disables the threshold path."""
        self._code_stats = stats or {}
        self._synthetic_factory.reset()

    def load_instructions(self, instructions: Iterable[Instruction]) -> None:
        self._instructions = list(instructions)
        self._invalidate_cache()
        logger.info(f"Retriever loaded {len(self._instructions)} instructions")

    def add_instruction(self, instruction: Instruction) -> None:
        self._instructions.append(instruction)
        self._invalidate_cache()

    @property
    def instructions(self) -> List[Instruction]:
        """Persistent + synthesised instructions (used by the Pipeline for ID lookup)."""
        return list(self._instructions) + list(self._synthetic_factory.cache.values())

    # Kept for backward compat with main.py's _save_efficacy_scores.
    @property
    def _synthetic_cache(self) -> Dict[int, Instruction]:
        return self._synthetic_factory.cache

    # ----------------------------------------------------- embedding cache
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

    def _has_embedding_cache(self) -> bool:
        return self._emb_matrix is not None and self._emb_matrix.size > 0

    # ------------------------------------------------------------ retrieval
    def retrieve(
        self,
        admission_note: str,
        previous_predicted_codes: Optional[List[str]],
        already_retrieved_ids: Optional[Set[int]] = None,
        previous_reasons: Optional[List[str]] = None,
        note_embedding: Optional[List[float]] = None,
        reason_embeddings: Optional[List[List[float]]] = None,
        note_sections: Optional[Dict[str, str]] = None,
        section_embeddings: Optional[Dict[str, List[float]]] = None,
        t0_predicted_codes: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Retrieve instructions for one case at one iteration.

        Returns a RetrievalResult containing the (deduped, count-capped)
        instruction list and a per-hit log of how each was triggered.
        """
        already_retrieved_ids = already_retrieved_ids or set()
        if self._emb_matrix is None:
            self._build_embedding_cache()

        predicted_set_3digit = frozenset(previous_predicted_codes or [])
        triggered: Dict[int, RetrievalEvent] = {}

        self._fire_semantic_path(
            triggered, admission_note, predicted_set_3digit,
            already_retrieved_ids, note_sections, section_embeddings, note_embedding,
        )
        self._fire_semantic_reason_path(
            triggered, predicted_set_3digit, already_retrieved_ids,
            previous_reasons, reason_embeddings,
        )
        self._fire_threshold_path(
            triggered, previous_predicted_codes, t0_predicted_codes,
            already_retrieved_ids,
        )

        return self._select(triggered)

    # ----------------------------------------- semantic path (note sections)
    def _fire_semantic_path(
        self,
        triggered: Dict[int, RetrievalEvent],
        admission_note: str,
        predicted_set_3digit: frozenset,
        already_retrieved_ids: Set[int],
        note_sections: Optional[Dict[str, str]],
        section_embeddings: Optional[Dict[str, List[float]]],
        note_embedding: Optional[List[float]],
    ) -> None:
        if not self._has_embedding_cache():
            return

        sections_to_query = note_sections or {"_full": admission_note}
        for section_name, section_text in sections_to_query.items():
            raw = self._section_embedding(
                section_name, section_text, section_embeddings, note_embedding, note_sections,
            )
            hits = self._cosine_hits(raw, self.sim_note_threshold)
            if hits is None:
                continue
            sims, hit_indices = hits
            path = section_to_path(section_name)
            for h in hit_indices:
                self._record_event(
                    triggered, h, sims[h], path, predicted_set_3digit, already_retrieved_ids,
                )

    def _section_embedding(
        self,
        section_name: str,
        section_text: str,
        section_embeddings: Optional[Dict[str, List[float]]],
        note_embedding: Optional[List[float]],
        note_sections: Optional[Dict[str, str]],
    ) -> Optional[List[float]]:
        if section_embeddings and section_name in section_embeddings:
            return section_embeddings[section_name]
        if not note_sections and note_embedding is not None:
            return note_embedding
        return encode_single_text(section_text)

    # ----------------------------------------- semantic-reason path
    def _fire_semantic_reason_path(
        self,
        triggered: Dict[int, RetrievalEvent],
        predicted_set_3digit: frozenset,
        already_retrieved_ids: Set[int],
        previous_reasons: Optional[List[str]],
        reason_embeddings: Optional[List[List[float]]],
    ) -> None:
        if not previous_reasons or not self._has_embedding_cache():
            return
        for idx, reason_text in enumerate(previous_reasons):
            if not reason_text.strip():
                continue
            raw = (
                reason_embeddings[idx]
                if reason_embeddings is not None and idx < len(reason_embeddings)
                else encode_single_text(reason_text)
            )
            hits = self._cosine_hits(raw, self.sim_icd_threshold)
            if hits is None:
                continue
            sims, hit_indices = hits
            for h in hit_indices:
                self._record_event(
                    triggered, h, sims[h], SEM_ICD,
                    predicted_set_3digit, already_retrieved_ids,
                )

    # ----------------------------------------- threshold path (FP / FN)
    def _fire_threshold_path(
        self,
        triggered: Dict[int, RetrievalEvent],
        previous_predicted_codes: Optional[List[str]],
        t0_predicted_codes: Optional[List[str]],
        already_retrieved_ids: Set[int],
    ) -> None:
        if not previous_predicted_codes or not self._code_stats:
            return

        predicted_set = set(previous_predicted_codes)
        self._fire_fp_gate(triggered, predicted_set, already_retrieved_ids)
        self._fire_fn_gate(
            triggered, predicted_set, t0_predicted_codes, already_retrieved_ids,
        )

    def _fire_fp_gate(
        self,
        triggered: Dict[int, RetrievalEvent],
        predicted_set: Set[str],
        already_retrieved_ids: Set[int],
    ) -> None:
        for code in predicted_set:
            stat = self._code_stats.get(code)
            if stat is None or stat.fpr is None or stat.fpr < self.fpr_threshold:
                continue
            self._record_synthetic_event(
                triggered, "fp", code, stat, THRESHOLD_FPR, stat.fpr,
                already_retrieved_ids, trigger_codes=None,
            )

    def _fire_fn_gate(
        self,
        triggered: Dict[int, RetrievalEvent],
        predicted_set: Set[str],
        t0_predicted_codes: Optional[List[str]],
        already_retrieved_ids: Set[int],
    ) -> None:
        # Seed from the T=0 (zero-shot) prediction only — using the growing
        # predicted set causes cascade hallucinations as FP codes introduced
        # by FN warnings seed further spurious expansions each iteration.
        # Additionally exclude seeds whose own FPR exceeds the FP threshold:
        # asking the model to drop a code AND seeding its co-occurring
        # neighbours is contradictory.
        fn_seed_base = set(t0_predicted_codes) if t0_predicted_codes is not None else predicted_set
        fn_seed_codes = {c for c in fn_seed_base if not self._is_fp_flagged(c)}

        parents_map = expand_cooccurring_with_parents(self._cooccurrence_index, list(fn_seed_codes))
        # Drop codes already in the current prediction (may have been added by
        # prior instructions in this same case).
        parents_map = {k: v for k, v in parents_map.items() if k not in predicted_set}

        if logger.isEnabledFor(logging.DEBUG) and fn_seed_codes != predicted_set:
            excluded = predicted_set - fn_seed_codes
            logger.debug(
                "FN gate: seeding from %d T=0 codes (excluded %d from full prediction: %s)",
                len(fn_seed_codes), len(excluded), ", ".join(sorted(excluded)),
            )

        for code, trigger_codes in parents_map.items():
            stat = self._code_stats.get(code)
            if stat is None or stat.fnr is None or stat.fnr < self.fnr_threshold:
                continue
            self._record_synthetic_event(
                triggered, "fn", code, stat, THRESHOLD_FNR, stat.fnr,
                already_retrieved_ids, trigger_codes=trigger_codes,
            )

    def _is_fp_flagged(self, code: str) -> bool:
        stat = self._code_stats.get(code)
        return (
            stat is not None and stat.fpr is not None and stat.fpr >= self.fpr_threshold
        )

    # ----------------------------------------- event recording
    def _cosine_hits(self, raw_embedding, threshold):
        emb = np.asarray(raw_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            return None
        dots = self._emb_matrix @ emb
        denom = self._emb_norms * norm
        with np.errstate(invalid="ignore", divide="ignore"):
            sims = np.where(denom > 0, dots / denom, 0.0)
        return sims, np.where(sims >= threshold)[0]

    def _record_event(
        self,
        triggered: Dict[int, RetrievalEvent],
        hit_index: int,
        sim_value: float,
        path: str,
        predicted_set_3digit: frozenset,
        already_retrieved_ids: Set[int],
    ) -> None:
        instr_idx = self._emb_indices[int(hit_index)]
        instr = self._instructions[instr_idx]
        if instr.instruction_id in already_retrieved_ids:
            return
        if instr.instruction_id in triggered:
            # First path to claim this instruction wins.
            return
        if _is_saturated(instr, predicted_set_3digit):
            return
        triggered[instr.instruction_id] = RetrievalEvent(
            instruction_id=instr.instruction_id,
            path=path,
            trigger_value=float(sim_value),
            efficacy_score=instr.efficacy_score,
            target_codes=list(instr.target_codes),
        )

    def _record_synthetic_event(
        self,
        triggered: Dict[int, RetrievalEvent],
        kind: str,
        code: str,
        stat: CodeStat,
        path: str,
        trigger_value: float,
        already_retrieved_ids: Set[int],
        trigger_codes: Optional[List[str]],
    ) -> None:
        instr = self._synthetic_factory.get_or_create(kind, code, stat, trigger_codes)
        if instr.instruction_id in already_retrieved_ids:
            return
        if instr.instruction_id in triggered:
            return
        triggered[instr.instruction_id] = RetrievalEvent(
            instruction_id=instr.instruction_id,
            path=path,
            trigger_value=trigger_value,
            efficacy_score=0.0,
            target_codes=[code],
            trigger_codes=list(trigger_codes) if trigger_codes else [],
        )

    # ----------------------------------------- selection / count caps
    def _select(self, triggered: Dict[int, RetrievalEvent]) -> RetrievalResult:
        id_to_instr = {i.instruction_id: i for i in self.instructions}

        fp_events  = _sorted_by_trigger(triggered, THRESHOLD_FPR)
        fn_events  = _sorted_by_trigger(triggered, THRESHOLD_FNR)
        semantic_events = _sorted_semantic_by_efficacy(triggered)

        semantic_events = self._apply_cluster_dedup(semantic_events)
        semantic_events = self._apply_per_code_cap(semantic_events)

        selected: List[Instruction] = []
        events: List[RetrievalEvent] = []
        skipped = 0

        skipped += self._admit(fp_events,       self.max_fp_warnings,      id_to_instr, selected, events)
        skipped += self._admit(fn_events,       self.max_fn_warnings,      id_to_instr, selected, events)
        skipped += self._admit(semantic_events, self.max_sem_instructions, id_to_instr, selected, events)

        if skipped:
            self._log_cap_hit(skipped, fp_events, fn_events, semantic_events, len(selected))

        return RetrievalResult(
            instructions=selected,
            events=events,
            skipped_for_budget=skipped,
        )

    def _apply_cluster_dedup(self, events: List[RetrievalEvent]) -> List[RetrievalEvent]:
        if self.dedup_cluster_threshold >= 1.0 or len(events) <= 1:
            return events
        id_to_instr = {i.instruction_id: i for i in self._instructions}
        n_before = len(events)
        events = cluster_deduplicate_semantic_events(
            events, id_to_instr, self.dedup_cluster_threshold,
        )
        n_dropped = n_before - len(events)
        if n_dropped:
            logger.debug(
                "Cluster dedup: dropped %d redundant semantic instruction(s) (threshold=%.2f)",
                n_dropped, self.dedup_cluster_threshold,
            )
        return events

    def _apply_per_code_cap(self, events: List[RetrievalEvent]) -> List[RetrievalEvent]:
        if self.max_instructions_per_code is None or len(events) <= 1:
            return events
        n_before = len(events)
        events = per_code_cap_semantic_events(events, self.max_instructions_per_code)
        n_dropped = n_before - len(events)
        if n_dropped:
            logger.debug(
                "Per-code cap: dropped %d semantic instruction(s) (max_per_code=%d)",
                n_dropped, self.max_instructions_per_code,
            )
        return events

    def _admit(
        self,
        events: List[RetrievalEvent],
        per_type_cap: Optional[int],
        id_to_instr: Dict[int, Instruction],
        selected: List[Instruction],
        out_events: List[RetrievalEvent],
    ) -> int:
        """Admit events under (per_type_cap, max_instructions_total). Returns # skipped."""
        admitted = 0
        skipped = 0
        for ev in events:
            if per_type_cap is not None and admitted >= per_type_cap:
                skipped += 1
                continue
            if self.max_instructions_total is not None and len(selected) >= self.max_instructions_total:
                skipped += 1
                continue
            selected.append(id_to_instr[ev.instruction_id])
            out_events.append(ev)
            admitted += 1
        return skipped

    def _log_cap_hit(self, skipped, fp_events, fn_events, sem_events, n_selected) -> None:
        def fmt(cap):
            return str(cap) if cap is not None else "∞"

        logger.debug(
            "Count cap hit: %d skipped  (fp=%d/%s fn=%d/%s sem=%d/%s total=%d/%s)",
            skipped,
            min(len(fp_events),  self.max_fp_warnings      or len(fp_events)),  fmt(self.max_fp_warnings),
            min(len(fn_events),  self.max_fn_warnings      or len(fn_events)),  fmt(self.max_fn_warnings),
            min(len(sem_events), self.max_sem_instructions or len(sem_events)), fmt(self.max_sem_instructions),
            n_selected, fmt(self.max_instructions_total),
        )


# --------------------------------------------------- module-level helpers

def _sorted_by_trigger(triggered: Dict[int, RetrievalEvent], path: str) -> List[RetrievalEvent]:
    return sorted(
        [ev for ev in triggered.values() if ev.path == path],
        key=lambda ev: (-ev.trigger_value, ev.instruction_id),
    )


def _sorted_semantic_by_efficacy(triggered: Dict[int, RetrievalEvent]) -> List[RetrievalEvent]:
    return sorted(
        [ev for ev in triggered.values() if is_semantic_path(ev.path)],
        key=lambda ev: (-ev.efficacy_score, -ev.trigger_value, ev.instruction_id),
    )


def _is_saturated(instr: Instruction, predicted_set_3digit: frozenset) -> bool:
    """Skip semantic instructions whose retrieval would be a no-op.

    action="add":    skip if ALL target codes are already predicted.
    action="remove": skip if NONE of the target codes are predicted.
    contrastive_swap is never suppressed (suppression would require knowing
    which direction the swap runs).
    """
    if instr.type == InstructionType.CONTRASTIVE_SWAP:
        return False
    if not instr.target_codes or not predicted_set_3digit:
        return False
    targets = set(instr.target_codes)
    if instr.action == "add" and targets.issubset(predicted_set_3digit):
        return True
    if instr.action == "remove" and targets.isdisjoint(predicted_set_3digit):
        return True
    return False
