"""Batched note-section and reason-text embedding for the Pipeline.

Each live case contributes its admission-note sections plus the reason
texts from its latest prediction. The Pipeline pools all texts into a
single `encode_texts` call so the embedding model is fully utilised, then
slices the result back per-case for the Retriever.

The state of each case is unchanged; this module returns a `BatchedEmbeddings`
object the Pipeline uses to feed `Retriever.retrieve(...)` for each case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.merlin2.pipeline_state import CaseState
from src.utils.embeddings import encode_texts


@dataclass
class BatchedEmbeddings:
    embeddings: List[List[float]]
    section_keys_per_case: List[List[str]]
    section_slices: List[Tuple[int, int]]
    reason_slices: List[Tuple[int, int]]
    reasons_per_case: List[List[str]]

    def section_embeddings(self, case_idx: int) -> Dict[str, List[float]]:
        sec_start, _ = self.section_slices[case_idx]
        return {
            key: self.embeddings[sec_start + j]
            for j, key in enumerate(self.section_keys_per_case[case_idx])
        }

    def reason_embeddings(self, case_idx: int) -> List[List[float]]:
        r_start, r_end = self.reason_slices[case_idx]
        return self.embeddings[r_start:r_end]

    def reasons(self, case_idx: int) -> List[str]:
        return self.reasons_per_case[case_idx]


def embed_cases(states: List[CaseState]) -> BatchedEmbeddings:
    """Collect and embed all per-case texts in one pooled `encode_texts` call."""
    all_texts: List[str] = []
    section_keys_per_case: List[List[str]] = []
    section_slices: List[Tuple[int, int]] = []
    reason_slices: List[Tuple[int, int]] = []
    reasons_per_case: List[List[str]] = []

    for s in states:
        sec_keys, sec_texts = _section_keys_and_texts(s)
        section_keys_per_case.append(sec_keys)
        section_slices.append(_append_slice(all_texts, sec_texts))

        reasons = _reasons_from_latest_prediction(s)
        reasons_per_case.append(reasons)
        reason_slices.append(_append_slice(all_texts, reasons))

    embeddings = encode_texts(all_texts) if all_texts else []
    return BatchedEmbeddings(
        embeddings=embeddings,
        section_keys_per_case=section_keys_per_case,
        section_slices=section_slices,
        reason_slices=reason_slices,
        reasons_per_case=reasons_per_case,
    )


def _section_keys_and_texts(state: CaseState) -> Tuple[List[str], List[str]]:
    if state.note_sections:
        keys = list(state.note_sections.keys())
        texts = [state.note_sections[k] for k in keys]
        return keys, texts
    return ["_full"], [state.admission_note]


def _reasons_from_latest_prediction(state: CaseState) -> List[str]:
    return [d.reason for d in state.predictions[-1].diagnoses if d.reason.strip()]


def _append_slice(target: List[str], items: List[str]) -> Tuple[int, int]:
    start = len(target)
    target.extend(items)
    return start, start + len(items)
