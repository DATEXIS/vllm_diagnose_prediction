"""Retrieval-path constants and helpers.

Every retrieval hit (`RetrievalEvent`) carries a `path` string telling us
which gate fired it. The Retriever uses these tags for logging, dedup, and
the per-type count caps. They are split into:

  * Semantic paths (`sem_*`) — one per known admission note section, plus
    `sem_icd` for the ICD-reasoning path and `sem_note` as the fallback
    when no section breakdown is available.
  * Threshold path — `threshold_fpr` for the false-positive warning gate.
"""

from __future__ import annotations

from typing import Dict


THRESHOLD_FPR = "threshold_fpr"

# Per-section semantic paths — one constant per known admission note section.
SEM_COMPLAINT  = "sem_complaint"    # CHIEF COMPLAINT
SEM_ILLNESS    = "sem_illness"      # PRESENT ILLNESS
SEM_MED_HIST   = "sem_med_hist"     # MEDICAL HISTORY
SEM_MEDICATION = "sem_medication"   # MEDICATION ON ADMISSION
SEM_ALLERGIES  = "sem_allergies"    # ALLERGIES
SEM_EXAM       = "sem_exam"         # PHYSICAL EXAM
SEM_FAMILY     = "sem_family"       # FAMILY HISTORY
SEM_SOCIAL     = "sem_social"       # SOCIAL HISTORY
SEM_ICD        = "sem_icd"          # ICD-level reasoning (replaces semantic_reason)
SEM_NOTE       = "sem_note"         # fallback: full note / unknown section

_SECTION_TO_PATH: Dict[str, str] = {
    "CHIEF COMPLAINT":         SEM_COMPLAINT,
    "PRESENT ILLNESS":         SEM_ILLNESS,
    "MEDICAL HISTORY":         SEM_MED_HIST,
    "MEDICATION ON ADMISSION": SEM_MEDICATION,
    "ALLERGIES":               SEM_ALLERGIES,
    "PHYSICAL EXAM":           SEM_EXAM,
    "FAMILY HISTORY":          SEM_FAMILY,
    "SOCIAL HISTORY":          SEM_SOCIAL,
}


def section_to_path(section_name: str) -> str:
    """Map an admission note section name to its retrieval path constant."""
    return _SECTION_TO_PATH.get(section_name.upper(), SEM_NOTE)


def is_semantic_path(path: str) -> bool:
    """True for any sem_* path (note sections or ICD reasoning)."""
    return path.startswith("sem_")
