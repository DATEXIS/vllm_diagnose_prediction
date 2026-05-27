"""Instruction schemas for MERLIN 2.

The `Instruction` model is the canonical record stored in the parquet
instruction database (see src/meta_verifier/store.py). It is consumed by
the Retriever during Loop A and produced by the Meta-Verifier during
Loop B.

The `RichErrorInstruction` model is the JSON output schema we ask the
Meta-Verifier LLM to emit. It is converted into one or more
`Instruction` records before being stored.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InstructionType:
    """Discriminator for retrieval and clustering."""
    CONTRASTIVE_SWAP = "contrastive_swap"   # Case-level: predict X INSTEAD OF Y
    SEMANTIC = "semantic"                   # Case-level: generic note-grounded rule
    FP_WARNING = "fp_warning"               # Aggregate: high FPR on a single code


class RichErrorInstruction(BaseModel):
    """LLM-generated error analysis output from the Meta-Verifier.

    Emitted for each error the Meta-Verifier identifies in a case. The
    Meta-Verifier converts these into `Instruction` records before
    storage; one `RichErrorInstruction` may produce multiple
    `Instruction` rows (one per related ICD code in the contrastive case).
    """
    type: str = Field(
        description="One of InstructionType values; defaults to SEMANTIC if the LLM does not classify."
    )
    section: str = Field(
        description=(
            "The admission note section this instruction is grounded in "
            "(e.g. 'CHIEF COMPLAINT', 'MEDICAL HISTORY', 'PHYSICAL EXAM') "
            "or 'icd_reasoning' when grounded in code-level reasoning only."
        )
    )
    description: str = Field(
        description="Short note-grounded text used as the embedding target for semantic retrieval."
    )
    instruction_text: str = Field(
        description="Thinking-style content injected into the Generator's <think> block."
    )
    related_icd_codes: List[str] = Field(
        default_factory=list,
        description="ICD codes this error pertains to (full codes; evaluated at 3-digit level).",
    )
    action: Literal["add", "remove"] = Field(
        default="add",
        description=(
            "'add' if the instruction corrects a missed code (FN — include this code); "
            "'remove' if the instruction corrects a hallucinated code (FP — drop this code)."
        ),
    )


class Instruction(BaseModel):
    """A single retrievable instruction record.

    Persisted (parquet store): only SEMANTIC and CONTRASTIVE_SWAP types.
    Each row has `semantic_embedding` set; the Retriever fires it via the
    semantic path. `fpr_at_creation` / `fnr_at_creation` are None and
    `efficacy_score` is updated online during Loop A.

    Synthesised at runtime (FP_WARNING): produced by the Retriever from
    the per-code stats table (`code_stats.parquet`); NOT persisted in the
    instruction store. `target_codes` has length 1, `semantic_embedding`
    is None, `efficacy_score` is always 0.0.
    """
    instruction_id: int

    type: str = InstructionType.SEMANTIC
    # Whether this instruction recommends adding or removing its target codes.
    # "add"    → FN correction: tell the Generator to include the code(s).
    # "remove" → FP correction: tell the Generator to drop the code(s).
    # Defaults to "add" for backward-compat with rows that pre-date this field.
    # Synthesised FP_WARNING sets this to "remove" in the Retriever.
    action: Literal["add", "remove"] = "add"
    # Admission note section this instruction is grounded in, or "icd_reasoning".
    # Empty string for legacy rows loaded from parquet (backward compat).
    section: str = ""
    instruction_text: str
    description: str = ""

    # 3-digit ICD codes. Length 1 for threshold instructions; >=1 for semantic.
    target_codes: List[str] = Field(default_factory=list)

    # Provenance: hadm_ids of the cases this instruction was derived from.
    # Required (non-empty) for semantic instructions; empty list for threshold instructions.
    source_hadm_ids: List[str] = Field(default_factory=list)

    # Frozen metric snapshots. Only set for threshold instructions.
    fpr_at_creation: Optional[float] = None
    fnr_at_creation: Optional[float] = None

    # Updated online during Loop A on training data.
    efficacy_score: float = 0.0

    # Brute-force cosine retrieval over this vector. Required for semantic
    # instructions; threshold instructions may store an embedding too so
    # they can also fire via the semantic path.
    semantic_embedding: Optional[List[float]] = None

    created_at: datetime = Field(default_factory=datetime.now)
