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
from typing import List, Optional

from pydantic import BaseModel, Field


class InstructionType:
    """Discriminator for retrieval and clustering."""
    CONTRASTIVE_SWAP = "contrastive_swap"   # Case-level: predict X INSTEAD OF Y
    SEMANTIC = "semantic"                   # Case-level: generic note-grounded rule
    FP_WARNING = "fp_warning"               # Aggregate: high FPR on a single code
    FN_WARNING = "fn_warning"               # Aggregate: high FNR on a single code


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
    description: str = Field(
        description="Short note-grounded text used as the embedding target for semantic retrieval."
    )
    instruction_text: str = Field(
        description="Thinking-style content injected into the Generator's <think> block."
    )
    related_icd_codes: List[str] = Field(
        default_factory=list,
        description="3-digit ICD codes this error pertains to.",
    )


class Instruction(BaseModel):
    """A single retrievable instruction record.

    Persisted (parquet store): only SEMANTIC and CONTRASTIVE_SWAP types.
    Each row has `semantic_embedding` set; the Retriever fires it via the
    semantic path. `fpr_at_creation` / `fnr_at_creation` are None and
    `efficacy_score` is updated online during Loop A.

    Synthesised at runtime (FP_WARNING / FN_WARNING): produced by the
    Retriever from the per-code stats table (`code_stats.parquet`); they
    are NOT persisted in the instruction store. `target_codes` has length
    1, `semantic_embedding` is None, `efficacy_score` is always 0.0
    (threshold warnings have no efficacy tracking).
    """
    instruction_id: int

    type: str = InstructionType.SEMANTIC
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
