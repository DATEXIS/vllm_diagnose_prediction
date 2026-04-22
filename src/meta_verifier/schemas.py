from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class ErrorType:
    """Classification of prediction errors for instruction generation."""
    FALSE_POSITIVE = "false_positive"  # Predicted but not in ground truth
    FALSE_NEGATIVE = "false_negative"  # In ground truth but not predicted
    REASONING_ERROR = "reasoning_error"  # Correct code but wrong reason
    PARTIAL_MATCH = "partial_match"  # Code partially correct (e.g., less specific)


class RichErrorInstruction(BaseModel):
    """LLM-generated error analysis instruction from meta-verifier."""
    error_type: str  # type / category / keyword
    description: str  # technical explanation referencing admission note
    instructions: str  # instruction to mitigate this error in future iterations
    related_icd_codes: List[str] = Field(default_factory=list)


class Instruction(BaseModel):
    id: int
    target_code: str
    contrastive_rule: str
    error_type: str = "unknown"
    quote: str = ""
    fpr: float = 0.0
    fnr: float = 0.0
    efficacy_score: float = 0.0
    semantic_embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)