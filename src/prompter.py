"""Pydantic schemas for the Generator's structured output.

Prompt construction lives in `src.merlin2.generator` (templates in
`configs/prompts/`); this module owns only the schema.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ICDPrediction(BaseModel):
    icd_code: str = Field(description="The ICD code for the diagnosis.")
    reason: str = Field(
        description="Clinical reasoning for assigning this code based on the admission note."
    )


class ICDsModel(BaseModel):
    diagnoses: List[ICDPrediction] = Field(
        description="A list of predicted ICD codes with clinical reasoning.",
        max_length=15,
    )


def get_schema() -> Dict[str, Any]:
    """JSON schema for vLLM guided decoding."""
    return {
        "name": ICDsModel.__name__,
        "schema": ICDsModel.model_json_schema(),
        "strict": True,
    }
