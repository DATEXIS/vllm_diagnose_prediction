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
    instruction_reasoning: str = Field(
        default="",
        description=(
            "Step-by-step reasoning through every item in the <coding_review> block "
            "before finalizing the diagnosis list. For each FP warning: state whether "
            "explicit evidence was found and whether the code is kept or removed. "
            "For each FN warning and semantic instruction: state whether a supporting "
            "cue exists. Empty string when no coding_review block is present."
        ),
    )
    diagnoses: List[ICDPrediction] = Field(
        description="A list of predicted ICD codes with clinical reasoning.",
    )


def get_schema(instruction_reasoning: bool = True) -> Dict[str, Any]:
    """JSON schema for vLLM guided decoding.

    minItems: 3 is injected here rather than on the Pydantic model so that
    vLLM enforces the constraint at generation time without breaking the
    Python-side ICDsModel(diagnoses=[]) sentinel used in parse-failure paths.

    When instruction_reasoning=False the field is dropped from the schema so
    the model doesn't burn tokens on it and guided decoding doesn't enforce it.
    """
    schema = ICDsModel.model_json_schema()
    schema.get("properties", {}).get("diagnoses", {})["minItems"] = 3
    if not instruction_reasoning:
        schema.get("properties", {}).pop("instruction_reasoning", None)
        required = schema.get("required", [])
        if "instruction_reasoning" in required:
            required.remove("instruction_reasoning")
    return {
        "name": ICDsModel.__name__,
        "schema": schema,
        "strict": True,
    }
