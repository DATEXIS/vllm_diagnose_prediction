from typing import List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd

class ICDPrediction(BaseModel):
    icd_code: str = Field(description="The ICD code for the diagnosis.")
    reason: str = Field(description="Clinical reasoning for assigning this code based on the admission note.")

class ICDsModel(BaseModel):
    diagnoses: List[ICDPrediction] = Field(
        description="A list of predicted ICD codes with clinical reasoning."
    )

def build_prompt(patient: Dict[str, Any]) -> str:
    """
    Constructs the prompt for the language model.
    Assumes 'admission_note' is in the patient dictionary.
    """
    admission_note = patient.get('admission_note', 'No note available.')
    
    # We use a simple but effective prompt for ICD prediction.
    # We can expand this with few-shot examples or manifestations if available in the dataset.
    system_instruction = (
        "You are an expert medical coder. Your task is to extract all relevant "
        "diagnoses from the provided medical admission note and assign the most appropriate "
        "ICD (International Classification of Diseases) codes for each. "
        "Provide a concise reason (1-2 sentences) for each assigned ICD code based on the clinical evidence in the text."
    )
    
    prompt = f"{system_instruction}\n\n### Admission Note:\n{admission_note}\n\n### Extracted ICD Codes:\n"
    return prompt

def build_prompts(df: pd.DataFrame) -> List[str]:
    """Builds a list of prompts from a dataframe of patients."""
    patient_dicts = df.to_dict(orient="records")
    return [build_prompt(patient) for patient in patient_dicts]

def get_schema() -> Dict[str, Any]:
    """Returns the JSON schema for guided decoding."""
    return {
        "name": ICDsModel.__name__,
        "schema": ICDsModel.model_json_schema(),
        "strict": True
    }
