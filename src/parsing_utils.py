import json
import logging
import re
from typing import List, Optional, Dict, Any
from prompter import ICDsModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)

def repair_json_truncation(json_str: str) -> str:
    """
    Attempts to repair a truncated JSON string by closing open structures.
    Handles mid-string, mid-object, and mid-list truncation.
    """
    json_str = json_str.strip()
    if not json_str:
        return json_str

    # 1. Close open strings
    # We count unescaped double quotes
    quotes = 0
    i = 0
    while i < len(json_str):
        if json_str[i] == '"':
            if i == 0 or json_str[i-1] != '\\':
                quotes += 1
        i += 1
    
    if quotes % 2 != 0:
        json_str += '"'

    # 2. Close brackets and braces using a stack-based approach
    stack = []
    i = 0
    # Re-scan to find balance (this is more robust than counting)
    in_string = False
    while i < len(json_str):
        char = json_str[i]
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
        i += 1
    
    # Append the necessary closing characters in reverse order
    for closer in reversed(stack):
        json_str += closer
        
    return json_str

def safe_parse_json(text: str) -> List[str]:
    """
    Tries to parse the LLM output as JSON and extract the ICD codes.
    Uses a multi-stage approach with specialized repair for truncation.
    """
    if not text:
        logger.debug("safe_parse_json: empty text")
        return []

    # Pre-processing
    original_text = text.strip()
    logger.debug(f"safe_parse_json input (first 500ch): {original_text[:500]}")
    logger.debug(f"safe_parse_json input (last 500ch): {original_text[-500:]}")
    
    def try_validate(candidate: str) -> Optional[List[str]]:
        try:
            # Stage A: Direct Pydantic validation
            validated = ICDsModel.model_validate_json(candidate)
            return [d.icd_code for d in validated.diagnoses if d.icd_code]
        except (ValidationError, ValueError, json.JSONDecodeError):
            try:
                # Stage B: Repair common LLM mistakes (literal newlines/tabs)
                repaired = candidate.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                validated = ICDsModel.model_validate_json(repaired)
                return [d.icd_code for d in validated.diagnoses if d.icd_code]
            except (ValidationError, ValueError, json.JSONDecodeError):
                return None

    # 1. Handle "Thinking" / Preamble blocks
    # We isolate the JSON block by finding the first '{'
    json_start = original_text.find('{')
    if json_start != -1:
        candidate = original_text[json_start:]
    else:
        candidate = original_text

    # 2. Try parsing the candidate (which might be the full text or just the JSON start)
    codes = try_validate(candidate)
    
    # 3. If failed, attempt truncation repair
    if codes is None:
        repaired_candidate = repair_json_truncation(candidate)
        codes = try_validate(repaired_candidate)

    # 4. Handle results
    if codes is not None:
        return codes
    
    # Final logging if all else fails
    logger.warning(f"Failed to parse or validate JSON after repair. Preview (1000ch):\n{original_text[:1000]}")
    return []
