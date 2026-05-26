"""Minimal JSON parsing for LLM responses.

Four output formats are accepted, tried in order:

  1. JSON object with `instruction_reasoning` + `diagnoses` (primary / new format)
       {"instruction_reasoning": "...", "diagnoses": [{icd_code, reason}, ...]}
     The model is instructed to emit this. The reasoning field is extracted and
     stored; it may be empty when no coding_review block was present.

  2. JSON object with a `diagnoses` key only (legacy object format)
       {"diagnoses": [{icd_code, reason}, ...]}
     Kept for backward compatibility.

  3. JSON array (legacy array format)
       [{icd_code, reason}, ...]

  4. Single ICDPrediction dict (common mis-generation)
       {"icd_code": "N20", "reason": "..."}
     Wraps the lone item into a single-diagnosis ICDsModel so the case is
     not silently dropped.

<think>…</think> blocks emitted by reasoning models are stripped before
any extraction is attempted.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from pydantic import ValidationError

from src.prompter import ICDPrediction, ICDsModel

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Matches an unclosed <think> block — model ran out of tokens mid-thinking.
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)


class JSONExtractionError(ValueError):
    """Raised when no valid prediction JSON can be extracted from a response."""


def extract_model_thinking(text: str) -> str:
    """Return the content of the first <think>…</think> block, or "" if absent.

    Captures the model's own chain-of-thought reasoning, separate from the
    <coding_review> block that was injected into the prompt.
    """
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------- stripping
def _strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> sections emitted by reasoning models.

    Also strips unclosed <think> blocks (model hit max_tokens during thinking
    and never emitted </think>). Without this, a response that is entirely
    an incomplete think block is passed as-is to the JSON extractors, which
    then fail on the raw <think> text.
    """
    # Strip complete blocks first, then any trailing unclosed one.
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# --------------------------------------------------------------- extractors
def extract_last_json_object(text: str) -> str:
    """Return the substring containing the last balanced `{...}` block.

    Raises JSONExtractionError if none is found.
    """
    n = len(text)
    i = n - 1
    while i >= 0:
        if text[i] == "}":
            depth = 0
            j = i
            in_string = False
            escape = False
            while j >= 0:
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "}":
                        depth += 1
                    elif ch == "{":
                        depth -= 1
                        if depth == 0:
                            candidate = text[j : i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
                j -= 1
            i = j - 1
        else:
            i -= 1

    raise JSONExtractionError("No balanced JSON object found in response.")


def extract_last_json_array(text: str) -> str:
    """Return the substring containing the last balanced `[...]` block.

    Raises JSONExtractionError if none is found.
    """
    n = len(text)
    i = n - 1
    while i >= 0:
        if text[i] == "]":
            depth = 0
            j = i
            in_string = False
            escape = False
            while j >= 0:
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "]":
                        depth += 1
                    elif ch == "[":
                        depth -= 1
                        if depth == 0:
                            candidate = text[j : i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
                j -= 1
            i = j - 1
        else:
            i -= 1

    raise JSONExtractionError("No balanced JSON array found in response.")


# --------------------------------------------------------------- main parser
def parse_prediction(response: str) -> ICDsModel:
    """Parse an LLM response into an ICDsModel.

    Tries formats in order:
      1. JSON array   → each element validated as ICDPrediction
      2. JSON object  → validated as ICDsModel (diagnoses key required)
      3. Single dict  → validated as ICDPrediction, wrapped as 1-item ICDsModel

    Think blocks are stripped before any extraction.
    Raises JSONExtractionError if all three attempts fail.
    """
    if not response:
        raise JSONExtractionError("Empty response.")

    text = _strip_think_blocks(response) or response
    errors: List[str] = []

    # ---- 1. {instruction_reasoning, diagnoses} object (primary format) --
    try:
        obj_str = extract_last_json_object(text)
        raw_obj = json.loads(obj_str)

        if isinstance(raw_obj, dict) and "diagnoses" in raw_obj:
            try:
                return ICDsModel.model_validate(raw_obj)
            except ValidationError as e:
                errors.append(f"Object (with diagnoses) invalid: {e}")

        # ---- 4. single ICDPrediction dict --------------------------------
        elif isinstance(raw_obj, dict) and "icd_code" in raw_obj:
            try:
                single = ICDPrediction.model_validate(raw_obj)
                logger.debug("Wrapped single ICDPrediction dict into ICDsModel")
                return ICDsModel(diagnoses=[single])
            except ValidationError as e:
                errors.append(f"Single dict invalid: {e}")
        else:
            errors.append("Object has neither 'diagnoses' nor 'icd_code' key")

    except (JSONExtractionError, json.JSONDecodeError) as e:
        errors.append(f"No object: {e}")

    # ---- 3. legacy array [{icd_code, reason}, ...] ----------------------
    try:
        arr_str = extract_last_json_array(text)
        raw_list = json.loads(arr_str)
        if isinstance(raw_list, list):
            try:
                diagnoses = [
                    ICDPrediction.model_validate(item)
                    for item in raw_list
                    if isinstance(item, dict)
                ]
                if diagnoses:
                    return ICDsModel(diagnoses=diagnoses)
                errors.append("Array was empty or contained no valid dicts")
            except ValidationError as e:
                errors.append(f"Array items invalid: {e}")
        else:
            errors.append("Parsed array token was not a list")
    except (JSONExtractionError, json.JSONDecodeError) as e:
        errors.append(f"No array: {e}")

    # ---- 4. truncated-object recovery: instruction_reasoning ate the budget,
    #         but "diagnoses" array was still emitted before truncation. ------
    diagnoses_match = re.search(r'"diagnoses"\s*:\s*(\[)', text)
    if diagnoses_match:
        try:
            arr_str = extract_last_json_array(text[diagnoses_match.start(1):])
            raw_list = json.loads(arr_str)
            if isinstance(raw_list, list):
                diagnoses = [
                    ICDPrediction.model_validate(item)
                    for item in raw_list
                    if isinstance(item, dict)
                ]
                if diagnoses:
                    logger.warning("Recovered diagnoses from truncated JSON object.")
                    return ICDsModel(diagnoses=diagnoses)
                errors.append("Truncated-object recovery: diagnoses array empty")
        except (JSONExtractionError, json.JSONDecodeError, ValidationError) as e:
            errors.append(f"Truncated-object recovery failed: {e}")

    post_think = _strip_think_blocks(response)
    if post_think:
        context = f"Post-think content ({len(post_think)} chars): {post_think[:400]!r}"
    else:
        # Think block was never closed — show the tail so we can see where
        # the model stopped (e.g. mid-sentence if max_tokens was hit)
        context = (
            f"Think block incomplete (response {len(response)} chars, likely truncated). "
            f"Tail: {response[-300:]!r}"
        )
    raise JSONExtractionError(
        f"Could not parse prediction. Attempts: {'; '.join(errors)}. {context}"
    )


def parse_prediction_codes(response: str) -> List[str]:
    """Convenience wrapper returning just the predicted ICD code strings."""
    model = parse_prediction(response)
    return [d.icd_code for d in model.diagnoses]
