"""Minimal JSON parsing for LLM responses.

Philosophy (per AGENTS.md): no repair, no regex fallbacks, no silent
failures. Find the last valid JSON object in the response and validate
it against the prediction schema. If the model's output cannot be
parsed, raise — the pipeline is meant to crash so the issue surfaces.
"""

import json
import logging
from typing import List

from pydantic import ValidationError

from src.prompter import ICDsModel

logger = logging.getLogger(__name__)


class JSONExtractionError(ValueError):
    """Raised when no valid prediction JSON can be extracted from a response."""


def extract_last_json_object(text: str) -> str:
    """Return the substring of `text` containing the last balanced top-level
    JSON object (i.e. the last `{...}` block whose braces balance).

    Scans backwards from the end of the string. Strings (with escape
    handling) are skipped so that braces inside string literals are not
    counted as structural.

    Raises JSONExtractionError if no balanced `{...}` block exists.
    """
    n = len(text)
    # Walk from the end looking for closing braces; for each, find the
    # matching opening brace and check that the slice parses as JSON.
    i = n - 1
    while i >= 0:
        if text[i] == '}':
            depth = 0
            j = i
            in_string = False
            escape = False
            while j >= 0:
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '}':
                        depth += 1
                    elif ch == '{':
                        depth -= 1
                        if depth == 0:
                            candidate = text[j:i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                # Not balanced/valid; keep scanning further left
                                # for an earlier closing brace.
                                break
                j -= 1
            i = j - 1
        else:
            i -= 1

    raise JSONExtractionError(
        f"No balanced JSON object found in response. Preview: {text[:300]!r}"
    )


def parse_prediction(response: str) -> ICDsModel:
    """Parse an LLM response into an ICDsModel.

    Extracts the last balanced JSON object and validates it. Raises on
    failure (no repair, no regex extraction).
    """
    if not response:
        raise JSONExtractionError("Empty response.")
    candidate = extract_last_json_object(response)
    try:
        return ICDsModel.model_validate_json(candidate)
    except ValidationError as e:
        raise JSONExtractionError(
            f"Extracted JSON did not match ICDsModel schema: {e}\nCandidate: {candidate[:500]}"
        ) from e


def parse_prediction_codes(response: str) -> List[str]:
    """Convenience wrapper returning just the predicted ICD code strings."""
    model = parse_prediction(response)
    return [d.icd_code for d in model.diagnoses]
