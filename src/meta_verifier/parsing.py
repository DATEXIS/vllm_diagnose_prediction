"""JSON parsing helpers for Meta-Verifier responses.

Reasoning models (e.g. Qwen3) prepend a <think>…</think> block before
their JSON output, sometimes truncated. We strip those first, then walk
back from the last `]` to find a balanced array. Each parsed item is
validated independently so a single malformed entry doesn't discard the
rest of the case.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from pydantic import ValidationError

from src.meta_verifier.schemas import RichErrorInstruction

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> sections emitted by reasoning models."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def extract_json_list(text: str) -> List[dict]:
    """Find the last balanced `[...]` and json.loads it.

    Behavior:
      * A complete think block followed by valid JSON → JSON is found.
      * A truncated think block (max_tokens hit mid-think, no JSON) →
        stripped text is empty → ValueError with the original preview.
    """
    if not text:
        raise ValueError("Empty Meta-Verifier response.")

    search_text = strip_think_blocks(text) or text
    start, end = _find_last_balanced_array(search_text)
    if start < 0 or end < 0:
        raise ValueError(
            f"No balanced '[...]' in Meta-Verifier response. Preview: {text[:300]!r}"
        )
    parsed = json.loads(search_text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed).__name__}")
    return parsed


def _find_last_balanced_array(text: str) -> tuple[int, int]:
    """Return (start, end) of the last balanced `[...]` substring, or (-1, -1)."""
    end = text.rfind("]")
    if end < 0:
        return -1, -1

    depth = 0
    in_str = False
    escape = False
    for i in range(end, -1, -1):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "]":
            depth += 1
        elif ch == "[":
            depth -= 1
            if depth == 0:
                return i, end
    return -1, -1


def validate_rich_items(items: List[dict]) -> List[RichErrorInstruction]:
    """Validate each item individually; skip non-dicts and invalid items.

    The LLM occasionally emits bare ICD strings (e.g. "S72") alongside
    proper dicts. Skipping bad entries per-item lets us salvage the rest
    of the case rather than discarding it entirely.
    """
    out: List[RichErrorInstruction] = []
    for raw in items:
        if not isinstance(raw, dict):
            logger.debug("Meta-Verifier: skipping non-dict item in JSON list: %r", raw)
            continue
        try:
            out.append(RichErrorInstruction.model_validate(raw))
        except ValidationError as e:
            logger.debug("Meta-Verifier: skipping invalid item %r: %s", raw, e)
    return out
