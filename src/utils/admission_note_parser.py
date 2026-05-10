"""Admission note section parser.

Splits a raw admission note into named sections and filters out sections
whose content carries no useful signal (empty, too short, or matching a
known boilerplate phrase).

The section list and ignore phrases are configured in
``configs/admission_note_sections.yaml`` so researchers can adapt them
without touching Python code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "admission_note_sections.yaml"


def load_section_config(path: str | Path = _DEFAULT_CONFIG) -> Dict[str, List[str]]:
    """Load section names and ignore phrases from a YAML config file."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def parse_sections(note: str, section_names: List[str]) -> Dict[str, str]:
    """Split *note* into {section_name: content} using *section_names* as
    delimiters.

    Headers are matched case-insensitively and must be followed by a colon
    (``CHIEF COMPLAINT:``). Content between two recognised headers is
    assigned to the preceding one. Text before the first recognised header
    is discarded. Unknown headers embedded in content are absorbed into the
    preceding section.

    Ordering of the returned dict mirrors the order sections appear in the
    note, not the order of *section_names*.
    """
    if not note or not section_names:
        return {}

    # Build a pattern that matches any known section header.
    # Each name is re.escaped then joined with alternation.
    escaped = [re.escape(name) for name in section_names]
    pattern = re.compile(
        r"(?:^|\n)\s*(" + "|".join(escaped) + r")\s*:",
        re.IGNORECASE,
    )

    result: Dict[str, str] = {}
    current_key: str | None = None
    last_end = 0

    for match in pattern.finditer(note):
        if current_key is not None:
            content = note[last_end:match.start()].strip()
            result[current_key] = content
        # Normalise to the canonical casing from section_names
        matched_text = match.group(1).upper()
        current_key = next(
            (name for name in section_names if name.upper() == matched_text),
            matched_text,
        )
        last_end = match.end()

    if current_key is not None:
        result[current_key] = note[last_end:].strip()

    return result


def filter_sections(
    sections: Dict[str, str],
    ignore_phrases: List[str],
    min_chars: int = 10,
) -> Dict[str, str]:
    """Return a copy of *sections* with empty / boilerplate entries removed.

    A section is dropped when its stripped content:
    - is empty or shorter than *min_chars*, OR
    - matches any phrase in *ignore_phrases* (case-insensitive, after
      stripping leading/trailing whitespace).
    """
    normalised_ignores = {p.strip().lower() for p in ignore_phrases}
    out: Dict[str, str] = {}
    for name, content in sections.items():
        stripped = content.strip()
        if len(stripped) < min_chars:
            continue
        if stripped.lower() in normalised_ignores:
            continue
        out[name] = stripped
    return out
