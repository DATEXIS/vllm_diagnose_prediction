"""Tiny prompt-template loader.

Prompts live in `configs/prompts/` as plain text files. They are loaded
once at startup and formatted with `str.format(**kwargs)` at use time.
Kept dead simple on purpose — no Jinja, no caching surprises, no
fallbacks. If a placeholder is missing, KeyError is the right answer.
"""

import json
from functools import lru_cache
from pathlib import Path

# Repo root is two levels up from this file: src/utils/prompt_loader.py
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by short name, e.g. "generator_system".

    Cached for the process lifetime; that is fine for research code where
    prompts only change between runs.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


# Canonical JSON example we ask the Generator to follow. Defined here
# (not in the prompt file) so it stays in sync with the pydantic schema.
GENERATOR_JSON_EXAMPLE = json.dumps(
    [
        {
            "icd_code": "I10",
            "reason": "Patient has persistent hypertension noted in the admission note.",
        },
        {
            "icd_code": "E11.9",
            "reason": "Elevated blood glucose levels indicating type 2 diabetes mellitus.",
        },
    ],
    indent=2,
)


META_VERIFIER_JSON_EXAMPLE = json.dumps(
    [
        {
            "type": "contrastive_swap",
            "section": "PRESENT ILLNESS",
            "description": "Mention of diabetic neuropathy without explicit DKA cues.",
            "instruction_text": (
                "If the note mentions diabetic neuropathy, prefer E11.4 over the "
                "unspecified E11.9."
            ),
            "related_icd_codes": ["E11"],
        }
    ],
    indent=2,
)
