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
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by short name, e.g. "generator_system".

    Cached for the process lifetime; that is fine for research code where
    prompts only change between runs.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


# General coding guidelines — loaded once, injected into every system prompt.
# Edit configs/general_guidelines.txt to add or revise principles without any
# code changes. Set to an empty string to disable.
_GUIDELINES_PATH = CONFIGS_DIR / "general_guidelines.txt"
GENERAL_GUIDELINES: str = _GUIDELINES_PATH.read_text()

# Canonical JSON example we ask the Generator to follow. Defined here
# (not in the prompt file) so it stays in sync with the pydantic schema.
_EXAMPLE_DIAGNOSES = [
    {
        "icd_code": "I10",
        "reason": "Patient has persistent hypertension noted in the admission note.",
    },
    {
        "icd_code": "E11.9",
        "reason": "Elevated blood glucose levels; type 2 diabetes documented in PMH.",
    },
    {
        "icd_code": "G47.33",
        "reason": "Obstructive sleep apnea with documented nightly CPAP use.",
    },
]


def build_json_example(instruction_reasoning: bool = True) -> str:
    """Build the JSON example injected into the generator system prompt.

    When instruction_reasoning=False the field is omitted so the prompt
    matches the guided-decoding schema the model is actually constrained to.
    """
    obj: dict = {}
    if instruction_reasoning:
        obj["instruction_reasoning"] = (
            "E11.9 → 'type 2 diabetes documented in PMH, on metformin' → KEEP. "
            "G47.33 → 'uses CPAP nightly for OSA' → KEEP. "
            "I10 → 'persistent hypertension noted on admission' → KEEP."
        )
    obj["diagnoses"] = _EXAMPLE_DIAGNOSES
    return json.dumps(obj, indent=2)


# Kept for backward compatibility with any import that uses the constant directly.
GENERATOR_JSON_EXAMPLE = build_json_example(instruction_reasoning=True)

# Field-description bullet injected into the system prompt when instruction_reasoning is on.
INSTRUCTION_REASONING_FIELD_DOC = (
    "\n  - instruction_reasoning: work through every item in the LAST\n"
    "    <coding_review> block before finalizing diagnoses. One phrase per item:\n"
    "      [CODE] → [direct evidence in note, or \"none\"] → REMOVE or KEEP\n"
    "    For semantic/FN instructions use ADD or SKIP instead of REMOVE/KEEP.\n"
    "    Earlier blocks are context only. Leave \"\" if no block is present.\n"
)


META_VERIFIER_JSON_EXAMPLE = json.dumps(
    [
        {
            "type": "contrastive_swap",
            "action": "add",
            "section": "PRESENT ILLNESS",
            "description": "Mention of diabetic neuropathy without explicit DKA cues.",
            "instruction_text": (
                "If the note mentions diabetic neuropathy, prefer E11.40 over the "
                "unspecified E11.9."
            ),
            "related_icd_codes": ["E11.40"],
        },
        {
            "type": "semantic",
            "action": "remove",
            "section": "MEDICAL HISTORY",
            "description": "Anxiety listed only as historical, resolved — not an active diagnosis.",
            "instruction_text": (
                "If anxiety appears only in the past medical history as resolved, "
                "do not assign F41.1; reserve it for currently active anxiety disorders."
            ),
            "related_icd_codes": ["F41.1"],
        },
    ],
    indent=2,
)
