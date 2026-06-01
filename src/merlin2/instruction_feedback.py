"""Render retrieved instructions into the Generator's <coding_review> block.

Loop A pre-fills a structured per-iteration block in the user prompt with
the instructions the Retriever surfaced. FP warnings are collapsed into one
prose line so they read like genuine self-reflection; semantic /
contrastive-swap instructions appear as individual bullets.

Kept in its own module so the Generator only concerns itself with the
network call and response parsing.
"""

from __future__ import annotations

from typing import List, Tuple

from src.meta_verifier.schemas import Instruction, InstructionType
from src.utils.prompt_loader import load_prompt


InstructionHistory = List[Tuple[List[str], List[Instruction]]]


def build_think_block(instruction_history: InstructionHistory) -> str:
    """Build the <coding_review> block prefixed to the user turn.

    Returns "" when no instructions have ever fired (zero-shot, or every
    retrieval has been empty). In that case the Generator runs without
    any pre-filled review block.

    Special case: if the most recent prediction was completely empty (0 codes),
    a zero-prediction escalation block is always emitted even when the retriever
    returned nothing — the model needs a hard redirect, not silence.
    """
    last_codes = instruction_history[-1][0] if instruction_history else []
    zero_prediction = len(last_codes) == 0 and len(instruction_history) > 0

    if not _has_any_instruction(instruction_history) and not zero_prediction:
        return ""

    blocks = [
        _render_iteration_block(t, codes, instructions)
        for t, (codes, instructions) in enumerate(instruction_history)
    ]
    content = "\n".join(blocks).rstrip()
    return load_prompt("think_block").format(content=content)


def _has_any_instruction(history: InstructionHistory) -> bool:
    return any(instructions for _, instructions in history)


def _render_iteration_block(
    iteration: int,
    predicted_codes: List[str],
    instructions: List[Instruction],
) -> str:
    iter_block_template = load_prompt("think_iteration_block")
    header = _iteration_header(iteration, predicted_codes)
    under_predicting = len(predicted_codes) < 3
    lines = _render_instruction_lines(instructions, suppress_fp=under_predicting)
    if under_predicting:
        lines = _render_low_count_warning(len(predicted_codes)) + ("\n" + lines if lines else "")
    return iter_block_template.format(
        iteration_header=header,
        instruction_lines=lines,
    )


def _iteration_header(iteration: int, predicted_codes: List[str]) -> str:
    codes_str = ", ".join(predicted_codes) if predicted_codes else "(none)"
    if iteration == 0:
        return (
            f"My initial prediction was: {codes_str}. "
            f"Instruction checklist — I must address each item below before changing any code:"
        )
    return (
        f"After applying the previous checklist, my updated prediction is: {codes_str}. "
        f"Additional instruction checklist — I must address each item below:"
    )


def _render_instruction_lines(
    instructions: List[Instruction],
    suppress_fp: bool = False,
) -> str:
    """Group instructions by type and render each group as one prompt block.

    When `suppress_fp` is True (current prediction is below min_prediction_size),
    FP warnings are omitted entirely. Telling an already under-predicting model
    to *remove* codes makes the under-prediction worse; the under-prediction
    warning rendered by the caller takes priority.
    """
    fp = [] if suppress_fp else [i for i in instructions if i.type == InstructionType.FP_WARNING]
    semantic = [i for i in instructions if i.type != InstructionType.FP_WARNING]

    parts: List[str] = []
    if fp:
        parts.append(_render_fp_line(fp))
    parts.append("GENERAL INSTRUCTIONS")
    parts.extend(_render_semantic_line(i, n) for n, i in enumerate(semantic, 1))
    return "\n".join(parts).rstrip()


def _render_low_count_warning(n_codes: int) -> str:
    if n_codes == 0:
        return (
            "- CRITICAL: My previous response contained zero diagnoses. "
            "This is always wrong — every hospital admission has at least one "
            "billable diagnosis. I must produce a non-empty list. "
            "I will read the note again from scratch, section by section, and "
            "assign a code for every condition, comorbidity, and finding I encounter. "
            "Outputting an empty diagnoses array again is not acceptable."
        )
    code_word = "code" if n_codes == 1 else "codes"
    return (
        f"- UNDER-PREDICTION WARNING: I predicted only {n_codes} {code_word}. "
        f"This is almost certainly too few. Real hospital admissions routinely "
        f"carry 8–15 ICD codes covering the principal diagnosis, comorbidities, "
        f"complications, chronic conditions, and relevant findings. "
        f"I must go back through the entire note and identify every documented "
        f"condition — I should not default to a minimal code set."
    )


def _render_fp_line(fp_warnings: List[Instruction]) -> str:
    code_stmts = "\n".join(
        f"{n}. [{i.target_codes[0]}] {i.instruction_text}"
        for n, i in enumerate(fp_warnings, 1)
    )
    return f"FALSE POSITIVE CODES:\n{code_stmts}"


def _render_semantic_line(instr: Instruction, n: int = 0) -> str:
    line_template = load_prompt("think_instruction_line")
    return line_template.format(
        n=n,
        instruction_id=instr.instruction_id,
        type=instr.type,
        target_codes=",".join(instr.target_codes),
        instruction_text=instr.instruction_text,
    )
