"""Render retrieved instructions into the Generator's <coding_review> block.

Loop A pre-fills a structured per-iteration block in the user prompt with
the instructions the Retriever surfaced. Threshold warnings (FP/FN) are
collapsed into one prose line each so they read like genuine self-reflection;
semantic / contrastive-swap instructions appear as individual bullets.

Kept in its own module so the Generator only concerns itself with the
network call and response parsing.
"""

from __future__ import annotations

from typing import List, Tuple

from src.meta_verifier.schemas import Instruction, InstructionType
from src.utils.prompt_loader import load_prompt


InstructionHistory = List[Tuple[List[str], List[Instruction]]]


def build_think_block(
    instruction_history: InstructionHistory,
    max_history_depth: int = 2,
) -> str:
    """Build the <coding_review> block prefixed to the user turn.

    Returns "" when no instructions have ever fired (zero-shot, or every
    retrieval has been empty). In that case the Generator runs without
    any pre-filled review block.

    Special case: if the most recent prediction was completely empty (0 codes),
    a zero-prediction escalation block is always emitted even when the retriever
    returned nothing — the model needs a hard redirect, not silence.

    `max_history_depth` caps how many of the most recent iteration blocks are
    rendered inside the <coding_review>. Older blocks are dropped — their
    instructions are already reflected in the current prediction. This keeps
    the block from growing unboundedly across many iterations and prevents
    context-length parse failures at late iterations (T=4+).
    Default is 2: the model always sees the baseline prediction and the most
    recent round of instructions, plus one prior round for context.
    Set to None to disable the cap (original behaviour, not recommended).
    """
    last_codes = instruction_history[-1][0] if instruction_history else []
    zero_prediction = len(last_codes) == 0 and len(instruction_history) > 0

    if not _has_any_instruction(instruction_history) and not zero_prediction:
        return ""

    # Apply depth cap: keep the tail of the history, but always include t=0
    # so the model can see where it started even when history is trimmed.
    if max_history_depth is not None and len(instruction_history) > max_history_depth:
        kept = instruction_history[-max_history_depth:]
        # Adjust iteration numbers so the header text stays accurate
        offset = len(instruction_history) - max_history_depth
    else:
        kept = instruction_history
        offset = 0

    blocks = [
        _render_iteration_block(offset + t, codes, instructions)
        for t, (codes, instructions) in enumerate(kept)
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
    fn = [i for i in instructions if i.type == InstructionType.FN_WARNING]
    semantic = [
        i for i in instructions
        if i.type not in (InstructionType.FP_WARNING, InstructionType.FN_WARNING)
    ]

    parts: List[str] = []
    if fp:
        parts.append(_render_fp_line(fp))
    if fn:
        parts.append(_render_fn_line(fn))
    parts.append("GENERAL INSTRUCTIONS:")
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
    code_stmts = "\n  ".join(f"* {i.instruction_text}" for i in fp_warnings)
    return (
        f"FALSE POSITIVE CODES:\nThe following predicted codes have been verified as false positives "
        f"in the vast majority of similar cases. "
        f"I most likely should remove each of them. "
        f"I should really only keep it if I can find very strong evidence:\n{code_stmts}"
    )


def _render_fn_line(fn_warnings: List[Instruction]) -> str:
    code_stmts = "; ".join(i.instruction_text for i in fn_warnings)
    return (
        f"FALSE NEGATIVE CODES:\nSome codes are frequently missed in cases like this: "
        f"{code_stmts}. "
        f"I should carefully check whether the note supports any of these — "
        f"even an indirect clinical cue is sufficient to add the code."
    )


def _render_semantic_line(instr: Instruction, n: int = 0) -> str:
    line_template = load_prompt("think_instruction_line")
    return line_template.format(
        n=n,
        instruction_id=instr.instruction_id,
        type=instr.type,
        target_codes=",".join(instr.target_codes),
        instruction_text=instr.instruction_text,
    )
