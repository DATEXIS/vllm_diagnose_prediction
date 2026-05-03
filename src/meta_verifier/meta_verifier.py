"""MERLIN 2 Meta-Verifier (Loop B).

Runs only on training data after a Loop-A pass. Two error-discovery paths:

  1. Case-level analysis (semantic instructions). For each closed case,
     send the LLM an audit prompt with admission_note + predicted_codes
     + ground_truth_codes + discharge_note + hadm_id. Parse a JSON list
     of RichErrorInstruction. Embed each `description` with PubMedBERT.
     Output: new `Instruction` rows for the instructions parquet store.

  2. Aggregate metrics (threshold stats). Scan per-3-digit-code FPR/FNR
     over the audited cases. For codes that exceed the threshold AND
     meet `min_support`, emit one row to the per-code stats table
     (`code_stats.parquet`) — NOT an Instruction row. The retriever
     synthesises the warning text at runtime from these stats.
     Snapshot fpr / fnr values are frozen at creation: codes already in
     the stats table are not touched on subsequent Loop-B passes (per
     MERLIN2_SPEC §3).

`audit(...)` returns an `AuditResult` with the new Instructions plus the
new code-stats rows the caller should merge into `code_stats.parquet`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from src.data.evaluate import normalize_icd
from src.inference import run_inference_with_system
from src.meta_verifier.code_stats import CodeStat, CodeStatsIndex
from src.meta_verifier.schemas import (
    Instruction,
    InstructionType,
    RichErrorInstruction,
)
from src.utils.embeddings import encode_texts
from src.utils.prompt_loader import META_VERIFIER_JSON_EXAMPLE, load_prompt

logger = logging.getLogger(__name__)


@dataclass
class MetaVerifierConfig:
    fpr_threshold: float = 0.5
    fnr_threshold: float = 0.5
    min_support: int = 3
    temperature: float = 0.4
    max_tokens: int = 8192


@dataclass
class AuditResult:
    """Two-part output of a Loop-B pass.

    `instructions` is appended to the instructions parquet store.
    `new_code_stats` is merged (additively, frozen-rate) into
    `code_stats.parquet`.
    """
    instructions: List[Instruction] = field(default_factory=list)
    new_code_stats: CodeStatsIndex = field(default_factory=dict)


# --------------------------------------------------------------- helpers
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> sections emitted by reasoning models."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_json_list(text: str) -> List[dict]:
    """Find the last balanced `[...]` and json.loads it.

    Reasoning models (e.g. Qwen3) prepend a <think>…</think> block before
    their JSON output.  We strip those first so that:
      - A complete think block followed by valid JSON → JSON is found.
      - A truncated think block (max_tokens hit mid-think, no JSON) →
        stripped text is empty → clear ValueError with the original preview.
    """
    if not text:
        raise ValueError("Empty Meta-Verifier response.")

    # Search in think-stripped text; fall back to raw if stripping left nothing
    search_text = _strip_think_blocks(text) or text

    # Find last `]`
    end = search_text.rfind("]")
    if end < 0:
        raise ValueError(f"No ']' in Meta-Verifier response. Preview: {text[:300]!r}")
    # Walk backwards to find the matching `[`
    depth = 0
    in_str = False
    escape = False
    start = -1
    for i in range(end, -1, -1):
        ch = search_text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    start = i
                    break
    if start < 0:
        raise ValueError(f"No balanced '[...]' in Meta-Verifier response. Preview: {text[:300]!r}")
    parsed = json.loads(search_text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed).__name__}")
    return parsed


def _validate_rich_items(items: List[dict]) -> List[RichErrorInstruction]:
    """Validate each item individually; skip non-dicts and invalid items.

    The LLM occasionally emits bare ICD strings (e.g. "S72") alongside
    proper dicts.  Skipping bad entries per-item lets us salvage the rest
    of the case rather than discarding it entirely.
    """
    out: List[RichErrorInstruction] = []
    for raw in items:
        if not isinstance(raw, dict):
            logger.debug(
                "Meta-Verifier: skipping non-dict item in JSON list: %r", raw
            )
            continue
        try:
            out.append(RichErrorInstruction.model_validate(raw))
        except ValidationError as e:
            logger.debug("Meta-Verifier: skipping invalid item %r: %s", raw, e)
    return out


def _three_digit(code: str) -> str:
    return normalize_icd(code)


def _aggregate_fpr_fnr(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Compute per-3-digit FPR, FNR, prediction-support, ground-truth-support."""
    fp = {}
    fn = {}
    pred_n = {}
    true_n = {}
    for _, row in df.iterrows():
        pred = {_three_digit(c) for c in row["pred_codes"] if _three_digit(c)}
        true = {_three_digit(c) for c in row["true_codes"] if _three_digit(c)}
        for c in pred - true:
            fp[c] = fp.get(c, 0) + 1
        for c in true - pred:
            fn[c] = fn.get(c, 0) + 1
        for c in pred:
            pred_n[c] = pred_n.get(c, 0) + 1
        for c in true:
            true_n[c] = true_n.get(c, 0) + 1
    fpr = {c: fp.get(c, 0) / pred_n[c] for c in pred_n}
    fnr = {c: fn.get(c, 0) / true_n[c] for c in true_n}
    return fpr, fnr, pred_n, true_n


# --------------------------------------------------------------- MetaVerifier
class MetaVerifier:
    """Generates new Instruction records from a Loop-A run's results."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config or {}).get("meta_verifier", {})
        merlin2_cfg = (config or {}).get("merlin2", {})
        self.cfg = MetaVerifierConfig(
            fpr_threshold=cfg.get("fpr_threshold", merlin2_cfg.get("fpr_threshold", 0.5)),
            fnr_threshold=cfg.get("fnr_threshold", merlin2_cfg.get("fnr_threshold", 0.5)),
            min_support=cfg.get("min_support", merlin2_cfg.get("min_support", 3)),
            temperature=cfg.get("temperature", 0.4),
            max_tokens=cfg.get("max_tokens", 2000),
        )
        self._full_config = config or {}

    # ------------------------------------------------------------ public
    async def audit(
        self,
        df: pd.DataFrame,
        starting_instruction_id: int = 1,
    ) -> AuditResult:
        """Audit a results dataframe.

        Path 1 produces new semantic/contrastive `Instruction` rows.
        Path 2 produces per-code `CodeStat` rows (no Instructions); the
        retriever synthesises FP/FN warning text from these at runtime.

        `df` must have columns: hadm_id, admission_note, discharge_note,
        pred_codes (list[str]), true_codes (list[str]).
        """
        for col in ("hadm_id", "admission_note", "discharge_note", "pred_codes", "true_codes"):
            if col not in df.columns:
                raise KeyError(f"Meta-Verifier requires column '{col}' on the results df")

        next_id = starting_instruction_id

        # Path 1: case-level semantic instructions
        case_instr, next_id = await self._semantic_path(df, next_id)

        # Path 2: aggregate-metric code-stats
        new_code_stats = self._compute_code_stats(df)

        logger.info(
            f"Meta-Verifier produced {len(case_instr)} semantic instructions + "
            f"{len(new_code_stats)} new code-stat rows"
        )
        return AuditResult(instructions=case_instr, new_code_stats=new_code_stats)

    # ------------------------------------------------------------ path 1
    async def _semantic_path(
        self, df: pd.DataFrame, next_id: int
    ) -> Tuple[List[Instruction], int]:
        prompts = [self._build_audit_prompt(row) for _, row in df.iterrows()]
        responses = await run_inference_with_system(
            self._full_config,
            prompts,
            system_prompt="You are a Senior Medical Coding Auditor.",
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        # parse + validate; skip cases the LLM produced no usable JSON for
        per_case_items: List[Tuple[str, List[RichErrorInstruction]]] = []
        for (_, row), resp in zip(df.iterrows(), responses):
            if resp is None:
                logger.warning(f"Meta-Verifier got None response for hadm_id={row['hadm_id']}")
                continue
            try:
                raw_items = _extract_json_list(resp)
                items = _validate_rich_items(raw_items)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Meta-Verifier parse failure for hadm_id={row['hadm_id']}: {e}"
                )
                continue
            per_case_items.append((str(row["hadm_id"]), items))

        # batch-embed the descriptions for efficiency
        descs: List[str] = []
        descs_origin: List[Tuple[int, int]] = []  # (case_idx, item_idx)
        for ci, (_, items) in enumerate(per_case_items):
            for ii, item in enumerate(items):
                descs.append(item.description or item.instruction_text or "")
                descs_origin.append((ci, ii))
        embeddings = encode_texts(descs) if descs else []

        instructions: List[Instruction] = []
        for (case_idx, item_idx), embedding in zip(descs_origin, embeddings):
            hadm_id, items = per_case_items[case_idx]
            item = items[item_idx]
            three_digit = sorted({_three_digit(c) for c in item.related_icd_codes if _three_digit(c)})
            # Path 1 only emits semantic/contrastive types; aggregate
            # threshold warnings live in code_stats, not in the
            # instruction store. Coerce anything else to semantic.
            inst_type = item.type or InstructionType.SEMANTIC
            if inst_type not in (
                InstructionType.SEMANTIC,
                InstructionType.CONTRASTIVE_SWAP,
            ):
                inst_type = InstructionType.SEMANTIC
            instructions.append(
                Instruction(
                    instruction_id=next_id,
                    type=inst_type,
                    instruction_text=item.instruction_text,
                    description=item.description,
                    target_codes=three_digit,
                    source_hadm_ids=[hadm_id],
                    fpr_at_creation=None,
                    fnr_at_creation=None,
                    efficacy_score=0.0,
                    semantic_embedding=embedding,
                )
            )
            next_id += 1
        return instructions, next_id

    # ------------------------------------------------------------ path 2
    def _compute_code_stats(self, df: pd.DataFrame) -> CodeStatsIndex:
        """Compute per-code threshold-eligible stats from this audit batch.

        Returns a dict[code -> CodeStat] containing only codes that
        - meet `min_support` AND
        - exceed `fpr_threshold` (then `fpr` is set, `fnr` is None) OR
        - exceed `fnr_threshold` (then `fnr` is set, `fpr` is None).

        A code can only be on one side at a time (FP or FN); if it
        crosses both, FP wins (it appears in the predictions either way,
        so the retriever needs the FP gate active).

        Whether to merge / overwrite vs. existing code_stats is the
        caller's responsibility; this function only describes "what does
        this batch suggest as a new threshold rule".
        """
        fpr, fnr, pred_n, true_n = _aggregate_fpr_fnr(df)
        out: CodeStatsIndex = {}

        for code, rate in fpr.items():
            if pred_n.get(code, 0) < self.cfg.min_support:
                continue
            if rate < self.cfg.fpr_threshold:
                continue
            out[code] = CodeStat(
                code=code,
                fpr=float(rate),
                fnr=None,
                support_pred=int(pred_n.get(code, 0)),
                support_true=int(true_n.get(code, 0)),
            )

        for code, rate in fnr.items():
            if true_n.get(code, 0) < self.cfg.min_support:
                continue
            if rate < self.cfg.fnr_threshold:
                continue
            if code in out:
                # FP gate already active for this code; FN flag is redundant
                # because every retrieval that fires the FN candidate set
                # already fires the FP one (the code is being predicted).
                continue
            out[code] = CodeStat(
                code=code,
                fpr=None,
                fnr=float(rate),
                support_pred=int(pred_n.get(code, 0)),
                support_true=int(true_n.get(code, 0)),
            )

        return out

    # ------------------------------------------------------------ prompt
    def _build_audit_prompt(self, row: pd.Series) -> str:
        return load_prompt("meta_verifier").format(
            admission_note=row["admission_note"],
            discharge_note=row.get("discharge_note", "") or "",
            predicted_codes=", ".join(_three_digit(c) for c in row["pred_codes"]),
            ground_truth_codes=", ".join(_three_digit(c) for c in row["true_codes"]),
            hadm_id=row["hadm_id"],
            json_example=META_VERIFIER_JSON_EXAMPLE,
        )
