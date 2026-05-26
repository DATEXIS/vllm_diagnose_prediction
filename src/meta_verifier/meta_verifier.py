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

Heavy lifting lives in sibling modules:

  * meta_verifier.parsing      — strip <think>, find balanced JSON list, validate
  * meta_verifier.aggregation  — FPR/FNR aggregation + code-stats row build-out
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from src.data.evaluate import normalize_icd
from src.inference import run_inference_with_system
from src.meta_verifier.aggregation import compute_threshold_code_stats
from src.meta_verifier.code_stats import CodeStatsIndex
from src.meta_verifier.parsing import extract_json_list, validate_rich_items
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


# --------------------------------------------------------------- MetaVerifier
class MetaVerifier:
    """Generates new Instruction records from a Loop-A run's results."""

    REQUIRED_COLUMNS = ("hadm_id", "admission_note", "discharge_note", "pred_codes", "true_codes")

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._full_config = config or {}
        self.cfg = self._build_config(self._full_config)

    @staticmethod
    def _build_config(config: Dict[str, Any]) -> MetaVerifierConfig:
        mv = config.get("meta_verifier", {})
        m2 = config.get("merlin2", {})
        return MetaVerifierConfig(
            fpr_threshold=mv.get("fpr_threshold", m2.get("fpr_threshold", 0.5)),
            fnr_threshold=mv.get("fnr_threshold", m2.get("fnr_threshold", 0.5)),
            min_support=mv.get("min_support", m2.get("min_support", 3)),
            temperature=mv.get("temperature", 0.4),
            max_tokens=mv.get("max_tokens", 2000),
        )

    # ------------------------------------------------------------ public
    async def audit(
        self,
        df: pd.DataFrame,
        starting_instruction_id: int = 1,
    ) -> AuditResult:
        """Audit a results dataframe and return new instructions + new code stats."""
        self._check_columns(df)

        case_instr, _ = await self._semantic_path(df, starting_instruction_id)
        new_code_stats = self._compute_code_stats(df)

        logger.info(
            f"Meta-Verifier produced {len(case_instr)} semantic instructions + "
            f"{len(new_code_stats)} new code-stat rows"
        )
        return AuditResult(instructions=case_instr, new_code_stats=new_code_stats)

    def _check_columns(self, df: pd.DataFrame) -> None:
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"Meta-Verifier requires column '{col}' on the results df")

    # ------------------------------------------------------------ path 1: semantic
    async def _semantic_path(
        self, df: pd.DataFrame, next_id: int
    ) -> Tuple[List[Instruction], int]:
        responses = await self._run_audit_inference(df)
        per_case_items = self._parse_responses(df, responses)
        return self._build_instructions(per_case_items, next_id)

    async def _run_audit_inference(self, df: pd.DataFrame) -> List[Optional[str]]:
        prompts = [self._build_audit_prompt(row) for _, row in df.iterrows()]
        return await run_inference_with_system(
            self._full_config,
            prompts,
            system_prompt="You are a Senior Medical Coding Auditor.",
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

    @staticmethod
    def _parse_responses(
        df: pd.DataFrame, responses: List[Optional[str]],
    ) -> List[Tuple[str, List[RichErrorInstruction]]]:
        per_case: List[Tuple[str, List[RichErrorInstruction]]] = []
        for (_, row), resp in zip(df.iterrows(), responses):
            if resp is None:
                logger.warning(f"Meta-Verifier got None response for hadm_id={row['hadm_id']}")
                continue
            try:
                raw_items = extract_json_list(resp)
                items = validate_rich_items(raw_items)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Meta-Verifier parse failure for hadm_id={row['hadm_id']}: {e}"
                )
                continue
            per_case.append((str(row["hadm_id"]), items))
        return per_case

    @staticmethod
    def _build_instructions(
        per_case_items: List[Tuple[str, List[RichErrorInstruction]]],
        next_id: int,
    ) -> Tuple[List[Instruction], int]:
        descriptions, origins = _flatten_descriptions(per_case_items)
        embeddings = encode_texts(descriptions) if descriptions else []

        instructions: List[Instruction] = []
        for (case_idx, item_idx), embedding in zip(origins, embeddings):
            hadm_id, items = per_case_items[case_idx]
            item = items[item_idx]
            instructions.append(_rich_item_to_instruction(item, hadm_id, next_id, embedding))
            next_id += 1
        return instructions, next_id

    # ------------------------------------------------------------ path 2: thresholds
    def _compute_code_stats(self, df: pd.DataFrame) -> CodeStatsIndex:
        return compute_threshold_code_stats(
            df,
            fpr_threshold=self.cfg.fpr_threshold,
            fnr_threshold=self.cfg.fnr_threshold,
            min_support=self.cfg.min_support,
        )

    # ------------------------------------------------------------ prompt
    @staticmethod
    def _build_audit_prompt(row: pd.Series) -> str:
        # Pass full ICD codes so the Meta-Verifier sees full clinical specificity.
        # Evaluation truncates to 3 digits, but the audit prompt should not hide
        # sub-code information.
        return load_prompt("meta_verifier").format(
            admission_note=row["admission_note"],
            discharge_note=row.get("discharge_note", "") or "",
            predicted_codes=", ".join(row["pred_codes"]),
            ground_truth_codes=", ".join(row["true_codes"]),
            hadm_id=row["hadm_id"],
            json_example=META_VERIFIER_JSON_EXAMPLE,
        )


# --------------------------------------------------------------- helpers

def _flatten_descriptions(
    per_case_items: List[Tuple[str, List[RichErrorInstruction]]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    descs: List[str] = []
    origins: List[Tuple[int, int]] = []
    for ci, (_, items) in enumerate(per_case_items):
        for ii, item in enumerate(items):
            descs.append(item.description or item.instruction_text or "")
            origins.append((ci, ii))
    return descs, origins


def _rich_item_to_instruction(
    item: RichErrorInstruction,
    hadm_id: str,
    next_id: int,
    embedding,
) -> Instruction:
    three_digit = sorted({normalize_icd(c) for c in item.related_icd_codes if normalize_icd(c)})
    # Path 1 only emits semantic / contrastive types; aggregate threshold
    # warnings live in code_stats, not in the instruction store. Coerce
    # anything else to semantic.
    inst_type = item.type or InstructionType.SEMANTIC
    if inst_type not in (InstructionType.SEMANTIC, InstructionType.CONTRASTIVE_SWAP):
        inst_type = InstructionType.SEMANTIC
    return Instruction(
        instruction_id=next_id,
        type=inst_type,
        action=item.action,
        section=item.section or "",
        instruction_text=item.instruction_text,
        description=item.description,
        target_codes=three_digit,
        source_hadm_ids=[hadm_id],
        fpr_at_creation=None,
        fnr_at_creation=None,
        efficacy_score=0.0,
        semantic_embedding=embedding,
    )
