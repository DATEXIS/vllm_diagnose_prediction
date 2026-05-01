"""MERLIN 2 Generator.

Owns prompt construction (system + user + optional pre-filled <think>
block) and response parsing. The Generator does NOT itself orchestrate
iterations — that lives in Pipeline. One `generate_batch` call = one
iteration across all live cases.

There is no mock branch in the production code path. Tests should patch
`_call_vllm_batch` (the only network boundary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.meta_verifier.schemas import Instruction
from src.prompter import ICDsModel, get_schema
from src.utils.parsing_utils import JSONExtractionError, parse_prediction
from src.utils.prompt_loader import GENERATOR_JSON_EXAMPLE, load_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    """One Generator job for one case at one iteration."""
    admission_note: str
    instructions: List[Instruction]                # may be empty (zero-shot)
    previous_predicted_codes: List[str]            # used to fill the think block; may be empty


@dataclass
class GenerateResult:
    prediction: ICDsModel
    raw_response: str
    prompt: str
    think_block: str = ""
    parse_failed: bool = False  # True if the model returned no usable JSON


class Generator:
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = config or {}

    # ----------------------------------------------------------------- API
    async def generate_batch(self, requests: List[GenerateRequest]) -> List[GenerateResult]:
        """Run one iteration across the batch concurrently.

        A malformed LLM response (no parseable JSON) does NOT crash the
        batch: we log a warning, return an empty `ICDsModel` for that
        case, and set `parse_failed=True`. The Pipeline halts the case
        with `HaltReason.PARSE_FAILURE` on its next halt-check.
        """
        prompts_and_thinks = [self._build_prompt(req) for req in requests]
        prompts = [p for p, _ in prompts_and_thinks]
        responses = await self._call_vllm_batch(prompts)
        results: List[GenerateResult] = []
        for (prompt, think_block), raw in zip(prompts_and_thinks, responses):
            try:
                prediction = parse_prediction(raw)
                parse_failed = False
            except JSONExtractionError as e:
                logger.warning(
                    f"Parse failure: {e}. Returning empty prediction for this case."
                )
                prediction = ICDsModel(diagnoses=[])
                parse_failed = True
            results.append(
                GenerateResult(
                    prediction=prediction,
                    raw_response=raw,
                    prompt=prompt,
                    think_block=think_block,
                    parse_failed=parse_failed,
                )
            )
        return results

    # --------------------------------------------------------- prompt build
    def _build_prompt(self, req: GenerateRequest) -> Tuple[str, str]:
        system = load_prompt("generator_system").format(json_example=GENERATOR_JSON_EXAMPLE)
        user = load_prompt("generator_user").format(admission_note=req.admission_note)
        think_block = self._build_think_block(req.instructions, req.previous_predicted_codes)
        # Order: system | user | optional pre-filled think block.
        # The think block is presented as "assistant scratch reasoning"
        # appended after the user turn so the Generator continues from it.
        if think_block:
            return f"{system}\n\n{user}\n{think_block}\n", think_block
        return f"{system}\n\n{user}\n", ""

    def _build_think_block(
        self,
        instructions: List[Instruction],
        previous_predicted_codes: List[str],
    ) -> str:
        if not instructions:
            return ""
        line_template = load_prompt("think_instruction_line")
        lines = "".join(
            line_template.format(
                instruction_id=instr.instruction_id,
                type=instr.type,
                target_codes=",".join(instr.target_codes),
                instruction_text=instr.instruction_text,
            )
            for instr in instructions
        ).rstrip()
        block = load_prompt("think_block").format(
            previous_codes=", ".join(previous_predicted_codes) if previous_predicted_codes else "(none)",
            instruction_lines=lines,
        )
        return block

    # ---------------------------------------------------------- vLLM bridge
    async def _call_vllm_batch(self, prompts: List[str]) -> List[str]:
        """Send `prompts` concurrently to the vLLM server."""
        from src.inference import run_inference

        guided = self.config.get("inference", {}).get("guided_decoding", False)
        schema = get_schema() if guided else None

        cfg = {
            "model": {"name": self.model, "api_base": self.api_base},
            "inference": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "guided_decoding": guided,
                "concurrency": self.config.get("inference", {}).get("concurrency", 64),
            },
            "job_name": self.config.get("job_name", "local"),
            "k8s": self.config.get("k8s", {}),
        }
        responses = await run_inference(cfg, prompts, schema)
        if len(responses) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(responses)} responses for {len(prompts)} prompts"
            )
        return responses
