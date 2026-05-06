"""MERLIN 2 Generator.

Owns prompt construction (system + user + optional pre-filled <coding_review>
block) and response parsing. The Generator does NOT itself orchestrate
iterations — that lives in Pipeline. One `generate_batch` call = one
iteration across all live cases.

Prompt roles are sent as proper chat messages:
  - system: task framing + output schema example
  - user  : admission note, with the <coding_review> block appended inline
            when retrieved instructions exist (t >= 1)

There is no mock branch in the production code path. Tests should patch
`_call_vllm_batch` (the only network boundary).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.meta_verifier.schemas import Instruction
from src.prompter import ICDsModel, get_schema
from src.utils.parsing_utils import JSONExtractionError, parse_prediction
from src.utils.prompt_loader import GENERATOR_JSON_EXAMPLE, load_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    """One Generator job for one case at one iteration.

    `instruction_history` encodes all refinement context accumulated so far.
    Each entry is `(predicted_codes, instructions)` where:
      - `predicted_codes`: 3-digit ICD codes predicted at the start of that round
      - `instructions`:    instructions retrieved in response to that prediction

    An empty list means zero-shot (t=0): no think block is emitted.
    At t=1 there is one entry (zero-shot codes + new instructions).
    At t=2 there are two entries, and so on.
    """
    admission_note: str
    instruction_history: List[Tuple[List[str], List[Instruction]]] = field(default_factory=list)


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
        messages_and_thinks = [self._build_prompt(req) for req in requests]
        messages_list = [msgs for msgs, _ in messages_and_thinks]
        responses = await self._call_vllm_batch(messages_list)
        results: List[GenerateResult] = []
        for (messages, think_block), raw in zip(messages_and_thinks, responses):
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
                    # Serialise messages as JSON for logging / debugging.
                    prompt=json.dumps(messages, ensure_ascii=False),
                    think_block=think_block,
                    parse_failed=parse_failed,
                )
            )
        return results

    # --------------------------------------------------------- prompt build
    def _build_prompt(
        self, req: GenerateRequest
    ) -> Tuple[List[Dict[str, str]], str]:
        """Return (messages, think_block_str).

        messages is a role-separated list ready for the chat/completions API:
          - system: task framing
          - user  : admission note, with the <coding_review> block appended
                    inline when instructions are available (t >= 1)

        The <coding_review> block sits in the user turn rather than as an
        assistant prefill. A closed assistant prefill causes models to emit EOS
        or enter repetition loops because the turn looks complete; guided
        decoding then fights with the </coding_review> closure. Keeping it in
        the user turn lets the model generate a clean JSON response with full
        context visible.
        """
        system = load_prompt("generator_system").format(json_example=GENERATOR_JSON_EXAMPLE)
        user = load_prompt("generator_user").format(admission_note=req.admission_note)
        think_block = self._build_think_block(req.instruction_history)

        user_content = f"{user}\n{think_block}" if think_block else user

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ]

        return messages, think_block

    def _build_think_block(
        self,
        instruction_history: List[Tuple[List[str], List[Instruction]]],
    ) -> str:
        """Build a structured, per-iteration <coding_review> block.

        Returns "" if there are no instructions anywhere (zero-shot or all
        retrieval misses) — in that case the Generator runs without a think
        block prefix.

        Otherwise builds one block per history entry with a natural-language
        header that reads like genuine model reasoning:

            My initial prediction was I69, N31, N39, so I considered the following:
            - Code N85 is missed in 100% of cases where it should have been assigned ...

            Based on this information I predicted I69, N31, N39, N85, so I considered the following:
            - Code Z16 is missed in 100% of cases ...
        """
        # Skip if nothing useful to show
        if not any(instrs for _, instrs in instruction_history):
            return ""

        line_template = load_prompt("think_instruction_line")
        iter_block_template = load_prompt("think_iteration_block")

        blocks: List[str] = []
        for t, (codes, instructions) in enumerate(instruction_history):
            codes_str = ", ".join(codes) if codes else "(none)"
            if t == 0:
                iteration_header = (
                    f"My initial prediction was {codes_str}, so I considered the following:"
                )
            else:
                iteration_header = (
                    f"Based on this information I predicted {codes_str}, "
                    f"so I considered the following:"
                )
            lines = "".join(
                line_template.format(
                    instruction_id=instr.instruction_id,
                    type=instr.type,
                    target_codes=",".join(instr.target_codes),
                    instruction_text=instr.instruction_text,
                )
                for instr in instructions
            ).rstrip()
            blocks.append(
                iter_block_template.format(
                    iteration_header=iteration_header,
                    instruction_lines=lines,
                )
            )

        content = "\n\n".join(blocks).rstrip()
        return load_prompt("think_block").format(content=content)

    # ---------------------------------------------------------- vLLM bridge
    async def _call_vllm_batch(
        self, messages_list: List[List[Dict[str, str]]]
    ) -> List[str]:
        """Send `messages_list` concurrently to the vLLM server."""
        from src.inference import run_inference_messages

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
        responses = await run_inference_messages(cfg, messages_list, schema)
        if len(responses) != len(messages_list):
            raise RuntimeError(
                f"vLLM returned {len(responses)} responses for {len(messages_list)} prompts"
            )
        return responses
