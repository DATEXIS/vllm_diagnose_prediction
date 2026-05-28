"""MERLIN 2 Generator.

Owns prompt construction (system + user + optional pre-filled <coding_review>
block) and response parsing. The Generator does NOT itself orchestrate
iterations — that lives in Pipeline. One `generate_batch` call = one
iteration across all live cases.

Prompt roles are sent as proper chat messages:
  - system: task framing + output schema example
  - user  : admission note, with the <coding_review> block appended inline
            when retrieved instructions exist (t >= 1)

The <coding_review> block itself is rendered in `instruction_feedback.py`.

There is no mock branch in the production code path. Tests should patch
`_call_vllm_batch` (the only network boundary).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 (Tuple used in _call_vllm_batch)

from src.merlin2.instruction_feedback import build_think_block
from src.meta_verifier.schemas import Instruction
from src.prompter import ICDsModel, get_schema
from src.utils.parsing_utils import JSONExtractionError, parse_prediction
from src.utils.prompt_loader import (
    GENERAL_GUIDELINES,
    INSTRUCTION_REASONING_FIELD_DOC,
    build_json_example,
    load_prompt,
)

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
    coding_review: str = ""
    parse_failed: bool = False  # True if the model returned no usable JSON
    thinking_content: str = ""  # reasoning_content from vLLM when inference.thinking=True


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
        inf_cfg = self.config.get("inference", {})
        self.instruction_reasoning: bool = inf_cfg.get("instruction_reasoning", True)
        self.thinking: bool = inf_cfg.get("thinking", False)

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
        responses, thinking_list = await self._call_vllm_batch(messages_list)
        return [
            self._build_result(messages, coding_review, raw, thinking)
            for (messages, coding_review), raw, thinking
            in zip(messages_and_thinks, responses, thinking_list)
        ]

    @staticmethod
    def _build_result(
        messages: List[Dict[str, str]],
        coding_review: str,
        raw: str,
        thinking_content: str = "",
    ) -> GenerateResult:
        try:
            prediction = parse_prediction(raw)
            parse_failed = False
        except JSONExtractionError as e:
            logger.warning(f"Parse failure: {e}. Returning empty prediction for this case.")
            prediction = ICDsModel(diagnoses=[])
            parse_failed = True
        return GenerateResult(
            prediction=prediction,
            raw_response=raw,
            prompt=json.dumps(messages, ensure_ascii=False),
            coding_review=coding_review,
            parse_failed=parse_failed,
            thinking_content=thinking_content,
        )

    # --------------------------------------------------------- prompt build
    def _build_prompt(
        self, req: GenerateRequest
    ) -> Tuple[List[Dict[str, str]], str]:
        """Return (messages, coding_review_str).

        messages is a role-separated list ready for the chat/completions API:
          - system: task framing, with the <coding_review> block injected via
                    the {think_block} placeholder when instructions exist (t >= 1)
          - user  : admission note only

        The <coding_review> block lives in the system prompt rather than the
        user turn. This keeps the user turn a clean note-only context: when
        the model is in refinement mode it should process instructions, not
        re-analyse note text.
        """
        coding_review = build_think_block(req.instruction_history)
        system = load_prompt("generator_system").format(
            json_example=build_json_example(self.instruction_reasoning),
            general_guidelines=GENERAL_GUIDELINES,
            instruction_reasoning_field_doc=(
                INSTRUCTION_REASONING_FIELD_DOC if self.instruction_reasoning else ""
            ),
            think_block=coding_review,
        )
        user = load_prompt("generator_user").format(admission_note=req.admission_note)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return messages, coding_review

    # ---------------------------------------------------------- vLLM bridge
    async def _call_vllm_batch(
        self, messages_list: List[List[Dict[str, str]]]
    ) -> Tuple[List[str], List[str]]:
        """Send `messages_list` concurrently to the vLLM server.

        Returns (responses, thinking_list). thinking_list contains the model's
        reasoning_content per response (empty strings when thinking=False).
        """
        from src.inference import run_inference_messages, run_inference_messages_with_thinking

        cfg = self._build_inference_config()
        schema = get_schema(self.instruction_reasoning) if cfg["inference"]["guided_decoding"] else None
        if self.thinking:
            responses, thinking_list = await run_inference_messages_with_thinking(cfg, messages_list, schema)
        else:
            responses = await run_inference_messages(cfg, messages_list, schema)
            thinking_list = [""] * len(responses)
        if len(responses) != len(messages_list):
            raise RuntimeError(
                f"vLLM returned {len(responses)} responses for {len(messages_list)} prompts"
            )
        return responses, thinking_list

    def _build_inference_config(self) -> Dict[str, Any]:
        inf_cfg = self.config.get("inference", {})
        built: Dict[str, Any] = {
            "model": {"name": self.model, "api_base": self.api_base},
            "inference": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "guided_decoding": inf_cfg.get("guided_decoding", False),
                "concurrency": inf_cfg.get("concurrency", 64),
                "thinking": self.thinking,
            },
            "job_name": self.config.get("job_name", "local"),
            "k8s": self.config.get("k8s", {}),
        }
        if "reasoning_budget" in inf_cfg:
            built["inference"]["reasoning_budget"] = inf_cfg["reasoning_budget"]
        return built
