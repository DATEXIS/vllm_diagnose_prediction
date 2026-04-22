import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from src.prompter import ICDsModel, get_schema
from src.meta_verifier.schemas import Instruction
from src.merlin2.prompts import BASE_PROMPT, THINKING_PROMPT, FINAL_PROMPT


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
        self.config = config

    def generate(
        self, admission_note: str, instructions: Optional[List[Instruction]] = None
    ) -> ICDsModel:
        if instructions:
            return self._generate_with_thinking(admission_note, instructions)
        else:
            return self._generate_simple(admission_note)

    def _generate_simple(self, admission_note: str) -> ICDsModel:
        prompt = BASE_PROMPT.format(admission_note=admission_note)
        response = self._call_vllm(prompt)
        return self._parse_response(response)

    def _generate_with_thinking(
        self, admission_note: str, instructions: List[Instruction]
    ) -> ICDsModel:
        prompt = BASE_PROMPT.format(admission_note=admission_note)
        previous_prediction = None
        thinking_blocks = []

        for i, instruction in enumerate(instructions):
            thinking_prompt = self._build_thinking_prompt(
                instructions[: i + 1], previous_prediction
            )
            full_prompt = prompt + thinking_prompt
            response = self._call_vllm(full_prompt)

            thinking_content = self._extract_thinking_block(response)
            thinking_blocks.append(thinking_content)

            try:
                previous_prediction = self._parse_response(response)
            except Exception:
                pass

        final_prompt = prompt + "\n".join(thinking_blocks) + "\n\n" + FINAL_PROMPT
        final_response = self._call_vllm(final_prompt)
        return self._parse_response(final_response)

    def _build_thinking_prompt(
        self, instructions: List[Instruction], previous_prediction: str = None
    ) -> str:
        prompt_parts = []
        for instruction in instructions:
            prev_pred_text = f"Previous prediction: {previous_prediction}\n" if previous_prediction else ""
            prompt_parts.append(
                THINKING_PROMPT.format(
                    instruction_id=instruction.id,
                    contrastive_rule=instruction.contrastive_rule,
                    previous_prediction=prev_pred_text,
                )
            )
        return "".join(prompt_parts)

    def _call_vllm(self, prompt: str) -> str:
        """Call vLLM API. Uses inference.py if config provided, otherwise mock."""
        if self.config is not None:
            return asyncio.run(self._call_vllm_async(prompt))
        else:
            # Mock response for testing/development
            sample_response = {
                "diagnoses": [
                    {
                        "icd_code": "I10",
                        "reason": "Patient presents with elevated blood pressure readings consistent with essential hypertension.",
                    },
                    {
                        "icd_code": "E11.9",
                        "reason": "Documented history of type 2 diabetes mellitus with elevated HbA1c values.",
                    },
                ]
            }
            return json.dumps(sample_response)

    async def _call_vllm_async(self, prompt: str) -> str:
        """Async call to vLLM using inference.py."""
        from src.inference import build_payload, run_inference

        schema = get_schema() if self.config.get('inference', {}).get('guided_decoding', False) else None
        
        # Build config with current generator settings
        cfg = {
            'model': {
                'name': self.model,
                'api_base': self.api_base,
            },
            'inference': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'guided_decoding': self.config.get('inference', {}).get('guided_decoding', False),
                'concurrency': 1,
            },
            'job_name': self.config.get('job_name', 'local'),
            'k8s': self.config.get('k8s', {}),
        }
        
        responses = await run_inference(cfg, [prompt], schema)
        if responses and responses[0]:
            return responses[0]
        raise RuntimeError("vLLM inference returned empty response")

    def _parse_response(self, response: str) -> ICDsModel:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError("No JSON found in response")
        json_str = json_match.group()
        data = json.loads(json_str)
        return ICDsModel(**data)

    def _extract_thinking_block(self, response: str) -> str:
        thinking_pattern = r"<thinking>(.*?)</thinking>"
        matches = re.findall(thinking_pattern, response, re.DOTALL)
        if matches:
            return "<thinking>" + matches[-1] + "</thinking>"
        return ""