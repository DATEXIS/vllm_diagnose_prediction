"""Tests for the Generator (prompt construction + response parsing).

Network is mocked at `_call_vllm_batch`, the only IO boundary.
"""

import json
from unittest.mock import patch

import pytest

from src.merlin2.generator import Generator, GenerateRequest
from src.meta_verifier.schemas import Instruction, InstructionType


def _payload(codes):
    return json.dumps({"diagnoses": [{"icd_code": c, "reason": "r"} for c in codes]})


def _mk_instruction(id_=1, text="prefer X over Y", target=("E11",)):
    return Instruction(
        instruction_id=id_,
        type=InstructionType.CONTRASTIVE_SWAP,
        instruction_text=text,
        description="cue",
        target_codes=list(target),
        source_hadm_ids=["1"],
    )


class TestPromptBuilding:
    def test_zero_shot_has_no_think_block(self):
        gen = Generator()
        prompt, think = gen._build_prompt(
            GenerateRequest(admission_note="note")
        )
        assert "<think>" not in prompt
        assert "note" in prompt
        assert think == ""

    def test_with_instructions_includes_think_block(self):
        gen = Generator()
        prompt, think = gen._build_prompt(
            GenerateRequest(
                admission_note="note",
                instruction_history=[
                    (["E11"], [_mk_instruction(text="prefer E11.4 over E11.9")]),
                ],
            )
        )
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "prefer E11.4 over E11.9" in prompt
        assert "E11" in prompt  # prediction codes shown in the [t=0] header

    def test_multiple_instructions_render_one_line_each(self):
        gen = Generator()
        prompt, _ = gen._build_prompt(
            GenerateRequest(
                admission_note="note",
                instruction_history=[
                    (["E11"], [
                        _mk_instruction(id_=1, text="rule one"),
                        _mk_instruction(id_=2, text="rule two"),
                    ]),
                ],
            )
        )
        assert "rule one" in prompt
        assert "rule two" in prompt

    def test_per_iteration_blocks_rendered(self):
        # Two history entries should produce two labelled [t=N] blocks.
        gen = Generator()
        prompt, _ = gen._build_prompt(
            GenerateRequest(
                admission_note="note",
                instruction_history=[
                    (["I10"], [_mk_instruction(id_=1, text="rule for t0")]),
                    (["I10", "N18"], [_mk_instruction(id_=2, text="rule for t1")]),
                ],
            )
        )
        assert "[t=0]" in prompt
        assert "[t=1]" in prompt
        assert "rule for t0" in prompt
        assert "rule for t1" in prompt

    def test_empty_instructions_across_history_gives_no_think_block(self):
        # instruction_history entries with no instructions → no think block.
        gen = Generator()
        prompt, think = gen._build_prompt(
            GenerateRequest(
                admission_note="note",
                instruction_history=[(["I10"], [])],
            )
        )
        assert "<think>" not in prompt
        assert think == ""


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_returns_parsed_predictions(self):
        gen = Generator()
        with patch.object(Generator, "_call_vllm_batch") as mock_call:
            mock_call.return_value = [_payload(["I10"]), _payload(["E11"])]
            results = await gen.generate_batch(
                [
                    GenerateRequest(admission_note="a"),
                    GenerateRequest(admission_note="b"),
                ]
            )
        assert [r.prediction.diagnoses[0].icd_code for r in results] == ["I10", "E11"]
        assert [r.raw_response for r in results] == [_payload(["I10"]), _payload(["E11"])]

    @pytest.mark.asyncio
    async def test_picks_last_json_when_response_has_thinking(self):
        gen = Generator()
        first = _payload(["WRONG"])
        last = _payload(["I10"])
        response_with_thinking = (
            f"<think>I considered {first} but...</think>\nFinal: {last}"
        )
        with patch.object(Generator, "_call_vllm_batch") as mock_call:
            mock_call.return_value = [response_with_thinking]
            results = await gen.generate_batch(
                [GenerateRequest(admission_note="a")]
            )
        assert results[0].prediction.diagnoses[0].icd_code == "I10"
