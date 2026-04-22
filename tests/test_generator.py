import json
import pytest
from unittest.mock import patch, MagicMock

from src.merlin2.generator import Generator
from src.prompter import ICDsModel
from src.meta_verifier.schemas import Instruction


class TestGeneratorInitialization:
    def test_default_initialization(self):
        gen = Generator()
        assert gen.api_base == "http://localhost:8000/v1"
        assert gen.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert gen.temperature == 0.0
        assert gen.max_tokens == 1024

    def test_custom_initialization(self):
        gen = Generator(
            api_base="http://custom:8000/v1",
            model="custom/model",
            temperature=0.5,
            max_tokens=2048
        )
        assert gen.api_base == "http://custom:8000/v1"
        assert gen.model == "custom/model"
        assert gen.temperature == 0.5
        assert gen.max_tokens == 2048


class TestGeneratorGenerate:
    @patch.object(Generator, '_call_vllm')
    def test_generate_without_instructions(self, mock_call_vllm):
        mock_call_vllm.return_value = json.dumps({
            "diagnoses": [
                {"icd_code": "I10", "reason": "Hypertension"}
            ]
        })
        gen = Generator()
        result = gen.generate("Patient has high blood pressure")
        assert isinstance(result, ICDsModel)
        assert len(result.diagnoses) == 1
        assert result.diagnoses[0].icd_code == "I10"

    @patch.object(Generator, '_call_vllm')
    def test_generate_with_instructions(self, mock_call_vllm):
        mock_call_vllm.return_value = json.dumps({
            "diagnoses": [
                {"icd_code": "I10", "reason": "Hypertension"}
            ]
        })
        gen = Generator()
        instruction = Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Check for hypertension",
            fpr=0.1,
            fnr=0.2,
            efficacy_score=0.8
        )
        result = gen.generate("Patient has high blood pressure", instructions=[instruction])
        assert isinstance(result, ICDsModel)
        assert len(result.diagnoses) == 1


class TestBuildThinkingPrompt:
    def test_thinking_block_format(self):
        gen = Generator()
        instruction = Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Consider essential hypertension",
            fpr=0.1,
            fnr=0.2,
            efficacy_score=0.8
        )
        prompt = gen._build_thinking_prompt([instruction])
        assert "<thinking>" in prompt
        assert "</thinking>" in prompt
        assert "Instruction 1: Consider essential hypertension" in prompt
        assert "Applying to prediction..." in prompt

    def test_thinking_block_with_previous_prediction(self):
        gen = Generator()
        instruction = Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Consider essential hypertension",
            fpr=0.1,
            fnr=0.2,
            efficacy_score=0.8
        )
        prompt = gen._build_thinking_prompt(
            [instruction],
            previous_prediction='{"diagnoses": [{"icd_code": "I10"}]}'
        )
        assert "Previous prediction" in prompt

    def test_multiple_instructions(self):
        gen = Generator()
        instructions = [
            Instruction(id=1, target_code="I10", contrastive_rule="Rule 1", fpr=0.1, fnr=0.2, efficacy_score=0.8),
            Instruction(id=2, target_code="E11.9", contrastive_rule="Rule 2", fpr=0.1, fnr=0.2, efficacy_score=0.8),
        ]
        prompt = gen._build_thinking_prompt(instructions)
        assert prompt.count("<thinking>") == 2
        assert "Instruction 1: Rule 1" in prompt
        assert "Instruction 2: Rule 2" in prompt


class TestParseResponse:
    def test_valid_json(self):
        gen = Generator()
        response = json.dumps({
            "diagnoses": [
                {"icd_code": "I10", "reason": "Hypertension"},
                {"icd_code": "E11.9", "reason": "Diabetes"}
            ]
        })
        result = gen._parse_response(response)
        assert isinstance(result, ICDsModel)
        assert len(result.diagnoses) == 2

    def test_invalid_json(self):
        gen = Generator()
        response = "This is not JSON"
        with pytest.raises(ValueError, match="No JSON found in response"):
            gen._parse_response(response)

    def test_malformed_json(self):
        gen = Generator()
        response = '{"diagnoses": [{"icd_code": "I10"}]'
        with pytest.raises(json.JSONDecodeError):
            gen._parse_response(response)


class TestThinkingBlocksContainContrastiveRule:
    @patch.object(Generator, '_call_vllm')
    def test_thinking_blocks_contain_contrastive_rule(self, mock_call_vllm):
        mock_call_vllm.return_value = json.dumps({
            "diagnoses": [
                {"icd_code": "I10", "reason": "Hypertension"}
            ]
        })
        gen = Generator()
        instruction = Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Check for elevated blood pressure readings",
            fpr=0.1,
            fnr=0.2,
            efficacy_score=0.8
        )
        thinking_prompt = gen._build_thinking_prompt([instruction])
        assert "Check for elevated blood pressure readings" in thinking_prompt

    @patch.object(Generator, '_call_vllm')
    def test_thinking_blocks_contain_rule_in_generate(self, mock_call_vllm):
        mock_call_vllm.return_value = json.dumps({
            "diagnoses": [
                {"icd_code": "I10", "reason": "Hypertension"}
            ]
        })
        gen = Generator()
        instruction = Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Verify hypertension diagnosis",
            fpr=0.1,
            fnr=0.2,
            efficacy_score=0.8
        )
        gen.generate("Patient has high blood pressure", instructions=[instruction])
        call_args = mock_call_vllm.call_args_list[0][0][0]
        assert "Verify hypertension diagnosis" in call_args