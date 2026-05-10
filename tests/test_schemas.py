"""Tests for the prediction and instruction schemas."""

import pytest
from pydantic import ValidationError

from src.meta_verifier.schemas import Instruction, InstructionType, RichErrorInstruction
from src.prompter import ICDPrediction, ICDsModel


class TestICDPrediction:
    def test_valid(self):
        p = ICDPrediction(icd_code="E11.9", reason="diabetes")
        assert p.icd_code == "E11.9"
        assert p.reason == "diabetes"

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            ICDPrediction(icd_code="J44.1")  # missing reason


class TestICDsModel:
    def test_round_trip(self):
        m = ICDsModel(diagnoses=[ICDPrediction(icd_code="I10", reason="HT")])
        restored = ICDsModel.model_validate_json(m.model_dump_json())
        assert restored.diagnoses[0].icd_code == "I10"


class TestInstruction:
    def test_semantic_instruction(self):
        i = Instruction(
            instruction_id=1,
            type=InstructionType.SEMANTIC,
            instruction_text="prefer E11.4 if neuropathy mentioned",
            description="diabetic neuropathy without DKA",
            target_codes=["E11"],
            source_hadm_ids=["12345"],
            efficacy_score=0.0,
            semantic_embedding=[0.1, 0.2, 0.3],
        )
        assert i.fpr_at_creation is None
        assert i.fnr_at_creation is None
        assert i.target_codes == ["E11"]
        assert i.source_hadm_ids == ["12345"]

    def test_threshold_instruction(self):
        i = Instruction(
            instruction_id=2,
            type=InstructionType.FP_WARNING,
            instruction_text="caution on I10",
            description="historical FPR high for I10",
            target_codes=["I10"],
            fpr_at_creation=0.78,
            fnr_at_creation=None,
            efficacy_score=0.0,
        )
        assert i.fpr_at_creation == 0.78
        assert i.source_hadm_ids == []

    def test_round_trip(self):
        original = Instruction(
            instruction_id=42,
            type=InstructionType.CONTRASTIVE_SWAP,
            instruction_text="prefer X over Y",
            description="cue text",
            target_codes=["E11", "I10"],
            source_hadm_ids=["a", "b"],
            efficacy_score=1.5,
            semantic_embedding=[0.5, 0.6],
        )
        restored = Instruction.model_validate_json(original.model_dump_json())
        assert restored.instruction_id == 42
        assert restored.target_codes == ["E11", "I10"]
        assert restored.semantic_embedding == [0.5, 0.6]

    def test_section_field_stored_and_loaded(self):
        instr = Instruction(
            instruction_id=7,
            type=InstructionType.SEMANTIC,
            instruction_text="text",
            section="PHYSICAL EXAM",
            semantic_embedding=[0.1],
        )
        restored = Instruction.model_validate_json(instr.model_dump_json())
        assert restored.section == "PHYSICAL EXAM"

    def test_section_defaults_to_empty_string(self):
        instr = Instruction(
            instruction_id=8,
            type=InstructionType.SEMANTIC,
            instruction_text="text",
        )
        assert instr.section == ""

    def test_icd_reasoning_section_accepted(self):
        instr = Instruction(
            instruction_id=9,
            type=InstructionType.CONTRASTIVE_SWAP,
            instruction_text="prefer E11.4 over E11.9",
            section="icd_reasoning",
        )
        assert instr.section == "icd_reasoning"


class TestRichErrorInstruction:
    def test_valid_with_section(self):
        item = RichErrorInstruction(
            type="semantic",
            section="PRESENT ILLNESS",
            description="Diabetic neuropathy cue.",
            instruction_text="Prefer E11.4 if neuropathy is documented.",
            related_icd_codes=["E11"],
        )
        assert item.section == "PRESENT ILLNESS"

    def test_icd_reasoning_section_accepted(self):
        item = RichErrorInstruction(
            type="contrastive_swap",
            section="icd_reasoning",
            description="Similar codes for sepsis.",
            instruction_text="Use A41.9 not A40.9 for unspecified sepsis.",
            related_icd_codes=["A41"],
        )
        assert item.section == "icd_reasoning"

    def test_missing_section_raises(self):
        with pytest.raises(ValidationError):
            RichErrorInstruction(
                type="semantic",
                description="some description",
                instruction_text="some text",
            )
