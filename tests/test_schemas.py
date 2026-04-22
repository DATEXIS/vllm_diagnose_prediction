import pytest
from pydantic import ValidationError

from src.prompter import ICDPrediction, ICDsModel
from src.meta_verifier.schemas import Instruction


class TestICDPrediction:
    def test_valid_icd_prediction(self):
        pred = ICDPrediction(icd_code="E11.9", reason="Patient has diabetes")
        assert pred.icd_code == "E11.9"
        assert pred.reason == "Patient has diabetes"
        assert pred.quote is None

    def test_icd_prediction_with_optional_quote(self):
        pred = ICDPrediction(
            icd_code="I10",
            reason="Patient has hypertension",
            quote="Blood pressure 180/110"
        )
        assert pred.icd_code == "I10"
        assert pred.reason == "Patient has hypertension"
        assert pred.quote == "Blood pressure 180/110"

    def test_icd_prediction_missing_required_field(self):
        with pytest.raises(ValidationError):
            ICDPrediction(icd_code="J44.1")

    def test_icd_prediction_empty_reason_allowed(self):
        pred = ICDPrediction(icd_code="E11.9", reason="")
        assert pred.reason == ""


class TestICDsModel:
    def test_icds_model_with_list_of_diagnoses(self):
        model = ICDsModel(
            diagnoses=[
                ICDPrediction(icd_code="E11.9", reason="Type 2 diabetes"),
                ICDPrediction(icd_code="I10", reason="Essential hypertension"),
                ICDPrediction(icd_code="J44.1", reason="COPD with acute exacerbation"),
            ]
        )
        assert len(model.diagnoses) == 3
        assert model.diagnoses[0].icd_code == "E11.9"
        assert model.diagnoses[1].icd_code == "I10"
        assert model.diagnoses[2].icd_code == "J44.1"

    def test_icds_model_empty_diagnoses(self):
        model = ICDsModel(diagnoses=[])
        assert len(model.diagnoses) == 0


class TestInstruction:
    def test_instruction_with_all_fields(self):
        inst = Instruction(
            id=1,
            target_code="E11.9",
            contrastive_rule="avoid_false_positive",
            fpr=0.05,
            fnr=0.10,
            efficacy_score=0.92,
            semantic_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert inst.id == 1
        assert inst.target_code == "E11.9"
        assert inst.contrastive_rule == "avoid_false_positive"
        assert inst.fpr == 0.05
        assert inst.fnr == 0.10
        assert inst.efficacy_score == 0.92
        assert inst.semantic_embedding == [0.1, 0.2, 0.3, 0.4]

    def test_instruction_with_none_semantic_embedding(self):
        inst = Instruction(
            id=2,
            target_code="I10",
            contrastive_rule="avoid_false_negative",
            fpr=0.08,
            fnr=0.03,
            efficacy_score=0.95,
            semantic_embedding=None,
        )
        assert inst.id == 2
        assert inst.semantic_embedding is None

    def test_instruction_default_created_at(self):
        inst = Instruction(
            id=3,
            target_code="J44.1",
            contrastive_rule="balance",
            fpr=0.1,
            fnr=0.1,
            efficacy_score=0.90,
        )
        assert inst.created_at is not None


class TestSerializationRoundtrip:
    def test_icd_prediction_roundtrip(self):
        original = ICDPrediction(
            icd_code="E11.9",
            reason="Patient has diabetes mellitus",
            quote="HbA1c 8.5%"
        )
        json_str = original.model_dump_json()
        restored = ICDPrediction.model_validate_json(json_str)
        assert restored.icd_code == original.icd_code
        assert restored.reason == original.reason
        assert restored.quote == original.quote

    def test_icds_model_roundtrip(self):
        original = ICDsModel(
            diagnoses=[
                ICDPrediction(icd_code="E11.9", reason="Type 2 diabetes"),
                ICDPrediction(icd_code="I10", reason="Hypertension"),
            ]
        )
        json_str = original.model_dump_json()
        restored = ICDsModel.model_validate_json(json_str)
        assert len(restored.diagnoses) == len(original.diagnoses)
        assert restored.diagnoses[0].icd_code == "E11.9"
        assert restored.diagnoses[1].icd_code == "I10"

    def test_instruction_roundtrip(self):
        original = Instruction(
            id=42,
            target_code="J44.1",
            contrastive_rule="strict",
            fpr=0.02,
            fnr=0.05,
            efficacy_score=0.97,
            semantic_embedding=[0.5, 0.6, 0.7],
        )
        json_str = original.model_dump_json()
        restored = Instruction.model_validate_json(json_str)
        assert restored.id == original.id
        assert restored.target_code == original.target_code
        assert restored.fpr == original.fpr
        assert restored.fnr == original.fnr
        assert restored.efficacy_score == original.efficacy_score
        assert restored.semantic_embedding == original.semantic_embedding


class TestInvalidData:
    def test_invalid_icd_code_type(self):
        with pytest.raises(ValidationError):
            ICDPrediction(icd_code=123, reason="Valid reason")

    def test_invalid_reason_type(self):
        with pytest.raises(ValidationError):
            ICDPrediction(icd_code="E11.9", reason=123)

    def test_icds_model_invalid_diagnoses_type(self):
        with pytest.raises(ValidationError):
            ICDsModel(diagnoses="not a list")

    def test_instruction_fpr_accepts_any_float(self):
        inst = Instruction(
            id=1,
            target_code="E11.9",
            contrastive_rule="test",
            fpr=-0.5,
            fnr=0.1,
            efficacy_score=0.9,
        )
        assert inst.fpr == -0.5

    def test_instruction_efficacy_score_accepts_any_float(self):
        inst = Instruction(
            id=1,
            target_code="E11.9",
            contrastive_rule="test",
            fpr=0.1,
            fnr=0.1,
            efficacy_score=1.5,
        )
        assert inst.efficacy_score == 1.5