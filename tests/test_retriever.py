import pytest

from src.merlin2.retriever import Retriever
from src.meta_verifier.schemas import Instruction


@pytest.fixture
def sample_instructions():
    return [
        Instruction(
            id=1,
            target_code="E11.9",
            contrastive_rule="avoid false positive for diabetes",
            error_type="false_positive",
            quote="Patient has elevated blood glucose levels",
            fpr=0.3,
            fnr=0.05,
            efficacy_score=0.75,
            semantic_embedding=None,
        ),
        Instruction(
            id=2,
            target_code="I10",
            contrastive_rule="avoid false negative for hypertension",
            error_type="false_negative",
            quote="History of hypertension documented",
            fpr=0.05,
            fnr=0.25,
            efficacy_score=0.65,
            semantic_embedding=None,
        ),
        Instruction(
            id=3,
            target_code="J44.1",
            contrastive_rule="high efficacy COPD instruction",
            error_type="false_positive",
            quote="COPD with acute exacerbation",
            fpr=0.08,
            fnr=0.08,
            efficacy_score=0.8,
            semantic_embedding=None,
        ),
        Instruction(
            id=4,
            target_code="Z00.0",
            contrastive_rule="low metrics checkup instruction",
            error_type="false_negative",
            quote="Routine checkup noted",
            fpr=0.01,
            fnr=0.02,
            efficacy_score=0.3,
            semantic_embedding=None,
        ),
    ]


class TestRetrieverAddInstruction:
    def test_add_instruction_stores_correctly(self, sample_instructions):
        retriever = Retriever()
        retriever.add_instruction(sample_instructions[0])
        retriever.add_instruction(sample_instructions[1])

        assert len(retriever._instructions) == 2
        assert retriever._instructions[0].target_code == "E11.9"
        assert retriever._instructions[1].target_code == "I10"


class TestRetrieverRetrieve:
    def test_retrieve_with_matching_predicted_codes_and_high_fpr(
        self, sample_instructions
    ):
        retriever = Retriever(fpr_threshold=0.1)
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("patient with diabetes", ["E11.9"])

        assert len(result) >= 1
        assert any(i.target_code == "E11.9" for i in result)

    def test_retrieve_with_matching_predicted_codes_and_high_fnr(
        self, sample_instructions
    ):
        retriever = Retriever(fnr_threshold=0.1)
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("patient with hypertension", ["I10"])

        assert len(result) >= 1
        assert any(i.target_code == "I10" for i in result)

    def test_retrieve_filters_by_predicted_codes(self, sample_instructions):
        retriever = Retriever()
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("patient visit", ["E11.9", "I10"])

        for instr in result:
            assert instr.target_code in ["E11.9", "I10"]

    def test_retrieve_respects_max_tokens_budget(self, sample_instructions):
        retriever = Retriever(max_tokens_budget=10)
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("patient checkup", ["E11.9", "I10", "J44.1", "Z00.0"])

        total_tokens = sum(retriever._estimate_tokens(i) for i in result)
        assert total_tokens <= 10

    def test_retrieve_returns_empty_when_no_instructions_meet_threshold(
        self, sample_instructions
    ):
        retriever = Retriever(
            sim_threshold=0.9, fpr_threshold=0.5, fnr_threshold=0.5
        )
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("random note text", ["E11.9", "I10", "J44.1"])

        assert len(result) == 0

    def test_priority_ordering_by_efficacy_score(self, sample_instructions):
        retriever = Retriever(fpr_threshold=0.1)
        for instr in sample_instructions:
            retriever.add_instruction(instr)

        result = retriever.retrieve("patient", ["E11.9", "I10", "J44.1", "Z00.0"])

        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i].efficacy_score >= result[i + 1].efficacy_score