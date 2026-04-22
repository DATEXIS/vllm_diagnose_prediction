import pytest
from unittest.mock import MagicMock, patch

from src.merlin2.pipeline import MERLINPipeline, run_inference
from src.merlin2.generator import Generator
from src.merlin2.retriever import Retriever
from src.merlin2.verifier import Verifier, HaltReason
from src.prompter import ICDsModel, ICDPrediction
from src.meta_verifier.schemas import Instruction


@pytest.fixture
def mock_generator():
    generator = MagicMock(spec=Generator)
    generator.generate = MagicMock(side_effect=[
        ICDsModel(diagnoses=[
            ICDPrediction(icd_code="I10", reason="Hypertension"),
            ICDPrediction(icd_code="E11.9", reason="Diabetes"),
        ]),
        ICDsModel(diagnoses=[
            ICDPrediction(icd_code="I10", reason="Hypertension"),
            ICDPrediction(icd_code="E11.9", reason="Diabetes"),
        ]),
        ICDsModel(diagnoses=[
            ICDPrediction(icd_code="I10", reason="Hypertension"),
            ICDPrediction(icd_code="E11.9", reason="Diabetes"),
        ]),
    ])
    return generator


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = MagicMock(return_value=[
        Instruction(
            id=1,
            target_code="I10",
            contrastive_rule="Consider essential hypertension for elevated BP readings.",
            fpr=0.8,
            fnr=0.1,
            efficacy_score=0.9,
        )
    ])
    return retriever


@pytest.fixture
def verifier():
    return Verifier(max_iterations=5)


@pytest.fixture
def sample_admission_note():
    return "Patient presents with elevated blood pressure 160/95 and history of type 2 diabetes."


@pytest.fixture
def test_config():
    """Minimal config for testing MERLINPipeline."""
    return {
        "job_name": "test-job",
        "k8s": {"namespace": "test-ns"},
        "model": {"name": "test/model"},
        "inference": {"temperature": 0.0, "max_tokens": 1024},
        "merlin2": {
            "max_iterations": 5,
            "convergence_threshold": 0.95,
            "max_tokens_budget": 512,
            "sim_threshold": 0.7,
            "fpr_threshold": 0.1,
            "fnr_threshold": 0.1,
        },
        "meta_verifier": {
            "enabled": False,
            "upload_results": False,
        },
    }


class TestMERLINPipelineInitialization:
    def test_initialization(self, mock_generator, mock_retriever, verifier, test_config):
        pipeline = MERLINPipeline(test_config, mock_generator, mock_retriever, verifier)

        assert pipeline.generator is mock_generator
        assert pipeline.retriever is mock_retriever
        assert pipeline.verifier is verifier


class TestRunInference:
    def test_returns_required_keys(self, mock_generator, mock_retriever, verifier, sample_admission_note):
        verifier.should_halt = MagicMock(side_effect=[
            (False, ""),
            (True, HaltReason.CONVERGENCE),
        ])

        result = run_inference(
            sample_admission_note,
            generator=mock_generator,
            retriever=mock_retriever,
            verifier=verifier,
        )

        assert "predictions" in result
        assert "iterations" in result
        assert "halt_reason" in result
        assert "metadata" in result

    def test_completes_within_max_iterations(self, mock_generator, mock_retriever, verifier, sample_admission_note):
        max_iters = 3
        verifier = Verifier(max_iterations=max_iters)

        def halt_side_effect(iteration, **kwargs):
            if iteration >= max_iters:
                return True, HaltReason.MAX_ITERATIONS_REACHED
            return False, ""

        verifier.should_halt = MagicMock(side_effect=halt_side_effect)

        result = run_inference(
            sample_admission_note,
            generator=mock_generator,
            retriever=mock_retriever,
            verifier=verifier,
        )

        assert result["iterations"] <= max_iters

    def test_halts_on_convergence(self, mock_generator, mock_retriever, verifier, sample_admission_note):
        verifier.should_halt = MagicMock(side_effect=[
            (False, ""),
            (True, HaltReason.CONVERGENCE),
        ])

        result = run_inference(
            sample_admission_note,
            generator=mock_generator,
            retriever=mock_retriever,
            verifier=verifier,
        )

        assert result["halt_reason"] == HaltReason.CONVERGENCE

    def test_halts_when_no_new_instructions(self, mock_generator, verifier, sample_admission_note):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.retrieve = MagicMock(return_value=[])

        verifier.should_halt = MagicMock(side_effect=[
            (False, ""),
            (True, HaltReason.NO_NEW_INSTRUCTIONS),
        ])

        result = run_inference(
            sample_admission_note,
            generator=mock_generator,
            retriever=mock_retriever,
            verifier=verifier,
        )

        assert result["halt_reason"] == HaltReason.NO_NEW_INSTRUCTIONS

    def test_halts_when_max_iterations_reached(self, mock_generator, mock_retriever, sample_admission_note):
        max_iters = 2
        verifier = Verifier(max_iterations=max_iters)

        def halt_side_effect(iteration, **kwargs):
            if iteration >= max_iters:
                return True, HaltReason.MAX_ITERATIONS_REACHED
            return False, ""

        verifier.should_halt = MagicMock(side_effect=halt_side_effect)

        result = run_inference(
            sample_admission_note,
            generator=mock_generator,
            retriever=mock_retriever,
            verifier=verifier,
        )

        assert result["iterations"] == max_iters
        assert result["halt_reason"] == HaltReason.MAX_ITERATIONS_REACHED


class TestExtractCodes:
    def test_extracts_icd_code_list(self, mock_generator, mock_retriever, verifier, test_config):
        pipeline = MERLINPipeline(test_config, mock_generator, mock_retriever, verifier)

        icd_model = ICDsModel(diagnoses=[
            ICDPrediction(icd_code="I10", reason="Hypertension"),
            ICDPrediction(icd_code="E11.9", reason="Diabetes"),
            ICDPrediction(icd_code="J44.0", reason="COPD"),
        ])

        codes = pipeline._extract_codes(icd_model)

        assert codes == ["I10", "E11.9", "J44.0"]

    def test_extracts_empty_list(self, mock_generator, mock_retriever, verifier, test_config):
        pipeline = MERLINPipeline(test_config, mock_generator, mock_retriever, verifier)

        icd_model = ICDsModel(diagnoses=[])

        codes = pipeline._extract_codes(icd_model)

        assert codes == []