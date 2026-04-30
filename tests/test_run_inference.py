"""Integration tests for the lockstep MERLIN pipeline.

We replace the Generator with an in-process fake so we can drive the
inner loop deterministically across multiple iterations.
"""

import json
from typing import List
from unittest.mock import patch

import pytest

from src.merlin2.generator import Generator, GenerateRequest, GenerateResult
from src.merlin2.pipeline import MERLINPipeline
from src.merlin2.retriever import Retriever
from src.merlin2.verifier import HaltReason, Verifier
from src.meta_verifier.schemas import Instruction, InstructionType
from src.prompter import ICDPrediction, ICDsModel


def _model(codes: List[str]) -> ICDsModel:
    return ICDsModel(diagnoses=[ICDPrediction(icd_code=c, reason="r") for c in codes])


class _ScriptedGenerator(Generator):
    """Returns a queue of pre-canned predictions, one per case per call."""

    def __init__(self, scripted_predictions_per_call):
        super().__init__()
        self._calls = list(scripted_predictions_per_call)

    async def generate_batch(self, requests):
        if not self._calls:
            raise AssertionError("Generator called more times than scripted")
        codes_per_case = self._calls.pop(0)
        return [
            GenerateResult(
                prediction=_model(codes),
                raw_response=json.dumps({"diagnoses": [{"icd_code": c, "reason": "r"} for c in codes]}),
                prompt="<prompt>",
            )
            for codes in codes_per_case
        ]


@pytest.fixture
def base_config():
    return {
        "job_name": "t",
        "k8s": {"namespace": "ns"},
        "model": {"name": "m"},
        "inference": {"temperature": 0.0, "max_tokens": 256},
        "merlin2": {
            "max_iterations": 3,
            "convergence_threshold": 0.9,
            "max_tokens_budget": 10_000,
            "sim_threshold": 0.5,
            "fpr_threshold": 0.5,
            "fnr_threshold": 0.5,
            "learning_rate": 1.0,
        },
    }


def _semantic_instr(id_, target=("E11",)):
    return Instruction(
        instruction_id=id_,
        type=InstructionType.SEMANTIC,
        instruction_text=f"rule-{id_}",
        description="cue",
        target_codes=list(target),
        source_hadm_ids=["x"],
        semantic_embedding=[1.0, 0.0],
    )


def _fp_warning(id_, code):
    """Threshold instruction tied to a single ICD code with high FPR."""
    return Instruction(
        instruction_id=id_,
        type=InstructionType.FP_WARNING,
        instruction_text=f"caution on {code}",
        description=f"high-FPR warning for {code}",
        target_codes=[code],
        fpr_at_creation=0.9,
    )


class TestEmptyDB:
    @pytest.mark.asyncio
    async def test_zero_shot_then_halts_when_db_empty(self, base_config):
        gen = _ScriptedGenerator([[["I10"]]])
        retriever = Retriever()
        retriever.load_instructions([])
        verifier = Verifier(max_iterations=3, max_tokens_budget=10_000, convergence_threshold=0.9)

        pipeline = MERLINPipeline(base_config, generator=gen, retriever=retriever, verifier=verifier)
        results = await pipeline.run(["a note"])
        assert len(results) == 1
        assert results[0].iterations == 1
        assert results[0].halt_reason == HaltReason.EMPTY_DB


class TestHalts:
    @pytest.mark.asyncio
    async def test_max_iterations(self, base_config):
        # Each iteration's prediction triggers a different threshold
        # instruction; that way no single iteration retrieves them all
        # and dedup doesn't short-circuit.
        gen = _ScriptedGenerator([[["A"]], [["B"]], [["C"]]])
        retriever = Retriever(sim_threshold=0.99, fpr_threshold=0.5)
        retriever.load_instructions(
            [_fp_warning(1, "A"), _fp_warning(2, "B"), _fp_warning(3, "C")]
        )
        verifier = Verifier(max_iterations=3, max_tokens_budget=100_000, convergence_threshold=0.99)

        pipeline = MERLINPipeline(base_config, generator=gen, retriever=retriever, verifier=verifier)
        with patch("src.merlin2.retriever.encode_single_text", return_value=[0.0, 0.0]):
            results = await pipeline.run(["note"])
        assert results[0].iterations == 3
        assert results[0].halt_reason == HaltReason.MAX_ITERATIONS_REACHED

    @pytest.mark.asyncio
    async def test_convergence(self, base_config):
        # Same prediction twice => Jaccard 1.0 => convergence at t=1.
        gen = _ScriptedGenerator([[["I10"]], [["I10"]]])
        retriever = Retriever(sim_threshold=0.0)
        retriever.load_instructions([_semantic_instr(1), _semantic_instr(2)])
        verifier = Verifier(max_iterations=5, max_tokens_budget=100_000, convergence_threshold=0.9)

        pipeline = MERLINPipeline(base_config, generator=gen, retriever=retriever, verifier=verifier)
        with patch("src.merlin2.retriever.encode_single_text", return_value=[1.0, 0.0]):
            results = await pipeline.run(["note"])
        assert results[0].halt_reason == HaltReason.CONVERGENCE

    @pytest.mark.asyncio
    async def test_no_new_instructions_after_dedup(self, base_config):
        # Only one instruction in the bank; iteration 1 retrieves it, iteration 2
        # has nothing left after dedup => halt with NO_NEW_INSTRUCTIONS.
        gen = _ScriptedGenerator([[["I10"]], [["E11"]]])
        retriever = Retriever(sim_threshold=0.0)
        retriever.load_instructions([_semantic_instr(1)])
        verifier = Verifier(max_iterations=5, max_tokens_budget=100_000, convergence_threshold=0.99)

        pipeline = MERLINPipeline(base_config, generator=gen, retriever=retriever, verifier=verifier)
        with patch("src.merlin2.retriever.encode_single_text", return_value=[1.0, 0.0]):
            results = await pipeline.run(["note"])
        assert results[0].halt_reason == HaltReason.NO_NEW_INSTRUCTIONS


class TestEfficacyUpdate:
    @pytest.mark.asyncio
    async def test_efficacy_increases_with_positive_delta_f1(self, base_config):
        # Ground truth = ["I10"]. t=0 predicts wrong -> f1=0. t=1 predicts right -> f1=1.
        # Instruction retrieved fresh at t=1 should get +1.0 * learning_rate * rareness.
        gen = _ScriptedGenerator([[["X"]], [["I10"]]])
        instr = _semantic_instr(1)
        retriever = Retriever(sim_threshold=0.0)
        retriever.load_instructions([instr])
        verifier = Verifier(max_iterations=3, max_tokens_budget=100_000, convergence_threshold=0.99)

        cfg = dict(base_config)
        cfg["merlin2"] = dict(base_config["merlin2"], learning_rate=2.0)
        pipeline = MERLINPipeline(cfg, generator=gen, retriever=retriever, verifier=verifier)

        with patch("src.merlin2.retriever.encode_single_text", return_value=[1.0, 0.0]):
            await pipeline.run(
                admission_notes=["note"],
                hadm_ids=["1"],
                ground_truth_codes=[["I10"]],
                rareness_factors=[3.0],
            )
        # delta F1 = 1.0 - 0.0 = 1.0; lr=2.0; rareness=3.0 => +6.0
        assert instr.efficacy_score == pytest.approx(6.0)
