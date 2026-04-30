"""Tests for the Hybrid-Retriever.

Two retrieval paths, OR'd:
  * Semantic — cosine similarity on the admission-note embedding (over
    persisted Instructions only).
  * Threshold — synthesised at runtime from the per-code stats table:
      - FP gate: predicted_set ∩ {code: code_stats[code].fpr >= thr}
      - FN gate: cooccurring_set ∩ {code: code_stats[code].fnr >= thr}

We monkeypatch encode_single_text to make the semantic path deterministic.
"""

from unittest.mock import patch

import pytest

from src.merlin2.retriever import (
    SEMANTIC,
    THRESHOLD_FNR,
    THRESHOLD_FPR,
    Retriever,
    synthetic_instruction_id,
)
from src.meta_verifier.code_stats import CodeStat
from src.meta_verifier.schemas import Instruction, InstructionType


def _semantic(id_, target_codes, embedding, description="d", text="t", efficacy=0.0):
    return Instruction(
        instruction_id=id_,
        type=InstructionType.SEMANTIC,
        instruction_text=text,
        description=description,
        target_codes=list(target_codes),
        source_hadm_ids=["x"],
        efficacy_score=efficacy,
        semantic_embedding=list(embedding),
    )


def _stat_fp(code, fpr, support_pred=10):
    return CodeStat(code=code, fpr=fpr, fnr=None, support_pred=support_pred, support_true=0)


def _stat_fn(code, fnr, support_true=10):
    return CodeStat(code=code, fpr=None, fnr=fnr, support_pred=0, support_true=support_true)


@pytest.fixture
def patched_encoder():
    with patch("src.merlin2.retriever.encode_single_text") as m:
        yield m


class TestFPThresholdPath:
    def test_fp_warning_synthesised_when_predicted_and_above_threshold(
        self, patched_encoder
    ):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78, support_pred=20)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert len(result.instructions) == 1
        assert result.instructions[0].type == InstructionType.FP_WARNING
        assert result.instructions[0].target_codes == ["I10"]
        assert result.events[0].path == THRESHOLD_FPR
        assert result.events[0].trigger_value == pytest.approx(0.78)
        # Synthesised instruction text actually contains the rate
        assert "0.78" in result.instructions[0].instruction_text
        # Synthesised IDs come from the deterministic-hash range
        assert result.instructions[0].instruction_id == synthetic_instruction_id("fp", "I10")

    def test_fp_warning_does_not_fire_below_threshold(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.3)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fp_warning_does_not_fire_when_code_not_predicted(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        # The FPR is high but the model didn't predict I10 in this iteration.
        result = r.retrieve("note", previous_predicted_codes=["E11"])
        assert result.instructions == []

    def test_threshold_skipped_at_t0(self, patched_encoder):
        # No previous predictions => no threshold path.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.1,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        result = r.retrieve("note", previous_predicted_codes=None)
        assert result.instructions == []

    def test_no_threshold_path_when_code_stats_empty(self, patched_encoder):
        # Without code_stats, the threshold path should be silent regardless
        # of what was predicted.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(sim_threshold=0.99, fpr_threshold=0.5, code_stats={})
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []


class TestFNCooccurrencePath:
    """FN warnings fire when their target code is in the cooccurring set
    of the previous prediction, NOT when it is the prediction itself."""

    def test_fn_warning_fires_when_target_cooccurs_with_predicted(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        # Predicted: I10 (hypertension). Cooccurring (high lift): N18 (CKD).
        r = Retriever(
            sim_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("N18", 5.0)]},
            code_stats={"N18": _stat_fn("N18", 0.6, support_true=15)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert len(result.instructions) == 1
        assert result.instructions[0].type == InstructionType.FN_WARNING
        assert result.instructions[0].target_codes == ["N18"]
        assert result.events[0].path == THRESHOLD_FNR
        assert result.events[0].trigger_value == pytest.approx(0.6)
        assert result.instructions[0].instruction_id == synthetic_instruction_id("fn", "N18")

    def test_fn_warning_does_not_fire_when_no_cooccurrence(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={},  # I10 has no cooccurring entries
            code_stats={"N18": _stat_fn("N18", 0.6)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fn_warning_does_not_fire_when_target_not_in_cooccurring_set(
        self, patched_encoder
    ):
        patched_encoder.return_value = [0.0, 0.0]
        # I10 cooccurs with E11, not N18 — FN for N18 must not fire even
        # though N18 has a high FNR in code_stats.
        r = Retriever(
            sim_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("E11", 5.0)]},
            code_stats={"N18": _stat_fn("N18", 0.6)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fn_warning_does_not_fire_below_fnr_threshold(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fnr_threshold=0.5,
            cooccurrence_index={"I10": [("N18", 5.0)]},
            code_stats={"N18": _stat_fn("N18", 0.3)},  # below threshold
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fn_warning_does_not_fire_when_code_not_in_code_stats(self, patched_encoder):
        # Cooccurrence says I10 → N18, but code_stats has no entry for N18
        # (i.e. its FNR never crossed the threshold during Loop B).
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("N18", 5.0)]},
            code_stats={},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []


class TestSyntheticInstructionCaching:
    def test_same_id_across_iterations(self, patched_encoder):
        # Synthesised instruction must keep the same instruction_id across
        # repeated retrieves so the dedup / carry-over machinery works.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        first = r.retrieve("note", previous_predicted_codes=["I10"])
        second = r.retrieve("note", previous_predicted_codes=["I10"])
        assert first.instructions[0].instruction_id == second.instructions[0].instruction_id

    def test_already_retrieved_synthetic_is_suppressed(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        synth_id = synthetic_instruction_id("fp", "I10")
        result = r.retrieve(
            "note",
            previous_predicted_codes=["I10"],
            already_retrieved_ids={synth_id},
        )
        assert result.instructions == []

    def test_synthetic_visible_via_instructions_property(self, patched_encoder):
        # The pipeline's _lookup_instructions iterates retriever.instructions
        # for prompt carry-over. After a synthetic warning has been
        # retrieved, it must show up in that property.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        r.retrieve("note", previous_predicted_codes=["I10"])
        ids = {i.instruction_id for i in r.instructions}
        assert synthetic_instruction_id("fp", "I10") in ids


class TestSemanticPath:
    def test_semantic_match_above_threshold(self, patched_encoder):
        # Note vector identical to the instruction embedding => cos = 1.0
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve("note", previous_predicted_codes=None)
        assert [i.instruction_id for i in result.instructions] == [1]
        assert result.events[0].path == SEMANTIC

    def test_semantic_below_threshold_skipped(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_threshold=0.99)
        r.load_instructions([_semantic(1, ["E11"], [0.0, 1.0])])  # cos = 0
        result = r.retrieve("note", previous_predicted_codes=None)
        assert result.instructions == []


class TestDeduplication:
    def test_already_retrieved_ids_are_suppressed(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_threshold=0.5)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve(
            "note", previous_predicted_codes=None, already_retrieved_ids={1}
        )
        assert result.instructions == []


class TestPriorityAndBudget:
    def test_higher_efficacy_first(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_threshold=0.5, max_tokens_budget=10_000)
        r.load_instructions(
            [
                _semantic(1, ["E11"], [1.0, 0.0], efficacy=0.1),
                _semantic(2, ["E11"], [1.0, 0.0], efficacy=0.9),
            ]
        )
        result = r.retrieve("note", previous_predicted_codes=None)
        assert [i.instruction_id for i in result.instructions] == [2, 1]

    def test_budget_limits_selection(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        long_text = "word " * 50
        r = Retriever(sim_threshold=0.5, max_tokens_budget=20)
        r.load_instructions(
            [
                _semantic(1, ["E11"], [1.0, 0.0], text=long_text, efficacy=1.0),
                _semantic(2, ["E11"], [1.0, 0.0], text=long_text, efficacy=0.5),
            ]
        )
        result = r.retrieve("note", previous_predicted_codes=None)
        assert len(result.instructions) <= 1
        assert result.skipped_for_budget >= 1
