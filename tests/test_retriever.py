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
    SEM_ICD,
    SEM_ILLNESS,
    SEM_ALLERGIES,
    SEM_COMPLAINT,
    SEM_NOTE,
    THRESHOLD_FNR,
    THRESHOLD_FPR,
    Retriever,
    is_semantic_path,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.3)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fp_warning_does_not_fire_when_code_not_predicted(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fpr_threshold=0.1,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        result = r.retrieve("note", previous_predicted_codes=None)
        assert result.instructions == []

    def test_no_threshold_path_when_code_stats_empty(self, patched_encoder):
        # Without code_stats, the threshold path should be silent regardless
        # of what was predicted.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(sim_note_threshold=0.99, sim_icd_threshold=0.99, fpr_threshold=0.5, code_stats={})
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []


class TestFNCooccurrencePath:
    """FN warnings fire when their target code is in the cooccurring set
    of the previous prediction, NOT when it is the prediction itself."""

    def test_fn_warning_fires_when_target_cooccurs_with_predicted(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        # Predicted: I10 (hypertension). Cooccurring (high lift): N18 (CKD).
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("E11", 5.0)]},
            code_stats={"N18": _stat_fn("N18", 0.6)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fn_warning_does_not_fire_below_fnr_threshold(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("N18", 5.0)]},
            code_stats={},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10"])
        assert result.instructions == []

    def test_fn_warning_text_names_triggering_predicted_codes(self, patched_encoder):
        # The warning text must tell the model which of its own predicted codes
        # co-occur with the missed code — so it understands *why* the warning fired.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("N18", 5.0)], "E11": [("N18", 4.0)]},
            code_stats={"N18": _stat_fn("N18", 0.6, support_true=15)},
        )
        result = r.retrieve("note", previous_predicted_codes=["I10", "E11"])
        assert len(result.instructions) == 1
        text = result.instructions[0].instruction_text
        # Both trigger codes must appear in the warning text
        assert "I10" in text
        assert "E11" in text

    def test_fn_warning_objects_are_independent_across_cases(self, patched_encoder):
        # Each case must get its own Instruction object with its own trigger-code
        # text — not a shared mutable reference that can be overwritten by a
        # later case in the same batch.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fnr_threshold=0.4,
            cooccurrence_index={"I10": [("N18", 5.0)], "K92": [("N18", 4.0)]},
            code_stats={"N18": _stat_fn("N18", 0.6, support_true=15)},
        )
        # Case 1: triggered via I10
        result1 = r.retrieve("note", previous_predicted_codes=["I10"])
        instr1 = result1.instructions[0]

        # Case 2: triggered via K92 (different patient, different prediction)
        result2 = r.retrieve("note", previous_predicted_codes=["K92"])
        instr2 = result2.instructions[0]

        # Each result must have correct trigger codes for its own case
        assert "K92" in instr2.instruction_text
        assert "I10" not in instr2.instruction_text

        # Critical: result1's object must NOT have been mutated by case 2's retrieval
        assert "I10" in instr1.instruction_text, (
            "Case 1's FN warning was mutated by case 2 — objects are shared instead of independent"
        )
        assert "K92" not in instr1.instruction_text

        # Confirm they are distinct objects (no aliasing)
        assert instr1 is not instr2


class TestSyntheticInstructionCaching:
    def test_same_id_across_iterations(self, patched_encoder):
        # Synthesised instruction must keep the same instruction_id across
        # repeated retrieves so the dedup / carry-over machinery works.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        first = r.retrieve("note", previous_predicted_codes=["I10"])
        second = r.retrieve("note", previous_predicted_codes=["I10"])
        assert first.instructions[0].instruction_id == second.instructions[0].instruction_id

    def test_already_retrieved_synthetic_is_suppressed(self, patched_encoder):
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
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
            sim_note_threshold=0.99, sim_icd_threshold=0.99,
            fpr_threshold=0.5,
            code_stats={"I10": _stat_fp("I10", 0.78)},
        )
        r.retrieve("note", previous_predicted_codes=["I10"])
        ids = {i.instruction_id for i in r.instructions}
        assert synthetic_instruction_id("fp", "I10") in ids


class TestSemanticPath:
    def test_semantic_match_above_threshold(self, patched_encoder):
        # No sections → full-note fallback → path is SEM_NOTE
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve("note", previous_predicted_codes=None)
        assert [i.instruction_id for i in result.instructions] == [1]
        assert result.events[0].path == SEM_NOTE

    def test_semantic_below_threshold_skipped(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_note_threshold=0.99, sim_icd_threshold=0.99)
        r.load_instructions([_semantic(1, ["E11"], [0.0, 1.0])])  # cos = 0
        result = r.retrieve("note", previous_predicted_codes=None)
        assert result.instructions == []


class TestSectionBasedSemanticPath:
    """Section embeddings are passed directly; encode_single_text is not called."""

    def test_matching_section_triggers_instruction(self):
        # PRESENT ILLNESS section → path SEM_ILLNESS
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve(
            "full note text",
            previous_predicted_codes=None,
            note_sections={"PRESENT ILLNESS": "DM with neuropathy"},
            section_embeddings={"PRESENT ILLNESS": [1.0, 0.0]},
        )
        assert [i.instruction_id for i in result.instructions] == [1]
        assert result.events[0].path == SEM_ILLNESS

    def test_section_path_reflects_section_name(self):
        # ALLERGIES section → path SEM_ALLERGIES when it matches.
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["Z88"], [1.0, 0.0])])
        result = r.retrieve(
            "full note text",
            previous_predicted_codes=None,
            note_sections={"ALLERGIES": "penicillin allergy"},
            section_embeddings={"ALLERGIES": [1.0, 0.0]},
        )
        assert result.events[0].path == SEM_ALLERGIES

    def test_non_matching_section_does_not_trigger(self):
        # Section embedding [0,1] is orthogonal to instruction embedding [1,0].
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve(
            "full note text",
            previous_predicted_codes=None,
            note_sections={"ALLERGIES": "NKDA"},
            section_embeddings={"ALLERGIES": [0.0, 1.0]},
        )
        assert result.instructions == []

    def test_first_matching_section_wins(self):
        # Two sections both match; instruction must appear exactly once with
        # the path of whichever section the dict iterates first.
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        sections = {"CHIEF COMPLAINT": "DM", "PRESENT ILLNESS": "DM neuropathy"}
        result = r.retrieve(
            "full note text",
            previous_predicted_codes=None,
            note_sections=sections,
            section_embeddings={k: [1.0, 0.0] for k in sections},
        )
        assert len(result.instructions) == 1
        assert result.instructions[0].instruction_id == 1
        # Path belongs to whichever section triggered first (CHIEF COMPLAINT in insertion order)
        assert result.events[0].path == SEM_COMPLAINT

    def test_fallback_to_full_note_when_sections_empty(self, patched_encoder):
        # When note_sections is None, falls back to encoding the full note → SEM_NOTE
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_note_threshold=0.8, sim_icd_threshold=0.8)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve(
            "full note text",
            previous_predicted_codes=None,
            note_sections=None,
        )
        assert [i.instruction_id for i in result.instructions] == [1]
        assert result.events[0].path == SEM_NOTE

    def test_is_semantic_path_helper(self):
        assert is_semantic_path(SEM_ILLNESS)
        assert is_semantic_path(SEM_ICD)
        assert is_semantic_path(SEM_NOTE)
        assert not is_semantic_path(THRESHOLD_FPR)
        assert not is_semantic_path(THRESHOLD_FNR)


class TestDeduplication:
    def test_already_retrieved_ids_are_suppressed(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_note_threshold=0.5, sim_icd_threshold=0.5)
        r.load_instructions([_semantic(1, ["E11"], [1.0, 0.0])])
        result = r.retrieve(
            "note", previous_predicted_codes=None, already_retrieved_ids={1}
        )
        assert result.instructions == []


class TestPriorityAndBudget:
    def test_higher_efficacy_first(self, patched_encoder):
        patched_encoder.return_value = [1.0, 0.0]
        r = Retriever(sim_note_threshold=0.5, sim_icd_threshold=0.5, max_tokens_budget=10_000)
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
        r = Retriever(sim_note_threshold=0.5, sim_icd_threshold=0.5, max_tokens_budget=20)
        r.load_instructions(
            [
                _semantic(1, ["E11"], [1.0, 0.0], text=long_text, efficacy=1.0),
                _semantic(2, ["E11"], [1.0, 0.0], text=long_text, efficacy=0.5),
            ]
        )
        result = r.retrieve("note", previous_predicted_codes=None)
        assert len(result.instructions) <= 1
        assert result.skipped_for_budget >= 1


class TestClusterDeduplication:
    """Post-retrieval cluster dedup collapses near-duplicate semantic
    instructions (instruction-to-instruction cosine sim >= threshold)
    to a single highest-efficacy representative per cluster.

    Tests pass note_embedding explicitly so encode_single_text is never
    called and no monkeypatching is needed.
    """

    def test_near_duplicate_keeps_higher_efficacy(self):
        # Both embeddings are [1,0] → cosine sim = 1.0 → same cluster.
        # Higher-efficacy instruction (id=1, efficacy=0.9) survives; id=2 dropped.
        r = Retriever(sim_note_threshold=0.5, dedup_cluster_threshold=0.9)
        r.load_instructions([
            _semantic(1, ["Z85"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z85"], [1.0, 0.0], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        assert [i.instruction_id for i in result.instructions] == [1]

    def test_dissimilar_instructions_both_kept(self):
        # Orthogonal embeddings ([1,0] vs [0,1]) → cosine sim = 0 → different
        # clusters → both survive.  Note embedding [1,1] (unnormalized, sim ≈ 0.71
        # to both) ensures both are retrieved before dedup runs.
        r = Retriever(sim_note_threshold=0.5, dedup_cluster_threshold=0.9)
        r.load_instructions([
            _semantic(1, ["Z85"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["I10"], [0.0, 1.0], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 1.0])
        assert {i.instruction_id for i in result.instructions} == {1, 2}

    def test_threshold_one_disables_dedup(self):
        # dedup_cluster_threshold=1.0 means nothing is ever "too similar";
        # all retrieved instructions pass through.
        r = Retriever(sim_note_threshold=0.5, dedup_cluster_threshold=1.0)
        r.load_instructions([
            _semantic(1, ["Z85"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z85"], [1.0, 0.0], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        assert len(result.instructions) == 2

    def test_cluster_dedup_does_not_affect_threshold_path(self, patched_encoder):
        # FPR/FNR instructions have no embedding and must not be collapsed by
        # the semantic dedup step regardless of threshold setting.
        patched_encoder.return_value = [0.0, 0.0]
        r = Retriever(
            sim_note_threshold=0.99,
            dedup_cluster_threshold=0.9,
            fpr_threshold=0.5,
            code_stats={
                "I10": _stat_fp("I10", 0.78),
                "E11": _stat_fp("E11", 0.81),
            },
        )
        result = r.retrieve("note", previous_predicted_codes=["I10", "E11"])
        assert len(result.instructions) == 2
        assert all(i.type == InstructionType.FP_WARNING for i in result.instructions)

    def test_three_cluster_two_distinct(self):
        # Three instructions: ids 1 and 2 are near-duplicates ([1,0] vs [1,0]),
        # id 3 is distinct ([0,1]).  After dedup: ids 1 and 3 survive.
        # Note embedding [1,1] hits all three (sim ≈ 0.71 to each axis).
        r = Retriever(sim_note_threshold=0.5, dedup_cluster_threshold=0.9)
        r.load_instructions([
            _semantic(1, ["Z85"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z85"], [1.0, 0.0], efficacy=0.5),
            _semantic(3, ["I10"], [0.0, 1.0], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 1.0])
        ids = {i.instruction_id for i in result.instructions}
        assert ids == {1, 3}


class TestPerCodeCap:
    """max_instructions_per_code limits how many semantic instructions can
    target the same 3-digit ICD code; highest-efficacy ones survive.

    Tests pass note_embedding explicitly to avoid encode_single_text calls.
    """

    def test_excess_instructions_for_same_code_are_dropped(self):
        # Three instructions all targeting Z86; cap=2 → only top-2 by efficacy kept.
        r = Retriever(sim_note_threshold=0.5, max_instructions_per_code=2)
        r.load_instructions([
            _semantic(1, ["Z86"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z86"], [0.9, 0.1], efficacy=0.5),
            _semantic(3, ["Z86"], [0.8, 0.2], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        ids = {i.instruction_id for i in result.instructions}
        assert ids == {1, 2}

    def test_different_codes_are_independent(self):
        # Two instructions for Z86 (cap reached) + two for I10 (separate counter).
        # All four should survive when cap=2.
        r = Retriever(sim_note_threshold=0.5, max_instructions_per_code=2)
        r.load_instructions([
            _semantic(1, ["Z86"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z86"], [1.0, 0.0], efficacy=0.5),
            _semantic(3, ["I10"], [1.0, 0.0], efficacy=0.4),
            _semantic(4, ["I10"], [1.0, 0.0], efficacy=0.2),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        assert {i.instruction_id for i in result.instructions} == {1, 2, 3, 4}

    def test_multi_code_instruction_counts_against_each_code(self):
        # Instruction 1 targets [Z86, I10] (counts against both).
        # Instruction 2 targets [Z86] — Z86 already at cap=1 → dropped.
        # Instruction 3 targets [I10] — I10 already at cap=1 → dropped.
        r = Retriever(sim_note_threshold=0.5, max_instructions_per_code=1)
        r.load_instructions([
            _semantic(1, ["Z86", "I10"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z86"],        [1.0, 0.0], efficacy=0.5),
            _semantic(3, ["I10"],        [1.0, 0.0], efficacy=0.3),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        assert [i.instruction_id for i in result.instructions] == [1]

    def test_none_disables_cap(self):
        # max_instructions_per_code=None (default) → no limit applied.
        r = Retriever(sim_note_threshold=0.5, max_instructions_per_code=None)
        r.load_instructions([
            _semantic(1, ["Z86"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z86"], [1.0, 0.0], efficacy=0.5),
            _semantic(3, ["Z86"], [1.0, 0.0], efficacy=0.1),
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        assert len(result.instructions) == 3

    def test_no_target_codes_always_admitted(self):
        # An instruction with empty target_codes has nothing to count against
        # and must always pass through regardless of cap.
        r = Retriever(sim_note_threshold=0.5, max_instructions_per_code=1)
        r.load_instructions([
            _semantic(1, ["Z86"], [1.0, 0.0], efficacy=0.9),
            _semantic(2, ["Z86"], [1.0, 0.0], efficacy=0.5),
            _semantic(3, [],      [1.0, 0.0], efficacy=0.1),  # no target codes
        ])
        result = r.retrieve("note", previous_predicted_codes=None, note_embedding=[1.0, 0.0])
        ids = {i.instruction_id for i in result.instructions}
        assert ids == {1, 3}  # id=2 dropped (Z86 at cap), id=3 always admitted
