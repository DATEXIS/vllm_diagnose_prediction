"""Tests for the Verifier (per-iteration halt decisions)."""

import pytest

from src.merlin2.verifier import HaltReason, Verifier


class TestShouldHalt:
    def test_no_conditions_met(self):
        v = Verifier(max_iterations=5, max_tokens_budget=2500)
        halt, reason = v.should_halt(
            iteration=1,
            current_predictions=["I10"],
            previous_predictions=["I10"],
            instructions_retrieved=2,
            cumulative_think_tokens=100,
        )
        assert (halt, reason) == (False, "")

    def test_max_iterations(self):
        v = Verifier(max_iterations=5)
        halt, reason = v.should_halt(
            iteration=5,
            current_predictions=["I10"],
            previous_predictions=["I10"],
            instructions_retrieved=2,
            cumulative_think_tokens=100,
        )
        assert (halt, reason) == (True, HaltReason.MAX_ITERATIONS_REACHED)

    def test_budget_exhausted(self):
        v = Verifier(max_tokens_budget=500)
        halt, reason = v.should_halt(
            iteration=1,
            current_predictions=["I10"],
            previous_predictions=["I10"],
            instructions_retrieved=2,
            cumulative_think_tokens=600,
        )
        assert (halt, reason) == (True, HaltReason.BUDGET_EXHAUSTED)

    def test_no_new_instructions(self):
        v = Verifier()
        halt, reason = v.should_halt(
            iteration=2,
            current_predictions=["I10"],
            previous_predictions=["I10"],
            instructions_retrieved=0,
            cumulative_think_tokens=100,
        )
        assert (halt, reason) == (True, HaltReason.NO_NEW_INSTRUCTIONS)

    def test_convergence(self):
        v = Verifier(convergence_threshold=0.9)
        halt, reason = v.should_halt(
            iteration=2,
            current_predictions=["I10", "E11"],
            previous_predictions=["I10", "E11"],
            instructions_retrieved=2,
            cumulative_think_tokens=100,
        )
        assert (halt, reason) == (True, HaltReason.CONVERGENCE)


class TestConverged:
    def test_identical_sets(self):
        v = Verifier(convergence_threshold=0.9)
        assert v._converged(["A", "B"], ["A", "B"]) is True

    def test_disjoint_sets(self):
        v = Verifier(convergence_threshold=0.9)
        assert v._converged(["A"], ["B"]) is False

    def test_empty_both(self):
        v = Verifier()
        assert v._converged([], []) is True

    def test_threshold_boundary(self):
        # Jaccard of {A,B,C} vs {A,B,D} = 2/4 = 0.5 -- below 0.9 threshold
        v = Verifier(convergence_threshold=0.9)
        assert v._converged(["A", "B", "C"], ["A", "B", "D"]) is False
