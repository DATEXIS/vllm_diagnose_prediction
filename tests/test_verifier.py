import pytest

from src.merlin2.verifier import Verifier, HaltReason


class TestVerifierInitialization:
    def test_default_initialization(self):
        verifier = Verifier()
        assert verifier.max_iterations == 5
        assert verifier.max_tokens_per_iteration == 512
        assert verifier.convergence_threshold == 0.95

    def test_custom_initialization(self):
        verifier = Verifier(
            max_iterations=10,
            max_tokens_per_iteration=1024,
            convergence_threshold=0.9
        )
        assert verifier.max_iterations == 10
        assert verifier.max_tokens_per_iteration == 1024
        assert verifier.convergence_threshold == 0.9


class TestShouldHalt:
    def test_no_conditions_met_returns_false(self):
        verifier = Verifier()
        should_halt, reason = verifier.should_halt(
            iteration=1,
            current_predictions=["I10"],
            previous_predictions=None,
            instructions_retrieved=1,
            tokens_used=100
        )
        assert should_halt is False
        assert reason == ""

    def test_max_iterations_reached(self):
        verifier = Verifier(max_iterations=5)
        should_halt, reason = verifier.should_halt(
            iteration=5,
            current_predictions=["I10"],
            previous_predictions=None,
            instructions_retrieved=1,
            tokens_used=100
        )
        assert should_halt is True
        assert reason == HaltReason.MAX_ITERATIONS_REACHED

    def test_budget_exhausted(self):
        verifier = Verifier(max_tokens_per_iteration=512)
        should_halt, reason = verifier.should_halt(
            iteration=2,
            current_predictions=["I10"],
            previous_predictions=None,
            instructions_retrieved=1,
            tokens_used=1024
        )
        assert should_halt is True
        assert reason == HaltReason.BUDGET_EXHAUSTED

    def test_no_new_instructions(self):
        verifier = Verifier()
        should_halt, reason = verifier.should_halt(
            iteration=1,
            current_predictions=["I10"],
            previous_predictions=None,
            instructions_retrieved=0,
            tokens_used=100
        )
        assert should_halt is True
        assert reason == HaltReason.NO_NEW_INSTRUCTIONS

    def test_predictions_converge(self):
        verifier = Verifier(convergence_threshold=0.95)
        should_halt, reason = verifier.should_halt(
            iteration=1,
            current_predictions=["I10", "E11.9"],
            previous_predictions=["I10", "E11.9"],
            instructions_retrieved=1,
            tokens_used=100
        )
        assert should_halt is True
        assert reason == HaltReason.CONVERGENCE

    def test_predictions_dont_converge(self):
        verifier = Verifier(convergence_threshold=0.95)
        should_halt, reason = verifier.should_halt(
            iteration=1,
            current_predictions=["I10", "E11.9"],
            previous_predictions=["J44.0"],
            instructions_retrieved=1,
            tokens_used=100
        )
        assert should_halt is False
        assert reason == ""


class TestCheckConvergence:
    def test_identical_predictions_returns_true(self):
        verifier = Verifier(convergence_threshold=0.95)
        predictions = ["I10", "E11.9", "J44.0"]
        result = verifier._check_convergence(predictions, predictions)
        assert result is True

    def test_completely_different_predictions_returns_false(self):
        verifier = Verifier(convergence_threshold=0.95)
        current = ["I10", "E11.9"]
        previous = ["J44.0", "K50.0"]
        result = verifier._check_convergence(current, previous)
        assert result is False

    def test_empty_predictions_returns_true(self):
        verifier = Verifier(convergence_threshold=0.95)
        result = verifier._check_convergence([], [])
        assert result is True

    def test_partial_overlap_below_threshold(self):
        verifier = Verifier(convergence_threshold=0.95)
        current = ["I10", "E11.9", "J44.0"]
        previous = ["I10", "K50.0", "K70.0"]
        result = verifier._check_convergence(current, previous)
        assert result is False


class TestHaltReason:
    def test_halt_reason_values(self):
        assert HaltReason.MAX_ITERATIONS_REACHED == "max_iterations_reached"
        assert HaltReason.BUDGET_EXHAUSTED == "budget_exhausted"
        assert HaltReason.NO_NEW_INSTRUCTIONS == "no_new_instructions"
        assert HaltReason.CONVERGENCE == "convergence"