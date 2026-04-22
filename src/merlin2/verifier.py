"""MERLIN 2 Verifier component for checking halting conditions."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HaltReason:
    MAX_ITERATIONS_REACHED: str = "max_iterations_reached"
    BUDGET_EXHAUSTED: str = "budget_exhausted"
    NO_NEW_INSTRUCTIONS: str = "no_new_instructions"
    CONVERGENCE: str = "convergence"


class Verifier:
    """Traffic controller for MERLIN 2 that checks halting conditions after each iteration."""

    def __init__(
        self,
        max_iterations: int = 5,
        max_tokens_per_iteration: int = 512,
        convergence_threshold: float = 0.95,
    ) -> None:
        """Initialize the Verifier.

        Args:
            max_iterations: Maximum number of inference iterations. Default: 5.
            max_tokens_per_iteration: Token budget per iteration. Default: 512.
            convergence_threshold: Jaccard similarity threshold for convergence. Default: 0.95.
        """
        self.max_iterations = max_iterations
        self.max_tokens_per_iteration = max_tokens_per_iteration
        self.convergence_threshold = convergence_threshold

    def should_halt(
        self,
        iteration: int,
        current_predictions: List[str],
        previous_predictions: Optional[List[str]] = None,
        instructions_retrieved: int = 0,
        tokens_used: int = 0,
    ) -> Tuple[bool, str]:
        """Check if the inference loop should halt.

        Args:
            iteration: Current iteration number (1-indexed).
            current_predictions: List of current predictions (ICD codes).
            previous_predictions: List of predictions from previous iteration.
            instructions_retrieved: Number of new instructions retrieved.
            tokens_used: Total tokens used so far.

        Returns:
            Tuple of (should_halt, reason). Reason is empty string if not halting.
        """
        if iteration >= self.max_iterations:
            return True, HaltReason.MAX_ITERATIONS_REACHED

        if tokens_used >= self.max_tokens_per_iteration * iteration:
            return True, HaltReason.BUDGET_EXHAUSTED

        if instructions_retrieved == 0:
            return True, HaltReason.NO_NEW_INSTRUCTIONS

        if previous_predictions is not None:
            if self._check_convergence(current_predictions, previous_predictions):
                return True, HaltReason.CONVERGENCE

        return False, ""

    def _check_convergence(self, current: List[str], previous: List[str]) -> bool:
        """Check if predictions have converged using Jaccard similarity.

        Args:
            current: Current set of predictions.
            previous: Previous set of predictions.

        Returns:
            True if Jaccard similarity >= convergence_threshold.
        """
        if not current and not previous:
            return True

        if not current or not previous:
            return False

        current_set = set(current)
        previous_set = set(previous)

        intersection = len(current_set & previous_set)
        union = len(current_set | previous_set)

        if union == 0:
            return True

        jaccard_similarity = intersection / union
        return jaccard_similarity >= self.convergence_threshold