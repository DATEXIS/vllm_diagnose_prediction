"""MERLIN 2 Verifier: per-case halting decisions.

The Verifier is stateless and pure. The Pipeline calls
`should_halt(...)` once per case per iteration and removes any case
whose result is (True, <reason>) from the next wave.

Halting conditions (any one halts):
  * MAX_ITERATIONS_REACHED   — iteration index has reached max_iterations
  * BUDGET_EXHAUSTED         — cumulative think-block tokens exceed budget
  * NO_NEW_INSTRUCTIONS      — retrieval returned 0 new instructions
  * CONVERGENCE              — Jaccard(prev, curr) >= convergence_threshold
  * EMPTY_DB                 — t=0 zero-shot run with an empty memory bank
                               (signaled by the Pipeline; see comment in run())
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HaltReason:
    MAX_ITERATIONS_REACHED: str = "max_iterations_reached"
    BUDGET_EXHAUSTED: str = "budget_exhausted"
    NO_NEW_INSTRUCTIONS: str = "no_new_instructions"
    CONVERGENCE: str = "convergence"
    EMPTY_DB: str = "empty_db"
    PARSE_FAILURE: str = "parse_failure"


class Verifier:
    def __init__(
        self,
        max_iterations: int = 5,
        max_tokens_budget: int = 2500,
        convergence_threshold: float = 0.9,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_tokens_budget = max_tokens_budget
        self.convergence_threshold = convergence_threshold

    def should_halt(
        self,
        iteration: int,
        current_predictions: List[str],
        previous_predictions: Optional[List[str]] = None,
        instructions_retrieved: int = 0,
        cumulative_think_tokens: int = 0,
    ) -> Tuple[bool, str]:
        """Return (halt?, reason). `reason` is "" if not halting."""
        if iteration >= self.max_iterations:
            return True, HaltReason.MAX_ITERATIONS_REACHED

        if cumulative_think_tokens >= self.max_tokens_budget:
            return True, HaltReason.BUDGET_EXHAUSTED

        if instructions_retrieved == 0:
            return True, HaltReason.NO_NEW_INSTRUCTIONS

        if previous_predictions is not None and self._converged(
            current_predictions, previous_predictions
        ):
            return True, HaltReason.CONVERGENCE

        return False, ""

    def _converged(self, current: List[str], previous: List[str]) -> bool:
        cur, prev = set(current), set(previous)
        if not cur and not prev:
            return True
        if not cur or not prev:
            return False
        union = cur | prev
        if not union:
            return True
        jaccard = len(cur & prev) / len(union)
        return jaccard >= self.convergence_threshold
