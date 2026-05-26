"""MERLIN 2 Verifier: per-case halting decisions.

The Verifier is stateless and pure. The Pipeline calls
`should_halt(...)` once per case per iteration and removes any case
whose result is (True, <reason>) from the next wave.

Halting conditions (any one halts):
  * MAX_ITERATIONS_REACHED   — iteration index has reached max_iterations
  * NO_NEW_INSTRUCTIONS      — retrieval returned 0 new instructions
  * CONVERGENCE              — Jaccard(prev, curr) >= convergence_threshold
  * EMPTY_DB                 — t=0 zero-shot run with an empty memory bank
                               (signaled by the Pipeline; see comment in run())

Convergence and NO_NEW_INSTRUCTIONS are suppressed when the current
prediction has fewer than `min_prediction_size` codes — the model is
under-predicting and needs more iterations to recover, regardless of
whether its (tiny) output looks stable.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HaltReason:
    MAX_ITERATIONS_REACHED: str = "max_iterations_reached"
    NO_NEW_INSTRUCTIONS: str = "no_new_instructions"
    CONVERGENCE: str = "convergence"
    EMPTY_DB: str = "empty_db"
    PARSE_FAILURE: str = "parse_failure"


class Verifier:
    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.9,
        min_prediction_size: int = 3,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_prediction_size = min_prediction_size

    def should_halt(
        self,
        iteration: int,
        current_predictions: List[str],
        previous_predictions: Optional[List[str]] = None,
        instructions_retrieved: int = 0,
    ) -> Tuple[bool, str]:
        """Return (halt?, reason). `reason` is "" if not halting."""
        if iteration >= self.max_iterations:
            return True, HaltReason.MAX_ITERATIONS_REACHED

        # Never halt on stale/convergent signal when the model is under-predicting.
        # A prediction this small almost certainly reflects a model failure or
        # over-pruning rather than a genuine clinical picture — keep trying.
        if len(current_predictions) < self.min_prediction_size:
            return False, ""

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
