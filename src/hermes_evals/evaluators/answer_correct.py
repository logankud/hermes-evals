"""The final answer contains every expected value.

Numbers match within a relative tolerance, ignoring how they're written ($, %, commas, k/M/B), so an expected
48210 accepts "$48,210" and "$48.2k". Text matches case-insensitively. Score is the share of values found.
"""

from __future__ import annotations

from ._matching import as_number, close, numbers
from .base import CodeEvaluator


class AnswerCorrect(CodeEvaluator):
    def __init__(self, rel_tol: float = 0.005, **kwargs):
        super().__init__(**kwargs)
        self.rel_tol = rel_tol

    def score(self, input, expected, trajectory):
        values = expected if isinstance(expected, list) else [expected]
        stated = numbers(trajectory.output)
        missing = []
        for value in values:
            target = as_number(value)
            found = (any(close(n, target, self.rel_tol) for n in stated) if target is not None
                     else str(value).lower() in trajectory.output.lower())
            if not found:
                missing.append(value)
        return 1 - len(missing) / len(values), f"missing {missing}" if missing else "every expected value present"
