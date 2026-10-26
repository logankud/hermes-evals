"""Every figure in the final answer appears in the run's own data: a tool result or the user's request.

Catches invented numbers without needing an expected answer. A figure the agent derived itself, such as a sum or a
percentage it computed, counts as ungrounded; that's deliberate, since the check can't verify the arithmetic.
Plain integers below `ignore_below` (list markers, small counts like "Top 3") are skipped; percentages,
decimals and amounts ("4.8%", "0.5", "$7") are always checked. Dates and bare years ("Q2 2026",
"June 30", "2026-04-01") are skipped too, since agents add them as context; turn that off with
`ignore_dates: false` if a year-like number could be a real quantity ("2026 orders"). Score is the share of
figures grounded.
"""

from __future__ import annotations

from ._matching import close, figures, numbers, without_dates
from .base import CodeEvaluator


class Grounded(CodeEvaluator):
    def __init__(self, rel_tol: float = 0.005, ignore_below: float = 10, ignore_dates: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.rel_tol, self.ignore_below, self.ignore_dates = rel_tol, ignore_below, ignore_dates

    def score(self, input, expected, trajectory):
        text = without_dates(trajectory.output) if self.ignore_dates else trajectory.output
        claimed = [f.value for f in figures(text) if not (f.plain and abs(f.value) < self.ignore_below)]
        if not claimed:
            return 1.0, "no figures to check"
        sources = numbers(input) + [n for call in trajectory.tool_calls for n in numbers(call.result or "")]
        ungrounded = [n for n in claimed if not any(close(n, s, self.rel_tol) for s in sources)]
        if ungrounded:
            return 1 - len(ungrounded) / len(claimed), f"not in any tool result: {ungrounded}"
        return 1.0, f"all {len(claimed)} figures found in tool results"
