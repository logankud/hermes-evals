"""The reference tool calls happen in the reference order.

Defined as Vertex AI's trajectory_in_order_match: the reference is a subsequence of the agent's calls, with extra
calls allowed in between. With `exact: true` it is trajectory_exact_match instead: the same calls, in the same
order, and nothing else. Scores 1 or 0.
`expected` is the reference: a list of tool names, or of {"name": exact name, "args": ...} (see ToolSpec).
"""

from __future__ import annotations

from ._matching import tool_specs
from .base import CodeEvaluator


class ToolOrder(CodeEvaluator):
    def __init__(self, exact: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.exact = exact

    def check(self, expected):
        tool_specs(expected)

    def score(self, input, expected, trajectory):
        specs, calls = tool_specs(expected), trajectory.tool_calls
        made = [c.name for c in calls]
        if self.exact:
            ok = len(specs) == len(calls) and all(s.matches(c) for s, c in zip(specs, calls, strict=True))
            return (1.0, "exact match") if ok else (0.0, f"calls were {made}")
        remaining = iter(calls)  # greedy subsequence match: each spec consumes the earliest call it matches
        matched = sum(any(s.matches(c) for c in remaining) for s in specs)
        if matched == len(specs):
            return 1.0, "reference calls made in order"
        return 0.0, f"only {matched} of {len(specs)} reference calls in order; calls were {made}"
