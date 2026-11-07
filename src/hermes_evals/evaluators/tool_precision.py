"""Share of the agent's tool calls that belong to the reference trajectory.

Defined as Vertex AI's trajectory_precision: matching calls / all calls the agent made. Extra or wrong calls
lower it; repeating a correct call doesn't. With no tools expected, any call scores 0. With no calls, 1.
`expected` is the reference: a list of tool names, or of {"name": exact name, "args": ...} (see ToolSpec).
"""

from __future__ import annotations

from ._matching import tool_specs
from .base import CodeEvaluator


class ToolPrecision(CodeEvaluator):
    def check(self, expected):
        tool_specs(expected)

    def score(self, input, expected, trajectory):
        calls = trajectory.tool_calls
        if not calls:
            return 1.0, "no tool calls"
        specs = tool_specs(expected)
        extra = [c.name for c in calls if not any(s.matches(c) for s in specs)]
        reason = f"unexpected calls: {extra}" if extra else f"all {len(calls)} calls expected"
        return 1 - len(extra) / len(calls), reason
