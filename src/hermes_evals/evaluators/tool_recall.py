"""Share of the reference trajectory's tool calls that the agent made.

Defined as Vertex AI's trajectory_recall: reference calls matched by some agent call / all reference calls.
Extra calls don't lower it. With nothing expected, 1.
`expected` is the reference: a list of tool names, or of {"name": exact name, "args": ...} (see ToolSpec).
"""

from __future__ import annotations

from ._matching import tool_specs
from .base import CodeEvaluator


class ToolRecall(CodeEvaluator):
    def check(self, expected):
        tool_specs(expected)

    def score(self, input, expected, trajectory):
        specs = tool_specs(expected)
        if not specs:
            return 1.0, "no tools expected"
        missed = [str(s) for s in specs if not any(s.matches(c) for c in trajectory.tool_calls)]
        reason = f"never called: {missed}" if missed else f"all {len(specs)} expected calls made"
        return 1 - len(missed) / len(specs), reason
