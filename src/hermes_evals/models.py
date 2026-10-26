"""Record types shared across the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HarnessError(RuntimeError):
    """A setup problem (bad config, unreachable host, provider rejected the key). Never scored."""


@dataclass
class Case:
    """One dataset entry: the input to send the agent, and what we expect of what it does."""

    id: str
    input: str
    category: str
    expected: Any = None  # default expectation, used by evaluators that don't set their own
    evaluators: list[dict[str, Any]] = field(default_factory=list)


class StepKind(StrEnum):
    """The fixed set of step kinds a trajectory is made of."""

    USER = "user"
    REASONING = "reasoning"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class Step:
    """One event in the trajectory, in order."""

    kind: StepKind
    content: str
    tool: str | None = None
    timestamp: float | None = None

    def __post_init__(self):
        self.kind = StepKind(self.kind)  # accepts "tool_call" or StepKind.TOOL_CALL; rejects anything else


@dataclass
class ToolCall:
    name: str
    arguments: str
    result: str | None = None


@dataclass
class Trajectory:
    """What the agent did for one input. Evaluators read this and nothing else.

    Fields are ordered so a stored record reads top-down: what went in and out, the totals, then every step.
    The conversion code fills the recorded facts; the three counts derived from steps are computed here, so they can
    never disagree with the steps.
    """

    session_id: str
    input: str  # the user message the agent actually received, as recorded
    output: str  # the agent's final response
    model: str | None = None
    tools_used: list[str] = field(init=False)  # distinct tool names, in first-use order
    total_steps: int = field(init=False)
    total_tool_calls: int = field(init=False)
    total_llm_calls: int = 0
    input_tokens: int = 0  # including cached prompt tokens
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    total_duration_s: float = 0.0
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # every call attempted, paired with its result

    def __post_init__(self):
        self.tools_used = list(dict.fromkeys(c.name for c in self.tool_calls))
        self.total_steps = len(self.steps)
        self.total_tool_calls = len(self.tool_calls)


@dataclass
class Score:
    evaluator: str
    score: float  # 0.0 to 1.0
    passed: bool
    reason: str

    def __post_init__(self):
        # Catches buggy custom evaluators; NaN fails this check too.
        if not 0.0 <= self.score <= 1.0:
            raise HarnessError(f"evaluator {self.evaluator!r} returned score {self.score}; scores must be from 0 to 1")


@dataclass
class RunScore:
    """The verdict for one run (one case, one repeat): every evaluator's Score, and whether the run passed."""

    run_id: str
    case_id: str
    rep: int
    category: str
    passed: bool
    scores: list[Score]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunScore:
        """Read back a line of scores.jsonl."""
        return cls(**{**d, "scores": [Score(**s) for s in d["scores"]]})
