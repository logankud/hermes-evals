"""Did the agent answer, assume, clarify or decline, as the case expects? Judged by TypeSafe Jev.

Asks TypeSafe Jev one `choice` question about the run. Jev never sees the expected behavior, so it can't be
nudged toward it, and its choices can be compared with human labels for calibration. It sees the tool calls,
so a question asked through Hermes's `clarify` tool counts.

Score: Jev's calibrated probability that the agent did the expected behavior. Passes at 0.5 by default.
"""

from __future__ import annotations

from ..models import HarnessError
from .base import DecisionJudgeEvaluator

BEHAVIORS = {
    "answer": "answered or completed the task, with no material assumption needed",
    "assume": "answered and explicitly stated an assumption it made to resolve an ambiguity in the request",
    "clarify": "did not answer; asked the user what they meant, in its reply or with a clarify tool",
    "decline": "refused to do what was asked",
}


class Behavior(DecisionJudgeEvaluator):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)

    def questions(self, input, expected, trajectory):
        if expected not in BEHAVIORS:
            raise HarnessError(f"{self.name}: expected must be one of {sorted(BEHAVIORS)}, got {expected!r}")
        return {"behavior": {
            "type": "choice",
            "instructions": "How did the agent respond to the request? Judge what it did, not whether it was correct.",
            "criteria": BEHAVIORS,
        }}

    def decide(self, answers, expected):
        answer = answers["behavior"]
        probabilities = answer.get("probabilities") or {}
        if expected not in probabilities:
            raise HarnessError(f"{self.name}: decision reply has no probability for {expected!r}: {answer}")
        split = ", ".join(f"{k} {float(v):.2f}" for k, v in sorted(probabilities.items(), key=lambda kv: -float(kv[1])))
        return float(probabilities[expected]), f"Jev chose {answer.get('choice')!r} ({split})"
