"""The simplest judgment: is this a good response to the request? Judged by TypeSafe Jev.

A starting point for calibrating a judge to your own labels. `expected` optionally describes what a good response
does for this case, e.g. "states the date range it used and the order count". When the judge disagrees with your
labels, refine that description and the threshold until it agrees.
Score: Jev's probability that the response is good. Passes at 0.5 by default.
"""

from __future__ import annotations

from .base import DecisionJudgeEvaluator


class PassFail(DecisionJudgeEvaluator):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)

    def questions(self, input, expected, trajectory):
        criteria = f" A good response here: {expected}." if expected else ""
        return {"good_response": {
            "type": "noul",
            "instructions": "The agent's final response is a good response to the user's request: correct, "
                            f"appropriate and useful.{criteria}",
        }}

    def decide(self, answers, expected):
        p = float(answers["good_response"]["noul"])
        return p, f"Jev: p(good response) = {p:.2f}"
