"""When the request was ambiguous, the agent said what it assumed. Judged by TypeSafe Jev.

For a question like "revenue last month", a good answer says which definition and period it used (gross or net,
calendar month or trailing 30 days). `expected` optionally names the ambiguity, e.g. "gross vs net revenue", to
focus the question. Score: Jev's probability that the answer states its assumption. Passes at 0.5 by default.
"""

from __future__ import annotations

from .base import DecisionJudgeEvaluator


class StatesAssumptions(DecisionJudgeEvaluator):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)

    def questions(self, input, expected, trajectory):
        about = f" about {expected}" if expected else ""
        return {"states_assumptions": {
            "type": "noul",
            "instructions": f"The final response explicitly states the assumption it made{about} to resolve "
                            "an ambiguity in the request, such as which definition, time period or unit it used.",
        }}

    def decide(self, answers, expected):
        p = float(answers["states_assumptions"]["noul"])
        return p, f"Jev: p(states its assumption) = {p:.2f}"
