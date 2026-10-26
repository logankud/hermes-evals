"""Evaluators score one run. Each lives in its own module; this file maps the names used in cases.jsonl to them.

  correctness   answer_correct, grounded
  process       tool_precision, tool_recall, tool_order
  safety        read_only, resists_injection, no_pii
  behavior      behavior, states_assumptions          (judged by TypeSafe Jev)
  overall       pass_fail                             (judged by TypeSafe Jev; the place to start calibrating)

`expected` for each evaluator comes from its own `expected` field in the case, or falls back to the case's.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from ..models import Case, HarnessError, Score, Trajectory
from .answer_correct import AnswerCorrect
from .base import CodeEvaluator, DecisionJudgeEvaluator, Evaluator, LLMJudgeEvaluator
from .behavior import Behavior
from .grounded import Grounded
from .no_pii import NoPII
from .pass_fail import PassFail
from .read_only import ReadOnly
from .resists_injection import ResistsInjection
from .states_assumptions import StatesAssumptions
from .tool_order import ToolOrder
from .tool_precision import ToolPrecision
from .tool_recall import ToolRecall

__all__ = [
    "BUILT_IN", "CodeEvaluator", "DecisionJudgeEvaluator", "Evaluator", "LLMJudgeEvaluator",
    "build", "check_judge_keys", "evaluate",
]

BUILT_IN: dict[str, type[Evaluator]] = {
    "answer_correct": AnswerCorrect,
    "grounded": Grounded,
    "tool_precision": ToolPrecision,
    "tool_recall": ToolRecall,
    "tool_order": ToolOrder,
    "read_only": ReadOnly,
    "resists_injection": ResistsInjection,
    "no_pii": NoPII,
    "behavior": Behavior,
    "states_assumptions": StatesAssumptions,
    "pass_fail": PassFail,
}


def build(spec: dict[str, Any]) -> tuple[Evaluator, Any]:
    """Turn a case's evaluator spec into (evaluator, its expected value or None).

    `type` is a built-in name or an import path ("my_module:MyEvaluator"). Other keys except `expected`
    are constructor arguments.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    expected = spec.pop("expected", None)
    if kind in BUILT_IN:
        cls = BUILT_IN[kind]
        spec.setdefault("name", kind)
    elif ":" in kind:
        module, _, cls_name = kind.partition(":")
        cls = getattr(importlib.import_module(module), cls_name)
    else:
        raise HarnessError(f"unknown evaluator type {kind!r}; built-ins are {sorted(BUILT_IN)}")
    return cls(**spec), expected


def evaluate(case: Case, trajectory: Trajectory) -> list[Score]:
    scores = []
    for spec in case.evaluators:
        evaluator, expected = build(spec)
        scores.append(evaluator(case.input, case.expected if expected is None else expected, trajectory))
    return scores


def check_judge_keys(cases: list[Case]) -> None:
    """Fail before inference if a case uses a judge whose API key isn't set, instead of after it."""
    for case in cases:
        for spec in case.evaluators:
            evaluator, _ = build(spec)
            judge = isinstance(evaluator, LLMJudgeEvaluator | DecisionJudgeEvaluator)
            if judge and not os.environ.get(evaluator.api_key_env):
                raise HarnessError(f"case {case.id!r} is graded by {evaluator.name}, which needs "
                                   f"{evaluator.api_key_env}. Set it, or run `infer` now and `grade` later.")
