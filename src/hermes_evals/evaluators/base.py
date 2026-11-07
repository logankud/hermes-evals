"""The evaluator bases. Every concrete evaluator lives in its own module and extends one of them.

  Evaluator                the contract: (input, expected, trajectory) -> Score
  CodeEvaluator            deterministic logic, or a small local classifier; no model call
  LLMJudgeEvaluator        asks a chat model to score the run; subclasses write the prompt
  DecisionJudgeEvaluator   asks a decision model (TypeSafe Jev) typed questions; answers are calibrated
                           probabilities; subclasses write the questions and turn answers into a score

All share one signature, so swapping a code check for a judge is a one-line change in cases.jsonl.
Judges are measurements that can be wrong: calibrate one against human labels before reporting its scores.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from ..models import HarnessError, Score, Trajectory


class Evaluator(ABC):
    def __init__(self, name: str | None = None, threshold: float = 1.0):
        self.name = name or type(self).__name__
        self.threshold = threshold

    def __call__(self, input: str, expected: Any, trajectory: Trajectory) -> Score:
        value, reason = self.score(input, expected, trajectory)
        return Score(self.name, value, value >= self.threshold, reason)

    def check(self, expected: Any) -> None:
        """Raise HarnessError if `expected` is malformed. Called when the dataset loads, before any agent call."""

    @abstractmethod
    def score(self, input: str, expected: Any, trajectory: Trajectory) -> tuple[float, str]:
        """Return a score from 0.0 to 1.0 and a one-line reason."""


class CodeEvaluator(Evaluator, ABC):
    """Deterministic evaluators: same trajectory in, same score out, no model call."""


def _post_json(name: str, url: str, api_key_env: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """POST JSON with a bearer key from the environment; shared by the model-backed judges."""
    key = os.environ.get(api_key_env)
    if not key:
        raise HarnessError(f"{name}: set {api_key_env} to use this judge")
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise HarnessError(f"{name}: {url} returned {exc.code}: {exc.read().decode()[:200]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HarnessError(f"{name}: call to {url} failed: {exc}") from exc


class LLMJudgeEvaluator(Evaluator, ABC):
    """Evaluators that ask a model to score the run.

    Subclasses implement `prompt()`. The model is told to reply with JSON {"score": 0..1, "reason": "..."};
    override `parse()` for a different reply format. The judge calls an OpenAI-compatible chat API directly,
    never the agent's Hermes profile, so judging creates no sessions and can use a different model.

    Judges are measurements that can be wrong: calibrate one against human labels before reporting its
    scores (see METHOD.md).
    """

    def __init__(
        self,
        model: str,
        name: str | None = None,
        threshold: float = 1.0,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 400,
        timeout_s: float = 60,
    ):
        super().__init__(name, threshold)
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.max_tokens, self.timeout_s = max_tokens, timeout_s

    @abstractmethod
    def prompt(self, input: str, expected: Any, trajectory: Trajectory) -> str:
        """The full judging prompt for one run."""

    def score(self, input: str, expected: Any, trajectory: Trajectory) -> tuple[float, str]:
        return self.parse(self.complete(self.prompt(input, expected, trajectory)))

    def parse(self, reply: str) -> tuple[float, str]:
        """Read {"score": 0..1, "reason": "..."} from the reply, tolerating code fences around it."""
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
            value = float(data["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise HarnessError(f"{self.name}: judge reply has no numeric 'score': {reply[:200]!r}") from None
        return value, str(data.get("reason", ""))

    def complete(self, prompt: str) -> str:
        """One chat completion at temperature 0. Returns the reply text."""
        data = _post_json(self.name, f"{self.base_url}/chat/completions", self.api_key_env, {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }, self.timeout_s)
        return data["choices"][0]["message"]["content"] or ""


class DecisionJudgeEvaluator(Evaluator, ABC):
    """Evaluators answered by a decision model: TypeSafe Jev, through OpenRouter's Decisions API by default.

    The run is sent as `state` (by default: the request, tool calls and final response) with typed questions:
      {"type": "noul", "instructions": "<statement>"}                        -> {"noul": p}
      {"type": "choice", "instructions": "...", "criteria": {key: meaning}}  -> {"choice": key,
                                                                               "probabilities": {key: p}}
      {"type": "score", "instructions": "...", "criteria": [lowest, ..., highest]}
    Subclasses implement `questions()` and `decide()`, which turns the answers into (score, reason).
    """

    def __init__(
        self,
        model: str = "~typesafe/jev-latest",
        name: str | None = None,
        threshold: float = 1.0,
        url: str = "https://openrouter.ai/api/alpha/decisions",
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout_s: float = 90,
    ):
        super().__init__(name, threshold)
        self.model, self.url, self.api_key_env, self.timeout_s = model, url, api_key_env, timeout_s

    def state(self, input: str, expected: Any, trajectory: Trajectory) -> dict[str, Any]:
        """What the decision model sees. Never includes `expected`, so it can't be nudged toward it."""
        return {
            "request": input,
            "tool_calls": [{"name": c.name, "arguments": (c.arguments or "")[:1000]} for c in trajectory.tool_calls],
            "final_response": trajectory.output[:4000],
        }

    @abstractmethod
    def questions(self, input: str, expected: Any, trajectory: Trajectory) -> dict[str, dict[str, Any]]:
        """Named typed questions, e.g. {"behavior": {"type": "choice", ...}}."""

    @abstractmethod
    def decide(self, answers: dict[str, dict[str, Any]], expected: Any) -> tuple[float, str]:
        """Turn the model's answers into a score from 0 to 1 and a reason."""

    def score(self, input: str, expected: Any, trajectory: Trajectory) -> tuple[float, str]:
        questions = self.questions(input, expected, trajectory)
        return self.decide(self.ask(self.state(input, expected, trajectory), questions), expected)

    def ask(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        data = _post_json(self.name, self.url, self.api_key_env,
                          {"model": self.model, "state": state, "questions": questions}, self.timeout_s)
        answers = data.get("answers")
        if not isinstance(answers, dict) or set(questions) - set(answers):
            raise HarnessError(f"{self.name}: decision reply is missing answers: {str(data)[:200]}")
        return answers
