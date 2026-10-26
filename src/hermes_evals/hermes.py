"""The Hermes agent under test, reached locally or over SSH.

Every inference run happens on a throwaway clone of the profile:

    with agent.clone() as clone:        # hermes profile create <clone> --clone --clone-from <profile>
        clone.snapshot                  # what is under test: model, MCP servers, skills, fingerprints
        clone.run(input)                # hermes -p <clone> chat ... -> (session_id, error)
        clone.fetch(session_id)         # hermes -p <clone> sessions export ... -> raw record
                                        # leaving the block deletes the clone, however it is left

    to_trajectory(raw)                  # raw record -> Trajectory, used by infer and again by grade

The clone carries config, credentials, SOUL.md, skills and MCP servers, but not memories, session history or
messaging channels, so eval runs can't change the real profile or reach its chat channels. Records must be
fetched inside the block, because the sessions live in the clone.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import yaml

from .models import HarnessError, Step, StepKind, ToolCall, Trajectory

SOURCE_TAG = "hermes-evals"  # session source tag inside the clone


class HermesAgent:
    """A Hermes profile to evaluate, and how to reach it. Holds settings only; runs happen on clones."""

    def __init__(
        self,
        ssh: str | None = None,
        profile: str = "default",
        model: str | None = None,
        provider: str | None = None,
        max_turns: int = 10,
        run_budget_s: int = 180,
        hermes_bin: str = "hermes",
    ):
        self.ssh, self.profile, self.hermes_bin = ssh, profile, hermes_bin
        self.model, self.provider = model, provider
        self.max_turns, self.run_budget_s = max_turns, run_budget_s

    @contextmanager
    def clone(self) -> Iterator[HermesClone]:
        name = f"hermes-evals-{datetime.now():%Y%m%d%H%M%S}"
        out = self._hermes(["profile", "create", name, "--clone", "--clone-from", self.profile, "--no-alias"])
        if out.returncode != 0:
            raise HarnessError(f"could not clone profile {self.profile!r}: {_tail(out)}")
        try:
            yield HermesClone(self, name, self._snapshot(name))
        except BaseException as exc:
            # Delete even on failure or Ctrl-C, but never let a failed delete hide the original error.
            try:
                self._delete(name)
            except HarnessError as cleanup:
                exc.add_note(f"also: {cleanup}")
            raise
        self._delete(name)

    def _delete(self, name: str) -> None:
        out = self._hermes(["profile", "delete", "-y", name], timeout=300)
        if out.returncode != 0:
            raise HarnessError(f"could not delete clone {name!r}; remove it with "
                               f"`hermes profile delete -y {name}`: {_tail(out)}")

    def _snapshot(self, name: str) -> dict[str, Any]:
        """What is under test, with no secrets: names, counts and fingerprints only."""
        show = self._hermes(["profile", "show", name]).stdout
        fields = dict(line.split(":", 1) for line in show.splitlines() if ":" in line and not line.startswith(" "))
        path = fields.get("Path", "").strip()
        config = yaml.safe_load(self._exec(["cat", f"{path}/config.yaml"]).stdout) or {}
        soul = self._exec(["cat", f"{path}/SOUL.md"]).stdout
        model_cfg = config.get("model") or {}
        return {
            "ssh": self.ssh,  # None when Hermes runs on this machine
            "profile": self.profile,
            "clone": name,
            "hermes_version": self._hermes(["--version"]).stdout.splitlines()[0].strip(),
            "model": self.model or model_cfg.get("default"),
            "provider": self.provider or model_cfg.get("provider"),
            "toolsets": config.get("toolsets"),
            "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
            "plugins": (config.get("plugins") or {}).get("enabled"),
            "skills": fields.get("Skills", "").strip(),
            "soul_sha256": hashlib.sha256(soul.encode()).hexdigest()[:16],
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "max_turns": self.max_turns,
            "run_budget_s": self.run_budget_s,
        }

    # -- shell plumbing ---------------------------------------------------------------------

    def _hermes(self, args: list[str], profile: str | None = None, stdin: str | None = None, timeout: float = 120):
        prefix = [self.hermes_bin] + (["-p", profile] if profile else [])
        return self._exec(prefix + args, stdin=stdin, timeout=timeout)

    def _command(self, argv: list[str]) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", self.ssh, shlex.join(argv)] if self.ssh else argv

    def _exec(self, argv: list[str], stdin: str | None = None, timeout: float = 120):
        cmd = self._command(argv)
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise HarnessError(f"'{cmd[0]}' not found. Is Hermes installed and on PATH?") from exc
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(f"timed out after {timeout:.0f}s: {shlex.join(cmd)}") from exc
        if proc.returncode == 255 and self.ssh:
            raise HarnessError(f"ssh to {self.ssh} failed: {proc.stderr.strip()[-300:]}")
        return proc


class HermesClone:
    """A live, throwaway clone of the profile. Only exists inside `with agent.clone() as clone:`."""

    def __init__(self, agent: HermesAgent, name: str, snapshot: dict[str, Any]):
        self.agent, self.name, self.snapshot = agent, name, snapshot

    def run(self, input: str) -> tuple[str, str | None]:
        """Run one input. Returns (session_id, error); raises if nothing reached the model."""
        a = self.agent
        args = ["chat", "--query-file", "-", "--format", "stream-json", "--source", SOURCE_TAG]
        args += (["--model", a.model] if a.model else []) + (["--provider", a.provider] if a.provider else [])
        args += ["--max-turns", str(a.max_turns), "--run-budget", str(a.run_budget_s)]
        # The input goes through stdin so quotes and $(...) in cases never meet a shell.
        out = a._hermes(args, profile=self.name, stdin=input, timeout=a.run_budget_s + 60)
        events = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if result is None:
            raise HarnessError(f"hermes chat returned no result event: {_tail(out)}")
        error = result.get("error")
        if error and not (result.get("tokens") or {}).get("total"):
            # Nothing reached the model: a setup problem (key, credit, provider), not an agent result.
            raise HarnessError(f"provider error before any output: {error}")
        return result["session_id"], error

    def fetch(self, session_id: str) -> dict[str, Any]:
        """The session's raw record, exactly as `hermes sessions export` gives it."""
        args = ["sessions", "export", "--session-id", session_id, "--format", "jsonl", "-"]
        out = self.agent._hermes(args, profile=self.name)
        rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
        if not rows:
            raise HarnessError(f"session {session_id} could not be exported: {_tail(out)}")
        return rows[0]


def _tail(proc) -> str:
    return (proc.stderr or proc.stdout or "").strip()[-300:]


def to_trajectory(raw: dict[str, Any]) -> Trajectory:
    """Convert a `hermes sessions export` row into a Trajectory. Pure: no I/O, so stored records can be
    re-converted at grading time with the current code."""
    steps: list[Step] = []
    calls: list[ToolCall] = []
    by_id: dict[str, ToolCall] = {}
    final = ""
    for msg in raw.get("messages") or []:
        role, ts, content = msg.get("role"), msg.get("timestamp"), msg.get("content") or ""
        if role == "user":
            steps.append(Step(StepKind.USER, content, timestamp=ts))
        elif role == "assistant":
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning:
                steps.append(Step(StepKind.REASONING, reasoning, timestamp=ts))
            if content:
                steps.append(Step(StepKind.ASSISTANT, content, timestamp=ts))
                final = content
            for tc in msg.get("tool_calls") or []:
                call = ToolCall(name=tc["function"]["name"], arguments=tc["function"].get("arguments") or "")
                calls.append(call)
                by_id[tc["id"]] = call
                steps.append(Step(StepKind.TOOL_CALL, call.arguments, tool=call.name, timestamp=ts))
        elif role == "tool":
            call = by_id.get(msg.get("tool_call_id"))
            if call:
                call.result = content
            steps.append(Step(StepKind.TOOL_RESULT, content, tool=msg.get("tool_name"), timestamp=ts))
    user_inputs = [st.content for st in steps if st.kind == StepKind.USER]
    cost = raw.get("actual_cost_usd") or raw.get("estimated_cost_usd")
    input_tokens = sum(int(raw.get(k) or 0) for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    output_tokens = int(raw.get("output_tokens") or 0)
    started, ended = raw.get("started_at"), raw.get("ended_at")
    return Trajectory(
        session_id=raw["id"],
        input=user_inputs[0] if user_inputs else "",
        output=final.strip(),
        model=raw.get("model"),
        total_llm_calls=int(raw.get("api_call_count") or 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=float(cost) if cost is not None else None,
        total_duration_s=round(ended - started, 2) if started and ended else 0.0,
        steps=steps,
        tool_calls=calls,
    )
