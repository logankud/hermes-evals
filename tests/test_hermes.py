import json
from pathlib import Path

import pytest

from hermes_evals.hermes import HermesAgent, to_trajectory
from hermes_evals.models import HarnessError, Step

SESSION = json.loads((Path(__file__).parent / "fixtures" / "hermes_run.json").read_text())["session"]


# -- converting a Hermes session record ----------------------------------------------------

def test_session_record_converts_to_ordered_steps():
    traj = to_trajectory(SESSION)
    assert [s.kind for s in traj.steps] == [
        "user", "reasoning", "tool_call", "tool_result", "tool_call", "tool_result", "assistant"]
    assert traj.steps[2].tool == "mcp__aws__manage_aws_athena_query_executions"
    assert traj.steps[0].timestamp < traj.steps[-1].timestamp
    assert traj.input.startswith("How many orders were placed in Q2 2026?")
    assert traj.output.startswith("Q2 2026 had 1,842 orders")


def test_tool_calls_are_paired_with_results():
    traj = to_trajectory(SESSION)
    assert "SELECT count(*)" in traj.tool_calls[0].arguments
    assert traj.tool_calls[0].result == '{"rows": [["1842"]]}'
    assert traj.tool_calls[1].result.startswith("AccessDeniedException")


def test_totals_come_from_the_record_and_the_steps():
    traj = to_trajectory(SESSION)
    assert traj.session_id == "20261024_120000_abc123"
    assert traj.model == "anthropic/claude-haiku-5.5"
    assert traj.tools_used == ["mcp__aws__manage_aws_athena_query_executions"]  # distinct names
    assert (traj.total_steps, traj.total_tool_calls, traj.total_llm_calls) == (7, 2, 3)
    assert (traj.input_tokens, traj.output_tokens, traj.total_tokens) == (31120, 85, 31205)  # input includes cache
    assert (traj.total_duration_s, traj.cost_usd) == (41.5, 0.0041)


def test_step_kind_rejects_unknown_values():
    assert Step("tool_call", "{}").kind == "tool_call"
    with pytest.raises(ValueError):
        Step("tool_use", "{}")


# -- the agent's commands, with the shell replaced -----------------------------------------

class Fake(HermesAgent):
    """HermesAgent whose shell records commands and returns canned output per command."""

    def __init__(self, stdout="", fail_delete=False, **kw):
        super().__init__(**kw)
        self.commands, self.stdout, self.fail_delete = [], stdout, fail_delete

    def _exec(self, argv, stdin=None, timeout=120):
        self.commands.append(self._command(argv))
        failed = self.fail_delete and "delete" in argv

        class Done:
            returncode = 1 if failed else 0
            stderr = "delete refused" if failed else ""
            stdout = "Hermes Agent v0" if "--version" in argv else self.stdout
        return Done()


def result_line(**fields):
    return json.dumps({"type": "result", "session_id": "s1", **fields})


def test_clone_creates_then_deletes_the_clone():
    agent = Fake()
    with agent.clone() as clone:
        assert clone.snapshot["clone"] == clone.name and clone.name.startswith("hermes-evals-")
    assert agent.commands[0][:3] == ["hermes", "profile", "create"]
    assert agent.commands[-1] == ["hermes", "profile", "delete", "-y", clone.name]


def test_clone_is_deleted_when_the_block_fails_and_the_error_survives():
    agent = Fake()
    with pytest.raises(RuntimeError, match="boom"), agent.clone():
        raise RuntimeError("boom")
    assert agent.commands[-1][:3] == ["hermes", "profile", "delete"]


def test_failed_delete_is_attached_to_the_original_error_not_swapped_for_it():
    agent = Fake(fail_delete=True)
    with pytest.raises(RuntimeError, match="boom") as info, agent.clone():
        raise RuntimeError("boom")
    assert any("could not delete clone" in note for note in info.value.__notes__)


def test_run_returns_session_id_and_error():
    agent = Fake(stdout=result_line(error="run budget exceeded", tokens={"total": 900}))
    with agent.clone() as clone:
        assert clone.run("q") == ("s1", "run budget exceeded")


def test_run_raises_when_nothing_reached_the_model():
    agent = Fake(stdout=result_line(error="HTTP 400: credit balance too low", tokens={"total": 0}))
    with pytest.raises(HarnessError, match="credit balance"), agent.clone() as clone:
        clone.run("q")


def test_fetch_targets_the_clone_over_ssh_as_one_quoted_string():
    agent = Fake(stdout=json.dumps({"id": "s1"}), ssh="box")
    with agent.clone() as clone:
        clone.fetch("s1")
        assert agent.commands[-1] == [
            "ssh", "-o", "BatchMode=yes", "box",
            f"hermes -p {clone.name} sessions export --session-id s1 --format jsonl -"]
