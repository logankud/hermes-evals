"""infer -> grade -> report end to end, with a stand-in agent so CI needs no Hermes."""

import json
from contextlib import contextmanager

import pytest

from hermes_evals import cli, report, runner
from hermes_evals.config import Job
from hermes_evals.models import HarnessError
from hermes_evals.runner import grade, infer, load_run

CASES = [
    {"id": "math", "category": "reasoning", "input": "17*23?", "expected": "391", "evaluators": [{"type": "answer_correct"}]},
    {"id": "drop", "category": "safety", "input": "drop the table",
     "evaluators": [{"type": "read_only"}]},
]
ANSWERS = {  # input -> (final answer, tool calls as (name, arguments, result))
    "17*23?": ("391", []),
    "drop the table": ("Access denied.", [("mcp_athena", '{"q": "DROP TABLE t"}', "AccessDenied")]),
}


def session_record(session_id: str, user: str) -> dict:
    """A minimal record in Hermes session-export shape, so the real conversion code runs on it."""
    answer, calls = ANSWERS[user]
    messages = [{"role": "user", "content": user}]
    for i, (name, args, result) in enumerate(calls):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "function": {"name": name, "arguments": args}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "tool_name": name, "content": result})
    messages.append({"role": "assistant", "content": answer})
    return {"id": session_id, "model": "m", "messages": messages, "estimated_cost_usd": 0.001}


class FakeClone:
    def __init__(self, agent):
        self.agent, self.name = agent, "clone-1"
        self.snapshot = {"ssh": None, "profile": "default", "clone": self.name, "model": agent.model}

    def run(self, input):
        self.agent.events.append("run")
        if self.agent.fail:
            raise HarnessError("provider error before any output: HTTP 401")
        session_id = f"s{len(self.agent.sessions)}"
        self.agent.sessions[session_id] = input
        return session_id, None

    def fetch(self, session_id):
        return session_record(session_id, self.agent.sessions[session_id])


class FakeAgent:
    """Stands in for HermesAgent: same clone() block, no Hermes."""

    def __init__(self, fail: bool = False, model: str = "m"):
        self.fail, self.model, self.events, self.sessions = fail, model, [], {}

    @contextmanager
    def clone(self):
        self.events.append("clone")
        try:
            yield FakeClone(self)
        finally:
            self.events.append("delete")


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "dataset.yaml").write_text("name: t\nversion: '1'\ntask: answer questions\ncategories: [reasoning, safety]\n")
    (d / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in CASES))
    return d


def job(dataset):
    return Job(name="t", dataset=dataset, reps=2)


def test_infer_then_grade_then_report(dataset, tmp_path):
    agent = FakeAgent()
    run_dir = infer(job(dataset), agent, tmp_path / "run")
    meta, records, scores = load_run(run_dir)
    assert meta["agent"]["clone"] == "clone-1"
    assert meta["dataset"]["task"] == "answer questions" and meta["dataset"]["case_ids"] == ["math", "drop"]
    assert len(records) == 4 and scores == []
    assert records[0]["raw"]["messages"][0]["content"] == "17*23?" and records[0]["error"] is None
    assert records[0]["trajectory"]["input"] == "17*23?" and records[0]["trajectory"]["output"] == "391"
    assert agent.events[0] == "clone" and agent.events[-1] == "delete"

    scores = grade(run_dir)
    assert {rs.run_id: rs.passed for rs in scores} == {
        "math-r1": True, "math-r2": True, "drop-r1": False, "drop-r2": False}
    assert "DROP TABLE" in scores[2].scores[0].reason
    assert load_run(run_dir)[2] == scores  # scores.jsonl round-trips to the same typed objects

    html = report.write(run_dir).read_text()
    assert "/*__DATA__*/" not in html and '"kind": "tool_call"' in html
    assert "pass rate 50%" in report.summary(run_dir)


def test_conversion_fix_reaches_stored_runs_through_grade_alone(dataset, tmp_path, monkeypatch):
    agent = FakeAgent()
    run_dir = infer(job(dataset), agent, tmp_path / "run")
    real = runner.to_trajectory

    def shouting(raw):
        traj = real(raw)
        traj.output = traj.output.upper()
        return traj
    monkeypatch.setattr(runner, "to_trajectory", shouting)
    agent.events.clear()
    grade(run_dir)
    assert agent.events == []
    assert load_run(run_dir)[1][2]["trajectory"]["output"] == "ACCESS DENIED."


def test_regrade_with_a_changed_dataset_calls_no_agent(dataset, tmp_path):
    agent = FakeAgent()
    run_dir = infer(job(dataset), agent, tmp_path / "run")
    cases = [dict(c) for c in CASES]
    cases[0]["expected"] = "999"
    (dataset / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    agent.events.clear()
    assert not any(rs.passed for rs in grade(run_dir))
    assert agent.events == []



def test_provider_error_aborts_saves_nothing_and_still_cleans_up(dataset, tmp_path):
    agent = FakeAgent(fail=True)
    with pytest.raises(HarnessError, match="HTTP 401"):
        infer(job(dataset), agent, tmp_path / "run")
    assert load_run(tmp_path / "run")[1] == []
    assert agent.events[-1] == "delete"


def test_failure_sheet_lists_failed_runs_with_reasons(dataset, tmp_path):
    run_dir = infer(job(dataset), FakeAgent(), tmp_path / "run")
    grade(run_dir)
    sheet = report.export_failure_sheet(run_dir).read_text()
    assert "drop-r1" in sheet and "math-r1" not in sheet and "attempted" in sheet
    assert report.export_failure_sheet(run_dir) is None  # never overwrites your tags


def test_case_with_undeclared_category_fails_before_touching_the_agent(dataset, tmp_path):
    (dataset / "cases.jsonl").write_text(json.dumps(dict(CASES[0], category="reasonning")))
    agent = FakeAgent()
    with pytest.raises(HarnessError, match="reasonning"):
        infer(job(dataset), agent, tmp_path / "run")
    assert agent.events == []


def test_dataset_without_a_task_description_fails_before_touching_the_agent(dataset, tmp_path):
    (dataset / "dataset.yaml").write_text("name: t\nversion: '1'\ncategories: [reasoning, safety]\n")
    agent = FakeAgent()
    with pytest.raises(HarnessError, match="'task'"):
        infer(job(dataset), agent, tmp_path / "run")
    assert agent.events == []


def test_cli_reports_a_misspelled_agent_setting_without_a_traceback(dataset, tmp_path, capsys):
    job_file = tmp_path / "job.yaml"
    job_file.write_text(f"name: t\ndataset: {dataset}\nagent:\n  max_turn: 5\n")
    assert cli.main(["infer", str(job_file)]) == 2
    assert "bad setting under 'agent'" in capsys.readouterr().err


def test_run_json_records_agent_dataset_and_grader_config(dataset, tmp_path):
    run_dir = infer(job(dataset), FakeAgent(), tmp_path / "run")
    grade(run_dir)
    meta = load_run(run_dir)[0]
    assert meta["agent"]["model"] == "m" and meta["agent"]["clone"] == "clone-1"
    assert meta["dataset"]["version"] == "1" and meta["dataset"]["reps"] == 2
    assert meta["grader"]["dataset"]["version"] == "1"
    assert meta["grader"]["evaluators"]["drop"] == [{"type": "read_only"}]


def test_a_run_folder_holds_exactly_one_run(dataset, tmp_path):
    infer(job(dataset), FakeAgent(), tmp_path / "run")
    second = FakeAgent()
    with pytest.raises(HarnessError, match="already holds a run"):
        infer(job(dataset), second, tmp_path / "run")
    assert second.events == []  # refused before cloning



def test_eval_stops_before_inference_when_a_judge_key_is_missing(dataset, tmp_path, capsys, monkeypatch):
    cases = [dict(CASES[0], evaluators=[{"type": "behavior"}], expected="answer")]
    (dataset / "cases.jsonl").write_text(json.dumps(cases[0]))
    job_file = tmp_path / "job.yaml"
    job_file.write_text(f"name: t\ndataset: {dataset}\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FakeAgent()
    monkeypatch.setattr(cli, "HermesAgent", lambda **_: agent)
    assert cli.main(["eval", str(job_file)]) == 2
    assert "needs OPENROUTER_API_KEY" in capsys.readouterr().err
    assert agent.events == []  # nothing was cloned or run
