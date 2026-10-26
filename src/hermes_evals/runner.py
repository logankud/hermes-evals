"""The two phases of an evaluation.

infer: clone the Hermes profile, run each case on the clone, fetch each session's raw record.  Needs the agent.
grade: convert the stored records into Trajectories and score them.                          Needs no agent.

A run folder holds run.json (what was evaluated), trajectories.jsonl and, once graded, scores.jsonl.
One infer call makes one clone and runs every case on it, so a run has a single configuration by construction.
Each trajectories.jsonl line keeps the raw record, so grading always re-converts it with the current
code: fixing the conversion needs a regrade, not a rerun.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import Job, load_dataset
from .evaluators import evaluate
from .hermes import HermesAgent, to_trajectory
from .models import HarnessError, RunScore

META_FILE, TRAJECTORIES_FILE, SCORES_FILE = "run.json", "trajectories.jsonl", "scores.jsonl"

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_dir(job: Job, root: Path = Path("results")) -> Path:
    return root / job.name / datetime.now().strftime("%Y%m%d-%H%M%S")


def infer(job: Job, agent: HermesAgent, run_dir: Path) -> Path:
    dataset = load_dataset(job.dataset)
    cases = dataset.cases[: job.limit] if job.limit else dataset.cases
    if (run_dir / META_FILE).exists():
        raise HarnessError(f"{run_dir} already holds a run; each run gets its own folder")
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / TRAJECTORIES_FILE

    with agent.clone() as clone:  # deleted when the block exits, however it exits
        logger.info("agent: %s cloned to %s", clone.snapshot.get("profile", "?"), clone.name)
        (run_dir / META_FILE).write_text(json.dumps({
            "job": job.name,
            "started_at": _now(),
            "agent": clone.snapshot,  # what the agent looked like
            "dataset": {  # what it was asked
                "name": dataset.name, "version": dataset.version, "path": str(dataset.path), "task": dataset.task,
                "case_ids": [c.id for c in cases], "reps": job.reps,
            },
        }, indent=2))

        total, step = len(cases) * job.reps, 0
        for case in cases:
            for rep in range(1, job.reps + 1):
                step += 1
                run_id = f"{case.id}-r{rep}"
                session_id, error = clone.run(case.input)  # raises on setup failures
                raw = clone.fetch(session_id)
                traj = to_trajectory(raw)
                record = {"run_id": run_id, "case_id": case.id, "rep": rep, "input": case.input,
                          "session_id": session_id, "error": error, "raw": raw, "trajectory": asdict(traj)}
                with traj_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                cost = f"${traj.cost_usd:.4f}" if traj.cost_usd is not None else "$?"
                status = f"  ERROR {error[:60]}" if error else ""
                logger.info(f"[{step}/{total}] {run_id:<28} {traj.total_duration_s:>6.1f}s  {traj.total_steps} steps  "
                    f"{traj.total_tool_calls} tool calls  {traj.total_tokens:,} tokens  {cost}{status}")
    logger.info("agent: deleted %s", clone.name)
    return run_dir


def grade(run_dir: Path, dataset_path: Path | None = None) -> list[RunScore]:
    meta, records, _ = load_run(run_dir)
    if not records:
        raise HarnessError(f"{run_dir}: no trajectories to grade; run `hermes-evals infer` first")
    dataset = load_dataset(dataset_path or meta["dataset"]["path"])
    if dataset.version != meta["dataset"]["version"]:
        logger.warning("grading with dataset v%s; inference used v%s", dataset.version, meta["dataset"]["version"])
    cases = {c.id: c for c in dataset.cases}
    scores: list[RunScore] = []
    for r in records:
        traj = to_trajectory(r["raw"])
        r["trajectory"] = asdict(traj)  # keep the stored trajectory in step with the current conversion code
        case = cases.get(r["case_id"])
        if case is None:
            logger.warning("%s: case %r is not in the dataset; skipped", r["run_id"], r["case_id"])
            continue
        case_scores = evaluate(case, traj)
        scores.append(RunScore(
            run_id=r["run_id"],
            case_id=case.id,
            rep=r["rep"],
            category=case.category,
            passed=bool(case_scores) and all(s.passed for s in case_scores) and r["error"] is None,
            scores=case_scores,
        ))
    (run_dir / TRAJECTORIES_FILE).write_text("".join(json.dumps(r) + "\n" for r in records))
    (run_dir / SCORES_FILE).write_text("".join(json.dumps(asdict(rs)) + "\n" for rs in scores))
    meta["grader"] = {  # how it was graded: the dataset version and every case's evaluator config
        "graded_at": _now(),
        "dataset": {"name": dataset.name, "version": dataset.version, "path": str(dataset.path)},
        "evaluators": {c.id: c.evaluators for c in dataset.cases if c.id in {rs.case_id for rs in scores}},
    }
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return scores

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[RunScore]]:
    if not (run_dir / META_FILE).exists():
        raise HarnessError(f"{run_dir}: not a run folder (no {META_FILE})")
    meta = json.loads((run_dir / META_FILE).read_text())
    scores = [RunScore.from_dict(d) for d in _read_jsonl(run_dir / SCORES_FILE)]
    return meta, _read_jsonl(run_dir / TRAJECTORIES_FILE), scores
