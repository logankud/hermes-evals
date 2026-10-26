"""Summarize a graded run: terminal text, a self-contained report.html, and a failure-tag sheet."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from .models import HarnessError, RunScore
from .runner import load_run
from .stats import PassRate, pass_rate

REPORT_FILE, TAGS_FILE = "report.html", "failure_tags.csv"
CAUSES = ("model", "prompt", "grader", "retrieval")


def _rate(scores: list[RunScore]) -> PassRate:
    by_case: dict[str, list[bool]] = defaultdict(list)
    for rs in scores:
        by_case[rs.case_id].append(bool(rs.passed))
    return pass_rate(by_case)


def _clean(value):
    """NaN is not valid JSON; send null instead."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def build(run_dir: Path) -> dict:
    meta, trajectories, scores = load_run(run_dir)
    if not scores:
        raise HarnessError(f"{run_dir}: not graded yet; run `hermes-evals grade {run_dir}`")
    traj = {r["run_id"]: r["trajectory"] for r in trajectories}
    errors = {r["run_id"]: r["error"] for r in trajectories}
    cases: dict[str, dict] = {}
    for rs in scores:
        t = traj[rs.run_id]
        entry = cases.setdefault(rs.case_id, {
            "id": rs.case_id, "category": rs.category, "input": t["input"], "runs": [],
        })
        entry["runs"].append({
            "run_id": rs.run_id, "rep": rs.rep, "passed": rs.passed, "scores": [asdict(sc) for sc in rs.scores],
            "error": errors[rs.run_id], "session_id": t["session_id"], "model": t["model"],
            "output": t["output"][:3000], "tools_used": t["tools_used"],
            "total_steps": t["total_steps"], "total_tool_calls": t["total_tool_calls"], "total_llm_calls": t["total_llm_calls"],
            "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"], "total_tokens": t["total_tokens"],
            "cost_usd": t["cost_usd"], "total_duration_s": t["total_duration_s"],
            "steps": [{"kind": st["kind"], "tool": st["tool"], "content": (st["content"] or "")[:1200]} for st in t["steps"]],
        })
    costs = [t["cost_usd"] for t in traj.values() if t["cost_usd"] is not None]
    return _clean({
        "meta": meta,
        "overall": asdict(_rate(scores)),
        "categories": {
            c: asdict(_rate([rs for rs in scores if rs.category == c])) for c in sorted({rs.category for rs in scores})
        },
        "totals": {
            "runs": len(scores),
            "passed": sum(rs.passed for rs in scores),
            "errors": sum(bool(e) for e in errors.values()),
            "cost_usd": sum(costs) if costs else None,
            "tokens": sum(t["total_tokens"] for t in traj.values()),
        },
        "cases": list(cases.values()),
        "failure_tags": failure_counts(run_dir, scores),
    })


def write(run_dir: Path) -> Path:
    data = build(run_dir)
    template = files("hermes_evals").joinpath("report_template.html").read_text()
    payload = json.dumps(data).replace("</", "<\\/")  # keep the JSON from closing the <script> tag
    out = run_dir / REPORT_FILE
    out.write_text(template.replace("/*__DATA__*/null", payload))
    return out


def summary(run_dir: Path) -> str:
    data = build(run_dir)
    o, t, m = data["overall"], data["totals"], data["meta"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    a = m["agent"]
    agent = f"{a.get('ssh') or 'local'}/{a.get('profile')}  ·  {a.get('model')}"
    lines = [
        f"{m['job']}  ·  {m['dataset']['name']} v{m['dataset']['version']}  ·  {agent}",
        f"pass rate {pct(o['rate'])}  (95% CI {pct(o['lo'])}–{pct(o['hi'])})  over {o['n_cases']} cases, {o['n_runs']} runs",
    ]
    for name, c in data["categories"].items():
        lines.append(f"  {name:<20} {pct(c['rate']):>5}  ({pct(c['lo'])}–{pct(c['hi'])})  n={c['n_runs']}")
    cost = "unknown" if t["cost_usd"] is None else f"${t['cost_usd']:.4f}"
    lines.append(f"errors {t['errors']}  ·  tokens {t['tokens']:,}  ·  agent cost {cost}")
    return "\n".join(lines)


def export_failure_sheet(run_dir: Path) -> Path | None:
    """Write failure_tags.csv: one row per failed run, with a blank cause column to fill in."""
    _, trajectories, scores = load_run(run_dir)
    path = run_dir / TAGS_FILE
    failed = [rs for rs in scores if not rs.passed]
    if not failed or path.exists():
        return None
    responses = {r["run_id"]: r["trajectory"]["output"] for r in trajectories}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "failed_evaluators", "response", f"cause ({'/'.join(CAUSES)})", "note"])
        for rs in failed:
            reasons = "; ".join(f"{sc.evaluator}: {sc.reason}" for sc in rs.scores if not sc.passed)
            w.writerow([rs.run_id, reasons or "run errored", responses.get(rs.run_id, ""), "", ""])
    return path


def failure_counts(run_dir: Path, scores: list[RunScore]) -> dict | None:
    path = run_dir / TAGS_FILE
    if not path.exists():
        return None
    tags = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            cause = row[3].strip().lower() if len(row) > 3 else ""
            if cause and cause not in CAUSES:
                raise HarnessError(f"{TAGS_FILE}: {row[0]} has cause {cause!r}; use one of {CAUSES}")
            if cause:
                tags.append(cause)
    if not tags:
        return None
    return {"counts": dict(Counter(tags)), "tagged": len(tags), "failed": sum(not rs.passed for rs in scores)}
