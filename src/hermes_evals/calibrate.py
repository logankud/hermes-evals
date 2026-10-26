"""Calibrate an evaluator against your own pass/fail labels: an eval of the evaluator.

A calibration set is a folder of responses you've labeled:

    calibration/<set>/
    ├── records.jsonl   one response per line: its trajectory and, per evaluator, what that evaluator expects
    └── labels.csv      your verdict per response and evaluator: pass or fail. You fill this in.

`sample()` copies responses from an eval run into a set, blind: the sheet never shows an evaluator's verdict.
Records can also be written by hand. `calibrate()` re-runs one evaluator on every labeled response and compares
it with your labels.

The method follows current practice for validating LLM judges against expert labels:
- Binary labels from a domain expert, ideally about as many fails as passes.
- TPR (when you said pass, the evaluator passed) and TNR (when you said fail, it failed) are the headline
  numbers. Aim for 90% on both; below 80% the evaluator isn't trustworthy. Judges often err lenient: high TPR,
  low TNR. Kappa is reported as a secondary number.
- Each response is assigned to a dev or a test split. Tune thresholds and criteria on dev; look at test once,
  for the number you report.
- An evaluator error (an unreadable judge reply) counts as a disagreement, never as a dropped row.
- With TPR and TNR known, a judged pass rate can be corrected for the judge's errors (`corrected_pass_rate`).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_dataset
from .evaluators import DecisionJudgeEvaluator, LLMJudgeEvaluator, build
from .hermes import to_trajectory
from .models import HarnessError, Step, ToolCall, Trajectory
from .runner import load_run
from .stats import cohen_kappa, wilson_interval

RECORDS_FILE, LABELS_FILE = "records.jsonl", "labels.csv"
CONTEXT_COLUMNS = ("input", "output", "tool_calls")
LABELS = {"pass": True, "fail": False}


# -- building a set -------------------------------------------------------------------------------

def split_of(record_id: str) -> str:
    """A stable 50/50 dev/test assignment from the record's id, so a response never moves between splits."""
    return "test" if int(hashlib.sha256(record_id.encode()).hexdigest(), 16) % 2 else "dev"


def sample(run_dir: Path, set_dir: Path, evaluators: list[str], n: int | None = None, seed: int = 0) -> int:
    """Copy (a random sample of) a run's responses into a calibration set. Returns how many were added."""
    meta, records, _ = load_run(run_dir)
    if not records:
        raise HarnessError(f"{run_dir}: no trajectories to sample")
    cases = {c.id: c for c in load_dataset(meta["dataset"]["path"]).cases}
    existing = {r["id"] for r in _read_records(set_dir)}
    fresh = [r for r in records if f"{run_dir.name}:{r['run_id']}" not in existing]
    chosen = random.Random(seed).sample(fresh, min(n, len(fresh))) if n else fresh

    added = []
    for r in chosen:
        case = cases.get(r["case_id"])
        specs = {}
        for name in evaluators:
            spec = next((dict(s) for s in (case.evaluators if case else []) if s["type"] == name), {"type": name})
            spec.setdefault("expected", case.expected if case else None)
            specs[name] = {k: v for k, v in spec.items() if k != "type"}
        added.append({
            "id": f"{run_dir.name}:{r['run_id']}",
            "case_id": r["case_id"],
            "source": {"run": str(run_dir), "run_id": r["run_id"], "session_id": r["session_id"]},
            "evaluators": specs,
            "trajectory": dataclasses.asdict(to_trajectory(r["raw"])),
        })
    set_dir.mkdir(parents=True, exist_ok=True)
    with (set_dir / RECORDS_FILE).open("a") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in added)
    _write_sheet(set_dir)
    return len(added)


def _read_records(set_dir: Path) -> list[dict[str, Any]]:
    path = set_dir / RECORDS_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _read_sheet(set_dir: Path) -> dict[str, dict[str, str]]:
    path = set_dir / LABELS_FILE
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _write_sheet(set_dir: Path) -> None:
    """Rewrite labels.csv for every record, keeping labels already filled in. Shows no evaluator verdicts."""
    records, old = _read_records(set_dir), _read_sheet(set_dir)
    names = sorted({name for r in records for name in r.get("evaluators", {})})
    columns = ["id", "split", *names, "note", *CONTEXT_COLUMNS]
    with (set_dir / LABELS_FILE).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            t, prev = r["trajectory"], old.get(r["id"], {})
            tools = "; ".join(f"{c['name']}({(c.get('arguments') or '')[:150]}) -> {(c.get('result') or '')[:150]}"
                              for c in t.get("tool_calls", []))
            writer.writerow({
                "id": r["id"], "split": prev.get("split") or split_of(r["id"]),
                **{name: prev.get(name, "") for name in names}, "note": prev.get("note", ""),
                "input": t.get("input", ""), "output": t.get("output", ""), "tool_calls": tools,
            })


def _trajectory(data: dict[str, Any], record_id: str) -> Trajectory:
    """Rebuild a Trajectory from a record; hand-written records may give only input, output and tool_calls."""
    fields = {f.name for f in dataclasses.fields(Trajectory) if f.init}
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs.setdefault("session_id", record_id)
    kwargs["steps"] = [Step(**s) for s in kwargs.get("steps", [])]
    kwargs["tool_calls"] = [ToolCall(**c) for c in kwargs.get("tool_calls", [])]
    return Trajectory(**kwargs)


# -- measuring agreement --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Verdict:
    id: str
    human: bool  # your label: True = pass
    score: float | None  # the evaluator's score; None if it errored
    reason: str

    def predicted(self, threshold: float) -> bool:
        """The evaluator's verdict at a threshold. An error counts as disagreeing with you."""
        return (not self.human) if self.score is None else self.score >= threshold


@dataclass(frozen=True, slots=True)
class Rate:
    value: float
    lo: float
    hi: float
    n: int


def _rate(hits: int, n: int) -> Rate:
    if n == 0:
        return Rate(math.nan, math.nan, math.nan, 0)
    lo, hi = wilson_interval(hits / n, n)
    return Rate(hits / n, lo, hi, n)


@dataclass(frozen=True, slots=True)
class Calibration:
    evaluator: str
    split: str
    threshold: float
    verdicts: list[Verdict]

    def tpr(self, threshold: float | None = None) -> Rate:
        t = self.threshold if threshold is None else threshold
        positives = [v for v in self.verdicts if v.human]
        return _rate(sum(v.predicted(t) for v in positives), len(positives))

    def tnr(self, threshold: float | None = None) -> Rate:
        t = self.threshold if threshold is None else threshold
        negatives = [v for v in self.verdicts if not v.human]
        return _rate(sum(not v.predicted(t) for v in negatives), len(negatives))

    @property
    def kappa(self) -> float:
        return cohen_kappa([v.human for v in self.verdicts], [v.predicted(self.threshold) for v in self.verdicts])

    @property
    def errors(self) -> int:
        return sum(v.score is None for v in self.verdicts)

    def disagreements(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.predicted(self.threshold) != v.human]

    def best_threshold(self) -> tuple[float, float]:
        """The threshold (0.05 to 0.95) with the highest balanced accuracy, (TPR + TNR) / 2, on this split."""
        def balanced(t: float) -> float:
            rates = [r.value for r in (self.tpr(t), self.tnr(t)) if not math.isnan(r.value)]
            return sum(rates) / len(rates) if rates else math.nan
        grid = [round(0.05 * i, 2) for i in range(1, 20)]
        best = max(grid, key=lambda t: (balanced(t), -abs(t - self.threshold)))
        return best, balanced(best)


def calibrate(set_dir: Path, evaluator: str, split: str = "dev", threshold: float | None = None,
              **overrides: Any) -> Calibration:
    """Re-run one evaluator on every response you've labeled for it in `split`, and compare with your labels.

    `overrides` replace the evaluator's settings (e.g. model=...); `threshold` replaces its pass threshold.
    """
    sheet = _read_sheet(set_dir)
    verdicts, judge_threshold = [], None
    for record in _read_records(set_dir):
        row = sheet.get(record["id"], {})
        label = (row.get(evaluator) or "").strip().lower()
        if not label or (row.get("split") or split_of(record["id"])) != split:
            continue
        if label not in LABELS:
            raise HarnessError(f"{LABELS_FILE}: {record['id']} has {label!r} for {evaluator}; use pass or fail")
        spec = {"type": evaluator, **record.get("evaluators", {}).get(evaluator, {}), **overrides}
        judge, expected = build(spec)
        if isinstance(judge, LLMJudgeEvaluator | DecisionJudgeEvaluator) and not os.environ.get(judge.api_key_env):
            raise HarnessError(f"{evaluator} needs {judge.api_key_env} to calibrate")
        judge_threshold = judge.threshold
        trajectory = _trajectory(record["trajectory"], record["id"])
        try:
            score = judge(trajectory.input, expected, trajectory)
            verdicts.append(Verdict(record["id"], LABELS[label], score.score, score.reason))
        except HarnessError as exc:  # an evaluator that can't answer is counted, not dropped
            verdicts.append(Verdict(record["id"], LABELS[label], None, f"error: {exc}"))
    if not verdicts:
        raise HarnessError(f"{set_dir}: no {split} responses labeled pass or fail for {evaluator}")
    used = threshold if threshold is not None else judge_threshold
    return Calibration(evaluator=evaluator, split=split, threshold=used, verdicts=verdicts)


def summary(cal: Calibration) -> str:
    def pct(r: Rate) -> str:
        return "n/a" if math.isnan(r.value) else f"{100 * r.value:.0f}% ({100 * r.lo:.0f}–{100 * r.hi:.0f}%, n={r.n})"

    tpr, tnr = cal.tpr(), cal.tnr()
    best, balanced = cal.best_threshold()
    counts = f"{len(cal.verdicts)} labeled ({tpr.n} pass, {tnr.n} fail) · {cal.errors} evaluator errors"
    lines = [
        f"{cal.evaluator} · {cal.split} split · threshold {cal.threshold:g} · {counts}",
        f"  TPR  {pct(tpr):<22} when you said pass, it passed",
        f"  TNR  {pct(tnr):<22} when you said fail, it failed",
        f"  kappa {cal.kappa:.2f}" if not math.isnan(cal.kappa) else "  kappa n/a",
        "  target: TPR and TNR both 90% or more; below 80% the evaluator isn't trustworthy",
    ]
    if cal.split == "dev":
        lines.append(f"  best threshold on dev: {best:g} (balanced accuracy {100 * balanced:.0f}%)")
    else:
        lines.append("  test split: report these numbers; don't tune on them")
    if min(tpr.n, tnr.n) < 10:
        lines.append("  note: under 10 labels in a class, so these rates are rough; label more, about as many fails as passes")
    misses = cal.disagreements()
    if misses:
        lines.append("  disagreements:")
        for v in misses:
            score = "error" if v.score is None else f"{v.score:.2f}"
            lines.append(f"    {v.id:<36} you: {'pass' if v.human else 'fail'}  it: {score}  {v.reason[:90]}")
    return "\n".join(lines)


def judged_pass_rate(run_dir: Path, evaluator: str) -> float:
    """Share of a run's responses that `evaluator` passed: the raw rate that `corrected_pass_rate` adjusts."""
    _, _, scores = load_run(run_dir)
    verdicts = [s.passed for rs in scores for s in rs.scores if s.evaluator == evaluator]
    if not verdicts:
        raise HarnessError(f"{run_dir}: no {evaluator} scores; grade the run with a dataset that uses it")
    return sum(verdicts) / len(verdicts)
