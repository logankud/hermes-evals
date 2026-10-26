"""Command line: infer, grade, or both (eval)."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from . import calibrate, report
from .config import load_dataset, load_job
from .evaluators import check_judge_keys
from .hermes import HermesAgent
from .models import HarnessError
from .runner import grade, infer, new_run_dir
from .stats import corrected_pass_rate


def _job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("job", help="job file, e.g. jobs/smoke.yaml")
    p.add_argument("--dataset", help="dataset folder (overrides the job)")
    p.add_argument("--ssh", help="run Hermes on this SSH host instead of locally")
    p.add_argument("--profile", help="Hermes profile to clone and evaluate (default: default)")
    p.add_argument("--model", help="model for the agent (default: the profile's)")
    p.add_argument("--provider", help="provider for the agent (default: the profile's)")
    p.add_argument("--limit", type=int, help="only the first N cases")
    p.add_argument("--reps", type=int, help="runs per case")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hermes-evals", description="Evaluate a Hermes agent from its trajectories.")
    sub = p.add_subparsers(dest="command", required=True)
    _job_args(sub.add_parser("eval", help="infer, then grade: a complete evaluation"))
    _job_args(sub.add_parser("infer", help="run the cases through the agent and save trajectories"))
    g = sub.add_parser("grade", help="score a run's saved trajectories and write report.html (no agent calls)")
    g.add_argument("run_dir", type=Path)
    g.add_argument("--dataset", type=Path, help="grade with this dataset instead of the one used for inference")
    lab = sub.add_parser("label", help="write failure_tags.csv for a graded run")
    lab.add_argument("run_dir", type=Path)
    smp = sub.add_parser("sample", help="copy a run's responses into a calibration set for you to label")
    smp.add_argument("run_dir", type=Path)
    smp.add_argument("--into", type=Path, required=True, help="calibration set folder, e.g. calibration/mine")
    smp.add_argument("--evaluators", required=True, help="evaluators you'll label, comma-separated")
    smp.add_argument("--n", type=int, help="how many responses to sample (default: all)")
    smp.add_argument("--seed", type=int, default=0)
    cal = sub.add_parser("calibrate", help="measure an evaluator against your labels in a calibration set")
    cal.add_argument("set_dir", type=Path)
    cal.add_argument("--evaluator", required=True)
    cal.add_argument("--split", choices=("dev", "test"), default="dev",
                     help="tune on dev; look at test once, for the number you report")
    cal.add_argument("--threshold", type=float, help="try a different pass threshold")
    cal.add_argument("--model", help="try a different judge model")
    cal.add_argument("--run", type=Path, help="also correct this run's judged pass rate for the evaluator's errors")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    run_dir = getattr(args, "run_dir", None)
    try:
        if args.command in ("eval", "infer"):
            job = load_job(args.job, dataset=args.dataset, ssh=args.ssh, profile=args.profile, model=args.model,
                           provider=args.provider, limit=args.limit, reps=args.reps)
            if args.command == "eval":
                cases = load_dataset(job.dataset).cases
                check_judge_keys(cases[: job.limit] if job.limit else cases)
            run_dir = new_run_dir(job)
            print(f"{args.command} {job.name}: {job.dataset} → {run_dir}", flush=True)
            try:
                agent = HermesAgent(**job.agent)
            except TypeError as exc:
                raise HarnessError(f"{args.job}: bad setting under 'agent': {exc}") from exc
            infer(job, agent, run_dir)
            if args.command == "infer":
                print(f"\ntrajectories saved. grade with: hermes-evals grade {run_dir}")
                return 0
        if args.command == "sample":
            names = [n.strip() for n in args.evaluators.split(",") if n.strip()]
            added = calibrate.sample(args.run_dir, args.into, names, n=args.n, seed=args.seed)
            print(f"added {added} responses to {args.into}. label them in {args.into / calibrate.LABELS_FILE} "
                  f"(pass or fail per evaluator), then run: hermes-evals calibrate {args.into} --evaluator {names[0]}")
            return 0
        if args.command == "calibrate":
            overrides = {"model": args.model} if args.model else {}
            result = calibrate.calibrate(args.set_dir, args.evaluator, args.split, args.threshold, **overrides)
            print(calibrate.summary(result))
            if args.run:
                observed = calibrate.judged_pass_rate(args.run, args.evaluator)
                corrected = corrected_pass_rate(observed, result.tpr().value, result.tnr().value)
                fixed = "undefined (judge no better than chance)" if math.isnan(corrected) else f"{100 * corrected:.0f}%"
                print()
                print(f"{args.run}: {args.evaluator} passed {100 * observed:.0f}% of runs; "
                      f"corrected for its errors: {fixed}")
            return 0
        if args.command == "label":
            path = report.export_failure_sheet(run_dir)
            print(f"wrote {path}" if path else "nothing to write (no failed runs, or the sheet already exists)")
            return 0
        grade(run_dir, getattr(args, "dataset", None) if args.command == "grade" else None)
        print()
        print(report.summary(run_dir))
        print(f"\nreport: {report.write(run_dir)}")
        return 0
    except HarnessError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\ninterrupted. finished runs are saved in {run_dir}" if run_dir else "\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
