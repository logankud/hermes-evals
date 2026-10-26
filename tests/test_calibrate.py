"""Calibration: sampling a run into a set, blind labeling, and agreement against your labels."""

import csv
import json
import math
from pathlib import Path

import pytest

from hermes_evals import calibrate
from hermes_evals.calibrate import Calibration, Verdict
from hermes_evals.models import HarnessError

EXAMPLE = Path(__file__).parents[1] / "calibration" / "example"


def verdicts(*rows):
    return [Verdict(f"r{i}", human, score, "") for i, (human, score) in enumerate(rows)]


def test_tpr_and_tnr_measure_each_direction_of_error():
    # 3 you passed (judge passes 2), 2 you failed (judge passes 1): TPR 2/3, TNR 1/2.
    cal = Calibration("j", "dev", 0.5, verdicts((True, 0.9), (True, 0.8), (True, 0.2), (False, 0.7), (False, 0.1)))
    assert cal.tpr().value == pytest.approx(2 / 3) and cal.tnr().value == 0.5
    assert cal.tpr().lo < cal.tpr().value < cal.tpr().hi  # small n gets a real interval
    assert [v.id for v in cal.disagreements()] == ["r2", "r3"]


def test_an_evaluator_error_counts_as_a_disagreement_not_a_dropped_row():
    cal = Calibration("j", "dev", 0.5, verdicts((True, None), (False, 0.1)))
    assert (cal.errors, cal.tpr().value, cal.tnr().value) == (1, 0.0, 1.0)


def test_best_threshold_maximizes_balanced_accuracy():
    # Judge scores separate the classes at 0.6 to 0.8: any threshold in (0.4, 0.6] is perfect.
    cal = Calibration("j", "dev", 0.9, verdicts((True, 0.8), (True, 0.6), (False, 0.4), (False, 0.2)))
    best, balanced = cal.best_threshold()
    assert balanced == 1.0 and 0.4 < best <= 0.6


def test_a_class_with_no_labels_reports_nan_not_a_crash():
    cal = Calibration("j", "dev", 0.5, verdicts((True, 0.9)))
    assert math.isnan(cal.tnr().value)
    assert "TNR  n/a" in calibrate.summary(cal)


def test_split_assignment_is_stable_and_roughly_even():
    ids = [f"run:case-{i}" for i in range(400)]
    assert [calibrate.split_of(i) for i in ids] == [calibrate.split_of(i) for i in ids]
    assert 160 < sum(calibrate.split_of(i) == "test" for i in ids) < 240


def test_calibrating_a_code_evaluator_on_the_example_set():
    cal = calibrate.calibrate(EXAMPLE, "grounded", "dev")
    assert len(cal.verdicts) > 0 and cal.threshold == 1.0
    assert {v.id for v in cal.verdicts} <= {json.loads(line)["id"] for line in (EXAMPLE / "records.jsonl").open()}


def test_threshold_override_changes_the_verdicts():
    strict = calibrate.calibrate(EXAMPLE, "grounded", "dev")
    lenient = calibrate.calibrate(EXAMPLE, "grounded", "dev", threshold=0.5)
    assert lenient.threshold == 0.5 and lenient.tpr().value >= strict.tpr().value


def test_invalid_label_is_a_setup_error(tmp_path):
    (tmp_path / "records.jsonl").write_text(json.dumps(
        {"id": "a", "evaluators": {"grounded": {}}, "trajectory": {"input": "q", "output": "no numbers"}}) + "\n")
    (tmp_path / "labels.csv").write_text("id,split,grounded\na,dev,maybe\n")
    with pytest.raises(HarnessError, match="use pass or fail"):
        calibrate.calibrate(tmp_path, "grounded", "dev")


def test_calibrating_a_judge_without_its_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(HarnessError, match="needs OPENROUTER_API_KEY"):
        calibrate.calibrate(EXAMPLE, "pass_fail", "dev")


def test_sample_copies_a_graded_run_blind_and_keeps_existing_labels(tmp_path):
    from test_pipeline import CASES, FakeAgent

    from hermes_evals.config import Job
    from hermes_evals.runner import grade, infer

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.yaml").write_text("name: t\nversion: '1'\ntask: answer\ncategories: [reasoning, safety]\n")
    (dataset / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in CASES))
    run_dir = infer(Job(name="t", dataset=dataset, reps=1), FakeAgent(), tmp_path / "run")
    grade(run_dir)

    set_dir = tmp_path / "set"
    assert calibrate.sample(run_dir, set_dir, ["read_only"]) == 2
    rows = list(csv.DictReader((set_dir / "labels.csv").open()))
    assert {r["id"] for r in rows} == {"run:math-r1", "run:drop-r1"}
    assert all(r["read_only"] == "" for r in rows)  # blind: no verdicts, nothing pre-filled
    assert "attempted" not in (set_dir / "labels.csv").read_text()  # no evaluator reasons leak in

    rows[0]["read_only"] = "pass"
    with (set_dir / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert calibrate.sample(run_dir, set_dir, ["read_only"]) == 0  # already present: nothing re-added
    kept = {r["id"]: r["read_only"] for r in csv.DictReader((set_dir / "labels.csv").open())}
    assert kept[rows[0]["id"]] == "pass"
