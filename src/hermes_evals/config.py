"""Load job files and datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import evaluators
from .models import Case, HarnessError


@dataclass
class Job:
    name: str
    dataset: Path
    reps: int = 1
    limit: int | None = None
    agent: dict[str, Any] = field(default_factory=dict)  # HermesAgent settings: ssh, profile, model, ...


@dataclass
class Dataset:
    name: str
    version: str
    path: Path
    task: str  # what the agent is being asked to do
    categories: list[str]
    cases: list[Case]


def load_job(path: str | Path, **overrides: Any) -> Job:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    agent = dict(data.get("agent") or {})
    for key in ("ssh", "profile", "model", "provider"):
        if overrides.get(key) is not None:
            agent[key] = overrides[key]
    dataset = overrides.get("dataset") or data.get("dataset")
    if not dataset:
        raise HarnessError(f"{path}: 'dataset' is required")
    return Job(
        name=data.get("name") or path.stem,
        dataset=Path(dataset),
        reps=int(overrides.get("reps") or data.get("reps") or 1),
        limit=overrides.get("limit") or data.get("limit"),
        agent=agent,
    )


def load_dataset(path: str | Path) -> Dataset:
    """Read dataset.yaml and cases.jsonl. Evaluators are built once and categories checked against the
    dataset's declared list, so typos fail before any agent call."""
    path = Path(path)
    if not (path / "dataset.yaml").exists():
        raise HarnessError(f"{path}: no dataset.yaml found")
    meta = yaml.safe_load((path / "dataset.yaml").read_text()) or {}
    task = str(meta.get("task") or "").strip()
    if not task:
        raise HarnessError(f"{path}/dataset.yaml: describe what the agent is asked to do under 'task'")
    categories = list(meta.get("categories") or [])
    if not categories:
        raise HarnessError(f"{path}/dataset.yaml: list the case categories under 'categories'")
    cases = []
    for n, line in enumerate((path / "cases.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case(**json.loads(line))
            if case.category not in categories:
                raise HarnessError(f"category {case.category!r} is not one of {categories}")
            for spec in case.evaluators:
                evaluators.build(spec)
        except (TypeError, KeyError, json.JSONDecodeError, HarnessError) as exc:
            raise HarnessError(f"{path}/cases.jsonl line {n}: {exc}") from exc
        cases.append(case)
    return Dataset(name=meta.get("name") or path.name, version=str(meta.get("version", "0")), path=path,
                   task=task, categories=categories, cases=cases)
