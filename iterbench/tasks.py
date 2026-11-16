"""Load benchmark tasks from tasks/<split>/.

Splits: "tune" (used while tuning) and "final" (held out; only used by the final test).
Categories, in run order: toolcall, codegen, bugfix, longctx, agent.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .util import ROOT

TASKS_DIR = ROOT / "tasks"
CATEGORIES = ("toolcall", "codegen", "bugfix", "longctx", "agent")
SPLITS = ("tune", "final")


@dataclass
class Task:
    id: str              # e.g. "tune/cg01"
    category: str
    split: str
    weight: float
    lite: bool
    data: dict = field(default_factory=dict)


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_tasks(split: str, tasks_dir: Path | None = None) -> list[Task]:
    base = Path(tasks_dir or TASKS_DIR) / split
    tasks: list[Task] = []
    for cat in ("toolcall", "codegen", "bugfix"):
        f = base / f"{cat}.toml"
        if not f.exists():
            continue
        for t in _read_toml(f).get("task", []):
            tasks.append(Task(f"{split}/{t['id']}", cat, split, float(t.get("weight", 1.0)),
                              bool(t.get("lite", False)), t))
    f = base / "longctx.toml"
    if f.exists():
        for ctx in _read_toml(f).get("context", []):
            for q in range(int(ctx.get("questions", 2))):
                data = dict(ctx, question_index=q)
                tasks.append(Task(f"{split}/{ctx['id']}-q{q + 1}", "longctx", split,
                                  float(ctx.get("weight", 1.0)), bool(ctx.get("lite", False)), data))
    agent_dir = base / "agent"
    if agent_dir.is_dir():
        for d in sorted(p for p in agent_dir.iterdir() if (p / "task.toml").exists()):
            meta = _read_toml(d / "task.toml")
            meta["dir"] = str(d)
            tasks.append(Task(f"{split}/{d.name}", "agent", split, float(meta.get("weight", 2.0)),
                              bool(meta.get("lite", False)), meta))
    order = {c: i for i, c in enumerate(CATEGORIES)}
    tasks.sort(key=lambda t: (order[t.category], t.id))
    return tasks


def select(tasks: list[Task], suite: str) -> list[Task]:
    if suite == "lite":
        return [t for t in tasks if t.lite]
    return list(tasks)
