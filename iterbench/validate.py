"""Check the task sets themselves before any model sees them.

* codegen/bugfix: the reference solution passes every hidden check; for bug fixes the buggy code fails.
* toolcall: tool names exist and the example answer is accepted by the grader.
* longctx: contexts generate deterministically and have enough questions.
* agent: the visible tests fail on the untouched repo and pass (with the hidden ones) after the
  reference fix in solution/ is applied.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .graders import codegen, longctx, toolcall
from .graders.agent import _run_tests
from .tasks import load_tasks
from .util import log


def _agent_check(task_dir: Path, tmp: Path) -> list[str]:
    problems = []
    repo = tmp / "repo"
    shutil.copytree(task_dir / "repo", repo)
    _, before = _run_tests(repo, ["tests"])
    if before and before.get("ok"):
        problems.append("visible tests already pass before any fix")
    if not (task_dir / "solution").is_dir():
        problems.append("missing solution/ folder")
    else:
        shutil.copytree(task_dir / "solution", repo, dirs_exist_ok=True)
        shutil.copytree(task_dir / "hidden", repo / "tests_hidden", dirs_exist_ok=True)
        out, after = _run_tests(repo, ["tests", "tests_hidden"])
        if not (after and after.get("ok")):
            problems.append("reference solution does not pass: " + out[-800:])
        elif after["dirs"][-1]["tests"] == 0:
            problems.append("hidden tests were not found")
    shutil.rmtree(repo, ignore_errors=True)
    return problems


def validate(splits=("tune", "final")) -> bool:
    ok = True
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="iterbench_validate_") as tmpd:
        tmp = Path(tmpd)
        for split in splits:
            tasks = load_tasks(split)
            counts: dict[str, int] = {}
            weight = {"all": 0.0, "lite": 0.0}
            for t in tasks:
                counts[t.category] = counts.get(t.category, 0) + 1
                weight["all"] += t.weight
                weight["lite"] += t.weight if t.lite else 0.0
                problems: list[str] = []
                if t.id in seen:
                    problems.append("duplicate id")
                seen.add(t.id)
                if t.category in ("codegen", "bugfix"):
                    ref = t.data.get("reference")
                    if not ref:
                        problems.append("no reference solution")
                    else:
                        g = codegen.grade_code(ref, t.data, tmp / "w")
                        if not g["passed"]:
                            problems.append(f"reference fails ({g.get('tests_passed')}/{g.get('tests_total')}): {g['detail']}")
                        elif g["tests_total"] < 5:
                            problems.append(f"only {g['tests_total']} checks")
                    if t.category == "bugfix":
                        g = codegen.grade_code(t.data["code"], t.data, tmp / "w")
                        if g["passed"]:
                            problems.append("the buggy code passes every check")
                    # The prompt must name the entry point(s) so the model knows what to write.
                    text = t.data.get("prompt", "") + t.data.get("code", "")
                    for e in codegen.entries_of(t.data):
                        if e not in text:
                            problems.append(f"entry {e} not mentioned in the prompt")
                elif t.category == "toolcall":
                    for name in t.data.get("tools", []):
                        if name not in toolcall.TOOLS:
                            problems.append(f"unknown tool {name}")
                    try:
                        toolcall.build_request(t.data)
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"bad messages: {e!r}")
                    ex = t.data.get("example")
                    if not ex:
                        problems.append("no example")
                    elif "answer" in ex:
                        g = toolcall.grade(t.data, ex["answer"], [])
                        if not g["passed"]:
                            problems.append("example answer rejected: " + g["detail"])
                    else:
                        call = {"name": ex["name"], "args": ex["args"], "arguments": json.dumps(ex["args"])}
                        g = toolcall.grade(t.data, "", [call])
                        if not g["passed"]:
                            problems.append("example call rejected: " + g["detail"])
                elif t.category == "longctx":
                    gen = longctx.generate(int(t.data["seed"]), int(t.data["target_tokens"]))
                    again = longctx.generate(int(t.data["seed"]), int(t.data["target_tokens"]))
                    if gen["code"] != again["code"] or gen["questions"] != again["questions"]:
                        problems.append("generation is not deterministic")
                    if len(gen["questions"]) <= t.data["question_index"]:
                        problems.append("not enough questions")
                    else:
                        q = gen["questions"][t.data["question_index"]]
                        if q["depth"] < 2:
                            problems.append(f"question depth only {q['depth']}")
                        if t.data["question_index"] == 0:
                            log(f"  {t.id}: ~{len(gen['code']) / longctx.CHARS_PER_TOKEN:,.0f} tokens, "
                                f"{len(gen['modules'])} modules")
                elif t.category == "agent":
                    problems += _agent_check(Path(t.data["dir"]), tmp)
                if problems:
                    ok = False
                    for p in problems:
                        log(f"PROBLEM {t.id}: {p}")
            log(f"{split}: {len(tasks)} tasks {counts}; weight {weight['all']:g} (lite {weight['lite']:g})")
    log("all tasks valid" if ok else "some tasks have problems")
    return ok
