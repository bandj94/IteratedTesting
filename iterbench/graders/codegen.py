"""Code generation and bug-fix tasks: the model writes a function; hidden unit tests grade it.

A task passes only if every hidden check passes (strict pass@1). The fraction of checks passed is
kept as `partial` for diagnostics. The model's code runs in a separate, guarded process.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from ..sandbox import run_guarded

SYSTEM = "You are an expert Python programmer. You write correct, complete code."
CODEGEN_SUFFIX = ("\n\nReply with the complete solution (the function plus any imports or helper functions "
                  "it needs) in a single ```python code block. Use only the Python standard library.")
BUGFIX_TEMPLATE = ("The Python function below has a problem.\n\n```python\n{code}\n```\n\n{symptom}"
                   "Fix it so it does exactly what its docstring says. Reply with the complete corrected "
                   "function (plus any imports it needs) in a single ```python code block.")

PRELUDE = r'''
import json as __json, math as __math, sys as __sys
__results = {"passed": 0, "total": 0, "failures": []}

def __same(got, exp):
    if isinstance(got, bool) != isinstance(exp, bool) and (isinstance(got, float) or isinstance(exp, float)):
        return False
    if isinstance(got, float) or isinstance(exp, float):
        if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
            return __math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9)
        return False
    if isinstance(exp, tuple):
        return isinstance(got, tuple) and len(got) == len(exp) and all(__same(a, b) for a, b in zip(got, exp))
    if isinstance(exp, list):
        return isinstance(got, list) and len(got) == len(exp) and all(__same(a, b) for a, b in zip(got, exp))
    if isinstance(exp, dict):
        return isinstance(got, dict) and got.keys() == exp.keys() and all(__same(got[k], exp[k]) for k in exp)
    try:
        return bool(got == exp)
    except Exception:
        return False

def __record(ok, label, detail):
    __results["total"] += 1
    if ok:
        __results["passed"] += 1
    elif len(__results["failures"]) < 8:
        __results["failures"].append((str(label) + ": " + str(detail))[:300])

def check(fn, expected, label=None):
    label = label or "check %d" % (__results["total"] + 1)
    try:
        got = fn()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        __record(False, label, "raised %r" % (e,))
        return
    __record(__same(got, expected), label, "got %r, expected %r" % (got, expected))

def check_raises(fn, exc, label=None):
    label = label or "check %d" % (__results["total"] + 1)
    try:
        got = fn()
    except exc:
        __record(True, label, "")
        return
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        __record(False, label, "raised %r, expected %s" % (e, getattr(exc, "__name__", exc)))
        return
    __record(False, label, "returned %r, expected %s" % (got, getattr(exc, "__name__", exc)))

def check_true(fn, label=None):
    check(lambda: bool(fn()), True, label)

__src = open("solution.py", encoding="utf-8").read()
__ns = {"__name__": "solution"}
try:
    exec(compile(__src, "solution.py", "exec"), __ns)
except BaseException as __e:
    print("@@RESULT@@" + __json.dumps({"passed": 0, "total": 0,
                                        "failures": ["solution failed to load: %r" % (__e,)]}), flush=True)
    raise SystemExit(0)
for __k, __v in list(__ns.items()):
    if not __k.startswith("__"):
        globals()[__k] = __v
'''

EPILOGUE = '\nprint("@@RESULT@@" + __json.dumps(__results), flush=True)\n'

_FENCED = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
_OPEN = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*)", re.S | re.I)


def entries_of(task_data: dict) -> list[str]:
    entry = task_data["entry"]
    return [entry] if isinstance(entry, str) else list(entry)


def build_messages(category: str, data: dict) -> list[dict]:
    if category == "bugfix":
        symptom = data.get("symptom", "").strip()
        user = BUGFIX_TEMPLATE.format(code=data["code"].strip("\n"), symptom=(symptom + "\n\n") if symptom else "")
    else:
        user = data["prompt"].strip() + CODEGEN_SUFFIX
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_code(content: str, entries: list[str]) -> str | None:
    """Pick the code to test from a model answer."""
    content = content or ""
    blocks = _FENCED.findall(content)
    if not blocks:
        m = _OPEN.search(content)            # an unterminated block (answer cut off)
        if m:
            blocks = [m.group(1)]
        elif any(re.search(rf"^\s*def\s+{re.escape(e)}\s*\(", content, re.M) for e in entries):
            return content                   # bare code without fences
        else:
            return None

    def defines(block: str, name: str) -> bool:
        return re.search(rf"\bdef\s+{re.escape(name)}\s*\(", block) is not None

    full = [b for b in blocks if all(defines(b, e) for e in entries)]
    if full:
        return max(full, key=len)
    some = [b for b in blocks if any(defines(b, e) for e in entries)]
    if some:
        return "\n\n".join(some)
    return blocks[-1]


def build_program(tests: str) -> str:
    return PRELUDE + "\n# ---- tests ----\n" + tests.strip("\n") + "\n" + EPILOGUE


def grade_code(code: str | None, data: dict, workdir: Path) -> dict:
    if code is None:
        return {"passed": False, "partial": 0.0, "detail": "no code found in the answer"}
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "check.py").write_text(build_program(data["tests"]), encoding="utf-8")
        r = run_guarded(workdir, ["script", "check.py"], timeout=float(data.get("timeout", 20)))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if r.timed_out:
        return {"passed": False, "partial": 0.0, "detail": "tests timed out"}
    res = r.result
    if not res:
        return {"passed": False, "partial": 0.0, "detail": "test program crashed: " + (r.stderr or r.stdout)[-400:]}
    total, passed = int(res.get("total", 0)), int(res.get("passed", 0))
    return {"passed": total > 0 and passed == total, "partial": round(passed / total, 3) if total else 0.0,
            "tests_passed": passed, "tests_total": total, "detail": " | ".join(res.get("failures", [])[:4])}


def grade(category: str, data: dict, content: str, workdir: Path) -> dict:
    code = extract_code(content, entries_of(data))
    out = grade_code(code, data, workdir)
    out["code"] = code
    return out
