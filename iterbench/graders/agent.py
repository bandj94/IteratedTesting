"""Agent tasks: a multi-turn tool loop that fixes a small repository until its tests pass.

The model gets four tools (list_files, read_file, write_file, run_tests) that only work inside a
throwaway copy of the task's repo. When it stops calling tools (or runs out of turns/tokens), the
original tests are restored (so editing tests cannot help) and the visible plus hidden tests run.
The task passes only if all of them pass.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..client import ChatResult, Client
from ..sandbox import run_guarded

SYSTEM = ("You are an autonomous coding agent working inside a small Python repository. Use the tools to "
          "inspect files, change the code and run the tests. Keep working until the task is done and the "
          "tests pass, then reply with a short summary and no tool call. Never modify files under tests/.")

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List all files in the repository.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Return the full text of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with the complete new content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "content": {"type": "string", "description": "The complete new file content"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite (python -m unittest) and return its output.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

MAX_READ = 20000
MAX_TEST_OUTPUT = 4000


def _resolve(root: Path, rel) -> Path | None:
    if not isinstance(rel, str) or not rel.strip():
        return None
    rel = rel.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("/") or ":" in rel:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _list_files(root: Path) -> str:
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    return "\n".join(files) or "(empty)"


def _run_tests(root: Path, dirs: list[str]) -> tuple[str, dict | None]:
    r = run_guarded(root, ["unittest", *dirs], timeout=60)
    text = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    text = text.replace("@@RESULT@@", "")
    if r.timed_out:
        text += "\n[tests timed out after 60 s]"
    if len(text) > MAX_TEST_OUTPUT:
        text = "...(earlier output cut)...\n" + text[-MAX_TEST_OUTPUT:]
    return text, r.result


def execute(root: Path, call: dict) -> str:
    name, args = call.get("name"), call.get("args")
    if args is None:
        return "Error: the tool arguments were not valid JSON."
    if name == "list_files":
        return _list_files(root)
    if name == "read_file":
        path = _resolve(root, args.get("path"))
        if path is None or not path.is_file():
            return f"Error: no such file: {args.get('path')!r}"
        text = path.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= MAX_READ else text[:MAX_READ] + "\n...(truncated)"
    if name == "write_file":
        path = _resolve(root, args.get("path"))
        content = args.get("content")
        if path is None:
            return f"Error: invalid path {args.get('path')!r}"
        if not isinstance(content, str):
            return "Error: content must be a string."
        if path.relative_to(root.resolve()).parts[:1] == ("tests",):
            return "Error: files under tests/ are read-only."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path.relative_to(root.resolve()).as_posix()}."
    if name == "run_tests":
        return _run_tests(root, ["tests"])[0]
    return f"Error: unknown tool {name!r}."


def run(client: Client, data: dict, workdir: Path, *, seed: int, max_tokens: int, tokenize) -> dict:
    """Run the agent loop for one task. Returns the grade plus per-request metrics."""
    task_dir = Path(data["dir"])
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    repo = workdir / "repo"
    shutil.copytree(task_dir / "repo", repo)
    max_turns = int(data.get("max_turns", 14))
    token_budget = int(data.get("token_budget", 40000))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": data["prompt"].strip()}]
    requests, transcript = [], []
    generated, reasoning_tokens, stop = 0, 0, "max_turns"
    for turn in range(max_turns):
        res: ChatResult = client.chat(messages, tools=TOOLS, max_tokens=max_tokens, seed=seed + turn)
        requests.append(res.metrics())
        if res.error:
            stop = "request_error: " + res.error[:200]
            break
        generated += res.predicted_n
        reasoning_tokens += tokenize(res.reasoning) if res.reasoning else 0
        msg: dict = {"role": "assistant", "content": res.content or ""}
        if res.reasoning:
            msg["reasoning_content"] = res.reasoning
        if res.raw_tool_calls:
            msg["tool_calls"] = res.raw_tool_calls
        messages.append(msg)
        transcript.append({"turn": turn + 1, "content": res.content[-1500:],
                           "calls": [{"name": c["name"], "arguments": (c["arguments"] or "")[:1500]}
                                     for c in res.tool_calls]})
        if not res.tool_calls:
            stop = "finished" if res.finish_reason != "length" else "cut_off"
            break
        for i, call in enumerate(res.tool_calls):
            output = execute(repo, call)
            call_id = call.get("id") or f"call_{turn}_{i}"
            if not call.get("id") and i < len(msg.get("tool_calls", [])):
                msg["tool_calls"][i]["id"] = call_id
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            transcript[-1].setdefault("results", []).append(output[:800])
        if generated >= token_budget:
            stop = "token_budget"
            break

    # Restore the original tests, add the hidden ones, and grade.
    shutil.rmtree(repo / "tests", ignore_errors=True)
    shutil.copytree(task_dir / "repo" / "tests", repo / "tests")
    if (task_dir / "hidden").is_dir():
        shutil.copytree(task_dir / "hidden", repo / "tests_hidden", dirs_exist_ok=True)
    output, result = _run_tests(repo, ["tests", "tests_hidden"])
    result = result or {}
    tests = int(result.get("tests", 0))
    bad = int(result.get("failures", 0)) + int(result.get("errors", 0))
    passed = bool(result.get("ok")) and tests > 0
    changed = {}
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if rel.startswith(("tests/", "tests_hidden/")):
            continue
        orig = task_dir / "repo" / rel
        text = p.read_text(encoding="utf-8", errors="replace")
        if not orig.exists() or orig.read_text(encoding="utf-8", errors="replace") != text:
            changed[rel] = text[:6000]
    shutil.rmtree(workdir, ignore_errors=True)
    return {"passed": passed, "partial": round((tests - bad) / tests, 3) if tests else 0.0,
            "detail": f"stop={stop}, turns={len(requests)}, tests {tests - bad}/{tests} passed"
                      + ("" if passed else " | " + output[-600:].replace("\n", " ")),
            "requests": requests, "reasoning_tokens": reasoning_tokens, "stop": stop,
            "transcript": transcript, "changed_files": changed}


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
