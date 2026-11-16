"""Run a list of tasks against a running server and summarize accuracy and speed."""
from __future__ import annotations

import time
from pathlib import Path

from .client import Client, ClientError
from .graders import agent, codegen, longctx, toolcall
from .tasks import Task
from .util import log, stable_seed, wilson, write_json

# Output-token caps per request (thinking + answer). Scaled per model by max_tokens_scale.
MAX_TOKENS = {"toolcall": 4096, "codegen": 8192, "bugfix": 8192, "longctx": 6144, "agent": 6144}


def run_task(client: Client, task: Task, work: Path, scale: float, seed_offset: int = 0) -> dict:
    seed = stable_seed(task.id) + seed_offset
    max_tokens = max(512, int(MAX_TOKENS[task.category] * scale))
    tokenize = client.tokenize
    if task.category == "agent":
        return agent.run(client, task.data, work / task.id.replace("/", "_"), seed=seed,
                         max_tokens=max_tokens, tokenize=tokenize)
    if task.category in ("codegen", "bugfix"):
        res = client.chat(codegen.build_messages(task.category, task.data), max_tokens=max_tokens, seed=seed)
        g = ({"passed": False, "partial": 0.0, "detail": "request failed: " + res.error} if res.error else
             codegen.grade(task.category, task.data, res.content, work / task.id.replace("/", "_")))
        transcript = {"content": res.content[-6000:], "reasoning_tail": res.reasoning[-1500:]}
    elif task.category == "toolcall":
        messages, tools = toolcall.build_request(task.data)
        res = client.chat(messages, tools=tools, max_tokens=max_tokens, seed=seed)
        g = ({"passed": False, "partial": 0.0, "detail": "request failed: " + res.error} if res.error else
             toolcall.grade(task.data, res.content, res.tool_calls))
        transcript = {"content": res.content[-3000:], "reasoning_tail": res.reasoning[-1000:],
                      "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in res.tool_calls]}
    elif task.category == "longctx":
        prompt, question = longctx.build(task.data)
        res = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, seed=seed)
        g = ({"passed": False, "partial": 0.0, "detail": "request failed: " + res.error} if res.error else
             longctx.grade(question, res.content))
        transcript = {"content": res.content[-3000:], "reasoning_tail": res.reasoning[-1500:],
                      "question": question}
    else:
        raise ValueError(task.category)
    g["requests"] = [res.metrics()]
    g["reasoning_tokens"] = tokenize(res.reasoning) if res.reasoning else 0
    g["transcript"] = transcript
    return g


def summarize(items: list[dict], wall_s: float) -> dict:
    total_w = sum(it["weight"] for it in items)
    passed_w = sum(it["weight"] for it in items if it["passed"])
    reqs = [r for it in items for r in it["requests"]]
    gen_n = sum(r["predicted_n"] for r in reqs)
    gen_ms = sum(r["predicted_ms"] for r in reqs)
    pp_n = sum(r["prompt_n"] for r in reqs)
    pp_ms = sum(r["prompt_ms"] for r in reqs)
    long_reqs = [r for r in reqs if r["prompt_n"] >= 4000]
    lp_n = sum(r["prompt_n"] for r in long_reqs)
    lp_ms = sum(r["prompt_ms"] for r in long_reqs)
    by_cat: dict[str, dict] = {}
    for it in items:
        c = by_cat.setdefault(it["category"], {"passed": 0, "total": 0, "weight_passed": 0.0, "weight": 0.0,
                                               "wall_s": 0.0})
        c["total"] += 1
        c["passed"] += int(it["passed"])
        c["weight"] += it["weight"]
        c["weight_passed"] += it["weight"] if it["passed"] else 0.0
        c["wall_s"] += it["wall_s"]
    for c in by_cat.values():
        c["score"] = round(100 * c["weight_passed"] / c["weight"], 1) if c["weight"] else 0.0
        c["wall_s"] = round(c["wall_s"], 1)
    lo, hi = wilson(passed_w, total_w)
    return {
        "score": round(100 * passed_w / total_w, 2) if total_w else 0.0,
        "ci95": [round(lo, 1), round(hi, 1)],
        "passed_weight": passed_w, "total_weight": total_w,
        "n_items": len(items), "n_passed": sum(int(it["passed"]) for it in items),
        "wall_s": round(wall_s, 1),
        "gen_tokens": gen_n, "gen_tps": round(gen_n / (gen_ms / 1000), 2) if gen_ms else None,
        "prompt_tokens": pp_n, "pp_tps": round(pp_n / (pp_ms / 1000), 1) if pp_ms else None,
        "long_pp_tps": round(lp_n / (lp_ms / 1000), 1) if lp_ms else None,
        "reasoning_tokens": sum(it.get("reasoning_tokens", 0) for it in items),
        "truncated": sum(1 for r in reqs if r.get("finish_reason") == "length"),
        "request_errors": sum(1 for r in reqs if r.get("error")),
        "requests": len(reqs),
        "solved_per_hour": round(passed_w / (wall_s / 3600), 2) if wall_s else None,
        "by_category": by_cat,
    }


def run_suite(client: Client, tasks: list[Task], out_dir: Path, *, scale: float = 1.0,
              seed_offset: int = 0, label: str = "") -> dict:
    out_dir = Path(out_dir)
    work = out_dir / "work"
    items = []
    t_start = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        t0 = time.perf_counter()
        try:
            g = run_task(client, task, work, scale, seed_offset)
        except ClientError:
            raise
        except Exception as e:  # a grader bug must not kill an hours-long run; record it instead
            g = {"passed": False, "partial": 0.0, "detail": f"harness error: {e!r}", "requests": [],
                 "reasoning_tokens": 0, "transcript": {}}
        wall = time.perf_counter() - t0
        item = {"id": task.id, "category": task.category, "weight": task.weight, "passed": bool(g["passed"]),
                "partial": g.get("partial", 0.0), "wall_s": round(wall, 2), "detail": g.get("detail", "")[:800],
                "reasoning_tokens": g.get("reasoning_tokens", 0), "requests": g["requests"]}
        items.append(item)
        write_json(out_dir / "items" / (task.id.replace("/", "_") + ".json"), {**item, **{
            k: v for k, v in g.items() if k not in item}})
        gen = sum(r["predicted_n"] for r in g["requests"])
        log(f"{label}[{i:>2}/{len(tasks)}] {task.category:<8} {task.id:<22} {'PASS' if g['passed'] else 'fail'}"
            f"  {wall:6.1f}s  {gen:>6} tok  {g.get('detail', '')[:70]}")
    wall_s = time.perf_counter() - t_start
    try:
        work.rmdir()
    except OSError:
        pass
    return {"items": items, "summary": summarize(items, wall_s)}
