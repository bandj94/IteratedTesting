"""Per-task pass/fail matrix across runs, to see which tasks decide a score and how much runs vary.

    python scripts/task_matrix.py                 # tuning runs of every model (columns = iterations)
    python scripts/task_matrix.py ornith-9b       # one model
    python scripts/task_matrix.py --final         # held-out runs (columns = model/settings)

Cell codes: . pass, x fail, T fail with an answer cut off at the output cap, blank = not in that suite.
For tuning runs only tasks whose result differs between runs are listed (plus per-category totals);
for held-out runs every task is listed, followed by the failure details. Reads runs/ (not in git).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def load_items(items_dir: Path) -> dict[str, dict]:
    res = {}
    for p in items_dir.glob("*.json"):
        it = json.loads(p.read_text(encoding="utf-8"))
        it["_cut"] = any(q.get("finish_reason") == "length" for q in it.get("requests", []))
        res[it["id"]] = it
    return res


def cell(res: dict, tid: str) -> str:
    if tid not in res:
        return " "
    it = res[tid]
    return "." if it["passed"] else ("T" if it["_cut"] else "x")


def tuning(state: dict, only: set[str]) -> None:
    for model, ms in state["models"].items():
        if only and model not in only:
            continue
        cols = [(r, load_items(RUNS / model / r["run_id"] / "tune" / "items")) for r in ms["runs"] if r.get("ok")]
        print(f"=== {model} (pick: {ms.get('incumbent')})")
        print("   runs: " + "  ".join(f"it{r['iteration']}={r['score']:.1f}" for r, _ in cols))
        cats = sorted({it["category"] for _, res in cols for it in res.values()})
        for cat in cats:
            row = []
            for _, res in cols:
                got = [it for it in res.values() if it["category"] == cat]
                row.append(f"{sum(it['passed'] for it in got):2d}/{len(got):2d}")
            print(f"   {cat:9s} " + "  ".join(row))
        print("   tasks that changed (columns = iterations in order):")
        for tid in sorted({t for _, res in cols for t in res}):
            cells = [cell(res, tid) for _, res in cols]
            if len(set(cells)) > 1:
                print(f"   {tid:16s} {' '.join(cells)}")
        print()


def final(state: dict, only: set[str]) -> None:
    cols, meta = [], {}
    for model, fin in state.get("final", {}).items():
        if only and model not in only:
            continue
        for kind, f in fin.items():
            if f.get("run_id"):
                res = load_items(RUNS / model / f["run_id"] / "final" / "items")
                cols.append((f"{model}/{kind}", res))
                for tid, it in res.items():
                    meta[tid] = (it["category"], it.get("weight"))
    print("columns: " + ", ".join(f"{i + 1}={name}" for i, (name, _) in enumerate(cols)))
    for tid in sorted(meta, key=lambda t: (meta[t][0], t)):
        print(f"{meta[tid][0]:9s} {tid:18s} w={meta[tid][1]}  {' '.join(cell(res, tid) for _, res in cols)}")
    print("\nfailures:")
    for tid in sorted(meta):
        fails = [(name, res[tid]) for name, res in cols if tid in res and not res[tid]["passed"]]
        if fails:
            print(tid)
            for name, it in fails:
                print(f"    {name}: {str(it.get('detail') or '')[:110]}")


def main() -> None:
    args = sys.argv[1:]
    held_out = "--final" in args
    only = {a for a in args if a != "--final"}
    state = json.loads((RUNS / "tuner_state.json").read_text(encoding="utf-8"))
    (final if held_out else tuning)(state, only)


if __name__ == "__main__":
    main()
