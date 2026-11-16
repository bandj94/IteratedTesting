"""Command line: python -m iterbench <command>."""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from .config import describe, describe_diff, load_config
from .util import RUNS_DIR, log, write_json


def _parse_value(text: str):
    if text == "":
        return None
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _apply_sets(flags: dict, sets: list[str]) -> dict:
    flags = dict(flags)
    for s in sets or []:
        key, _, value = s.partition("=")
        v = _parse_value(value)
        if v is None:
            flags.pop(key, None)
        else:
            flags[key] = v
    return flags


def _flags_for(cfg, model_name: str, which: str) -> dict:
    model = cfg.models[model_name]
    if which == "baseline":
        return dict(model.baseline)
    from .tuner import load_state, run_by_id
    ms = load_state(cfg)["models"][model_name]
    inc = run_by_id(ms, ms["incumbent"])
    if not inc:
        raise SystemExit(f"{model_name} has no tuned pick yet")
    return dict(inc["flags"])


def cmd_doctor(args, cfg) -> int:
    from .sandbox import run_guarded
    from .server import find_llama_servers, port_in_use
    from .telemetry import cpu_name, gpu_name, gpu_query, ram_available_mib

    ok = True
    print(f"iterbench {__version__}, Python {platform.python_version()} on {platform.platform()}")
    exe = cfg.server_exe
    version = "?"
    if exe.exists():
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
        text = (out.stdout + out.stderr).strip().splitlines()
        version = next((ln.split("version:")[-1].strip() for ln in text if "version" in ln), "?")
        print(f"[ok]   llama-server: {version}")
    else:
        print(f"[FAIL] llama-server not found at {exe}")
        ok = False
    for name, m in cfg.models.items():
        if m.file.exists():
            print(f"[ok]   {name}: {m.file.name} ({m.file.stat().st_size / 2**30:.1f} GiB), suite={m.suite}")
        else:
            print(f"[FAIL] {name}: missing {m.file}")
            ok = False
    others = find_llama_servers()
    print(("[WARN] other llama-server processes running: " + str(others)) if others else "[ok]   no other llama-server running")
    print(f"[{'WARN' if port_in_use(cfg.host, cfg.port) else 'ok'}]   port {cfg.port}")
    g = gpu_query()
    print(f"[ok]   GPU {gpu_name()}: {g.get('memory_used_mib') if g else '?'} MiB in use" if g else "[WARN] no NVIDIA GPU found")
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "probe.py"
        script.write_text("import subprocess\ntry:\n    subprocess.run(['cmd', '/c', 'echo hi'])\n    print('@@RESULT@@{\"blocked\": false}')\n"
                          "except PermissionError:\n    print('@@RESULT@@{\"blocked\": true}')\n", encoding="utf-8")
        r = run_guarded(Path(d), ["script", "probe.py"], timeout=20)
        blocked = bool(r.result and r.result.get("blocked"))
        print(("[ok]   sandbox guard blocks process creation") if blocked else
              f"[FAIL] sandbox guard did not block process creation: {r.stdout[-300:]} {r.stderr[-300:]}")
        ok = ok and blocked
    ram_total = None
    if sys.platform == "win32":
        import ctypes
        kb = ctypes.c_ulonglong(0)
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb)):
            ram_total = round(kb.value / 2**20)
    env = {"cpu": cpu_name(), "gpu": gpu_name(), "ram_gb": ram_total,
           "ram_available_mib": round(ram_available_mib() or 0), "llama_cpp": version,
           "python": platform.python_version(), "os": platform.platform(), "iterbench": __version__}
    write_json(RUNS_DIR / "environment.json", env)
    print(f"environment: {env}")
    return 0 if ok else 1


def cmd_validate(args, cfg) -> int:
    from .validate import validate
    return 0 if validate(tuple(args.splits)) else 1


def cmd_bench(args, cfg) -> int:
    from .bench import run_benchmark
    from .util import stamp
    flags = _apply_sets(_flags_for(cfg, args.model, args.config), args.set)
    model = cfg.models[args.model]
    run_dir = RUNS_DIR / "adhoc" / f"{stamp()}_{args.model}_{args.config}"
    rec = run_benchmark(cfg, model, flags, splits=tuple(args.split), suite=args.suite, label=f"adhoc-{args.config}",
                        run_dir=run_dir, limit=args.limit, only=args.only, seed_offset=args.seed_offset)
    for split, s in rec["splits"].items():
        print(f"{split}: score {s['score']}% ({s['n_passed']}/{s['n_items']}), {s['wall_s'] / 60:.1f} min, "
              f"gen {s['gen_tps']} tok/s, prefill {s['pp_tps']} tok/s, thinking {s['reasoning_tokens']} tokens, "
              f"cut off {s['truncated']}")
        for cat, c in s["by_category"].items():
            print(f"   {cat:<9} {c['passed']}/{c['total']}  ({c['wall_s']:.0f} s)")
    print(f"details: {run_dir}")
    return 1 if rec.get("error") else 0


def cmd_probe(args, cfg) -> int:
    from .probe import run_probe
    from .util import stamp
    flags = _apply_sets(_flags_for(cfg, args.model, args.config), args.set)
    r = run_probe(cfg, cfg.models[args.model], flags, RUNS_DIR / "adhoc" / f"{stamp()}_{args.model}_probe.log",
                  label=" ".join(args.set or []) or args.config)
    print(r)
    return 1 if r.get("error") else 0


def cmd_tune(args, cfg) -> int:
    from .tuner import tune
    models = args.models or list(cfg.models)
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations
    tune(cfg, models, iterations=args.iterations)
    return 0


def cmd_final(args, cfg) -> int:
    from .tuner import run_final
    run_final(cfg, args.models or list(cfg.models), seeds=args.seeds)
    return 0


def cmd_report(args, cfg) -> int:
    from .report import build
    print(f"wrote {build(cfg)}")
    return 0


def cmd_launchers(args, cfg) -> int:
    from .launchers import write_launchers
    for p in write_launchers(cfg, port=args.port):
        print(f"wrote {p}")
    return 0


def cmd_status(args, cfg) -> int:
    from .tuner import load_state, run_by_id
    st = load_state(cfg)
    print(f"iteration {st['iteration']} of at most {cfg.max_iterations}")
    for name, ms in st["models"].items():
        inc = run_by_id(ms, ms["incumbent"])
        print(f"\n{name}: {ms['status']}, {len(ms['runs'])} runs")
        for r in ms["runs"]:
            mark = "*" if inc and r["run_id"] == inc["run_id"] else " "
            score = f"{r['score']:5.1f}%" if r.get("score") is not None else "  n/a "
            wall = f"{r['wall_s'] / 60:5.1f} min" if r.get("wall_s") else "    -    "
            print(f" {mark} it{r['iteration']:>2} {score} {wall}  {r.get('change', r['label'])[:60]:<60} "
                  f"{(r.get('decision') or '')[:60]}")
        if inc:
            print(f"   pick vs original: {describe_diff(cfg.models[name].baseline, inc['flags'])}")
            print(f"   full flags: {describe(inc['flags'])}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    p = argparse.ArgumentParser(prog="iterbench", description="Benchmark and tune local llama.cpp models.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check paths, GPU, sandbox; save environment info")
    v = sub.add_parser("validate", help="check the task sets (reference solutions, generators, repos)")
    v.add_argument("--splits", nargs="*", default=["tune", "final"])
    b = sub.add_parser("bench", help="one ad-hoc benchmark (not recorded by the tuner)")
    b.add_argument("--model", required=True)
    b.add_argument("--config", default="baseline", choices=["baseline", "pick"])
    b.add_argument("--split", nargs="*", default=["tune"])
    b.add_argument("--suite", choices=["full", "lite"])
    b.add_argument("--limit", type=int)
    b.add_argument("--only", nargs="*", help="task id suffixes or categories")
    b.add_argument("--set", nargs="*", default=[], help="override flags, e.g. reasoning=off threads=6")
    b.add_argument("--seed-offset", type=int, default=0)
    pr = sub.add_parser("probe", help="one speed probe")
    pr.add_argument("--model", required=True)
    pr.add_argument("--config", default="baseline", choices=["baseline", "pick"])
    pr.add_argument("--set", nargs="*", default=[])
    t = sub.add_parser("tune", help="run (or resume) the tuning iterations")
    t.add_argument("--models", nargs="*")
    t.add_argument("--iterations", type=int, help="stop after this many iterations in this call")
    t.add_argument("--max-iterations", type=int, help="override the global cap from models.toml")
    f = sub.add_parser("final", help="held-out test: original vs tuned settings")
    f.add_argument("--models", nargs="*")
    f.add_argument("--seeds", type=int, default=1)
    sub.add_parser("report", help="write results/REPORT.md, iterations.csv, tradeoff.svg")
    la = sub.add_parser("launchers", help="write launchers/start-<model>-tuned.bat")
    la.add_argument("--port", type=int, default=8080)
    sub.add_parser("status", help="show the tuning state")
    args = p.parse_args(argv)
    if args.cmd in ("bench", "probe", "tune", "final") and sys.platform == "win32":
        # Keep Windows from idle-sleeping while this process runs (released automatically on exit).
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
    cfg = load_config()
    handler = {"doctor": cmd_doctor, "validate": cmd_validate, "bench": cmd_bench, "probe": cmd_probe,
               "tune": cmd_tune, "final": cmd_final, "report": cmd_report, "launchers": cmd_launchers,
               "status": cmd_status}[args.cmd]
    return handler(args, cfg)
