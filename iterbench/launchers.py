"""Write ready-to-use Windows launchers (launchers/start-<model>-tuned.bat) for the tuned settings."""
from __future__ import annotations

from pathlib import Path

from .config import Config, describe_diff, flags_to_args
from .tuner import load_state, run_by_id
from .util import ROOT

LAUNCHER_DIR = ROOT / "launchers"
DEFAULT_FIT_TARGET = 1024    # MiB; llama-server's default VRAM margin for --fit


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or not arg) else arg


def margin_costs(ms: dict, fit: int) -> dict[int, float]:
    """How much longer an agent turn took in the speed probes with each bigger VRAM margin, in percent,
    comparing probes that differ from the chosen one only in --fit-target (none = llama.cpp's default)."""
    results = [p for p in (ms.get("probe") or {}).get("results", []) if not p.get("error") and p.get("turn_s")]
    chosen = next((p for p in results if p["flags"].get("fit-target") == fit), None)
    if chosen is None:
        return {}
    rest = {k: v for k, v in chosen["flags"].items() if k != "fit-target"}
    costs = {}
    for p in results:
        margin = int(p["flags"].get("fit-target", DEFAULT_FIT_TARGET))
        if margin > fit and {k: v for k, v in p["flags"].items() if k != "fit-target"} == rest:
            costs[margin] = 100 * (p["turn_s"] / chosen["turn_s"] - 1)
    return costs


def write_launchers(cfg: Config, port: int = 8080) -> list[Path]:
    st = load_state(cfg)
    LAUNCHER_DIR.mkdir(exist_ok=True)
    written = []
    for name, ms in st["models"].items():
        inc = run_by_id(ms, ms["incumbent"])
        model = cfg.models.get(name)
        if not inc or not model:
            continue
        args = flags_to_args(inc["flags"])
        pairs, i = [], 0
        while i < len(args):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                pairs.append(f"{args[i]} {_quote(args[i + 1])}")
                i += 2
            else:
                pairs.append(args[i])
                i += 1
        lines = [" ".join(pairs[j:j + 6]) for j in range(0, len(pairs), 6)]
        body = " ^\n  ".join(lines)
        fit = inc["flags"].get("fit-target")
        margin_note = ""
        if fit is not None and int(fit) < DEFAULT_FIT_TARGET:
            costs = margin_costs(ms, int(fit))
            cost_text = ", ".join(f"{m} MiB {pct:.0f}% longer" for m, pct in sorted(costs.items()))
            margin_note = (f"rem --fit-target {fit} leaves only {fit} MiB of VRAM free (llama.cpp's default: "
                           f"{DEFAULT_FIT_TARGET} MiB), measured with nothing else\r\n"
                           f"rem on the GPU. If the server stops with \"CUDA error: out of memory\", or other "
                           f"programs need the GPU, use a bigger margin.\r\n"
                           + (f"rem Agent turns in the speed probes with a bigger margin: {cost_text}.\r\n"
                              if cost_text else ""))
        text = (f"@echo off\r\ntitle llama-server: {name} (tuned)\r\n"
                f"rem Tuned by Iterated Testing (see results/REPORT.md).\r\n"
                f"rem Changes vs. the original launcher: {describe_diff(model.baseline, inc['flags'])}\r\n"
                + margin_note +
                f"rem OpenCode model id: llamacpp/{name}\r\n"
                f"set LLAMA={cfg.server_exe}\r\nset MODEL={model.file}\r\n"
                f"\"%LLAMA%\" -m \"%MODEL%\" --alias {name} --port {port} --metrics ^\r\n  "
                + body.replace("\n", "\r\n") + "\r\npause\r\n")
        path = LAUNCHER_DIR / f"start-{name}-tuned.bat"
        path.write_bytes(text.encode("utf-8"))
        written.append(path)
    return written
