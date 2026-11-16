"""Load config/models.toml (plus config/local.toml for machine-specific paths)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import ROOT

CONFIG_DIR = ROOT / "config"


@dataclass
class ModelSpec:
    name: str
    file: Path
    suite: str                       # "full" or "lite"
    baseline: dict[str, Any]
    max_tokens_scale: float = 1.0
    max_iterations: int | None = None
    probe_tokens: tuple[int, int] = (128, 256)   # generated tokens: (at 16K depth, at short depth)
    dims: list[str] = field(default_factory=list)
    only: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class Config:
    server_exe: Path
    models_dir: Path
    host: str
    port: int
    startup_timeout_s: int
    max_iterations: int
    tolerance_pts: float
    speed_margin: float
    probe_margin: float
    speed_run_slack: float
    models: dict[str, ModelSpec]
    speed_dims: list[dict]
    dims: dict[str, dict]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(config_dir: Path | None = None) -> Config:
    config_dir = Path(config_dir or CONFIG_DIR)
    data = tomllib.loads((config_dir / "models.toml").read_text(encoding="utf-8"))
    local = config_dir / "local.toml"
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text(encoding="utf-8")))
    paths = data.get("paths", {})
    if "server_exe" not in paths or "models_dir" not in paths:
        raise SystemExit("Missing [paths] in config/local.toml (copy config/local.example.toml).")
    server, tuning, search = data.get("server", {}), data.get("tuning", {}), data.get("search", {})
    models_dir = Path(paths["models_dir"])
    default_dims = list(search.get("default_dims", ["reasoning", "speed", "sampling", "kv"]))
    models = {}
    for name, m in data.get("models", {}).items():
        msearch = m.get("search", {})
        models[name] = ModelSpec(
            name=name,
            file=models_dir / m["file"],
            suite=m.get("suite", "full"),
            baseline=dict(m["baseline"]),
            max_tokens_scale=float(m.get("max_tokens_scale", 1.0)),
            max_iterations=m.get("max_iterations"),
            probe_tokens=tuple(m.get("probe_tokens", (128, 256))),
            dims=list(msearch.get("dims", default_dims)),
            only={k: list(v) for k, v in msearch.get("only", {}).items()},
            notes=m.get("notes", ""),
        )
    return Config(
        server_exe=Path(paths["server_exe"]),
        models_dir=models_dir,
        host=server.get("host", "127.0.0.1"),
        port=int(server.get("port", 8090)),
        startup_timeout_s=int(server.get("startup_timeout_s", 900)),
        max_iterations=int(tuning.get("max_iterations", 20)),
        tolerance_pts=float(tuning.get("tolerance_pts", 4.0)),
        speed_margin=float(tuning.get("speed_margin", 0.05)),
        probe_margin=float(tuning.get("probe_margin", 0.02)),
        speed_run_slack=float(tuning.get("speed_run_slack", 0.10)),
        models=models,
        speed_dims=list(search.get("speed_dims", [])),
        dims={k: v for k, v in search.get("dims", {}).items()},
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text
    return str(value)


def flags_to_args(flags: dict[str, Any]) -> list[str]:
    """{"ctx-size": 4096, "flash-attn": "on", "jinja": True} -> ["--ctx-size", "4096", ...]."""
    args: list[str] = []
    for key, value in flags.items():
        if value is None or value is False:
            continue
        args.append("--" + key)
        if value is not True:
            args.append(_fmt(value))
    return args


def describe(flags: dict[str, Any]) -> str:
    return " ".join(flags_to_args(flags))


def diff_flags(old: dict[str, Any], new: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = sorted(set(old) | set(new))
    return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}


def describe_diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    parts = []
    for k, (a, b) in diff_flags(old, new).items():
        if b is None:
            parts.append(f"-{k}")
        else:
            parts.append(f"{k}={_fmt(b)}")
    return ", ".join(parts) if parts else "(no change)"


def thinking_enabled(flags: dict[str, Any]) -> bool:
    return flags.get("reasoning") != "off" and flags.get("reasoning-budget") != 0
