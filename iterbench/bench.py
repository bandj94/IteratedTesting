"""One benchmark run: start llama-server with a configuration, run task splits, stop the server."""
from __future__ import annotations

import re
from pathlib import Path

from .client import Client, ClientError
from .config import Config, ModelSpec
from .server import LlamaServer
from .suite import run_suite
from .tasks import load_tasks, select
from .telemetry import GpuSampler
from .util import RUNS_DIR, log, now_iso, stamp, write_json


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


def run_benchmark(cfg: Config, model: ModelSpec, flags: dict, *, splits=("tune",), suite: str | None = None,
                  label: str = "", run_dir: Path | None = None, seed_offset: int = 0,
                  limit: int | None = None, only: list[str] | None = None) -> dict:
    """Run one benchmark and return its record (also saved as run.json in the run folder)."""
    suite = suite or model.suite
    run_id = f"{stamp()}_{model.name}_{slug(label)}"
    run_dir = Path(run_dir or RUNS_DIR / model.name / run_id)
    record: dict = {"run_id": run_id, "model": model.name, "label": label, "flags": dict(flags), "suite": suite,
                    "seed_offset": seed_offset, "started": now_iso(), "splits": {}, "error": None,
                    "run_dir": str(run_dir)}
    log(f"--- {model.name}: {label or 'benchmark'} ({suite} suite, splits {', '.join(splits)})")
    server = LlamaServer(cfg, model, flags, run_dir / "server.log")
    record["server"] = server.start()
    sampler = GpuSampler()
    sampler.start()
    items_all = {}
    try:
        client = Client(server.base_url)
        for split in splits:
            tasks = select(load_tasks(split), suite)
            if only:
                tasks = [t for t in tasks if any(t.id.endswith(o) or t.category == o for o in only)]
            if limit:
                tasks = tasks[:limit]
            res = run_suite(client, tasks, run_dir / split, scale=model.max_tokens_scale,
                            seed_offset=seed_offset, label=f"{model.name} {split} ")
            record["splits"][split] = res["summary"]
            items_all[split] = res["items"]
            s = res["summary"]
            log(f"=== {model.name} {split}: score {s['score']:.1f}% ({s['n_passed']}/{s['n_items']} items) in "
                f"{s['wall_s'] / 60:.1f} min, gen {s['gen_tps']} tok/s, prefill {s['pp_tps']} tok/s, "
                f"{s['reasoning_tokens']} thinking tokens")
    except ClientError as e:
        record["error"] = f"server stopped responding: {e}"
        record["server_log_tail"] = server.log_tail(40)
        log("ERROR " + record["error"])
    finally:
        try:
            sampler.stop()
        finally:
            server.stop()           # always free the GPU, even if something above failed
        record["gpu"] = sampler.summary()
        record["finished"] = now_iso()
        write_json(run_dir / "run.json", {**record, "items": items_all})
    return record
