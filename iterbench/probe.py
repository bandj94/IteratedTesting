"""Speed probe: a short, fixed measurement for settings that do not change the model's answers.

One probe = start the server, then
  1. read a ~16K-token prompt with the cache off      -> prompt-reading speed (pp)
  2. generate N tokens after that 16K prompt           -> writing speed at agent-like depth (tg_deep)
  3. generate M tokens after a tiny prompt             -> writing speed at short depth (tg_short)
The probe's score is the estimated time of one typical agent turn at ~16K context:
    turn_s = 3000 / pp + 500 / tg_deep        (3K new prompt tokens read, 500 tokens written)
"""
from __future__ import annotations

from pathlib import Path

from .client import Client
from .config import Config, ModelSpec
from .graders.longctx import generate
from .server import LlamaServer, ServerError
from .util import log, now_iso

DEEP_TOKENS = 16000
# A probe request normally takes 10-90 s (the 27B's 16K prefill is the slowest). A setting that makes it
# take 15 minutes (for example a load mode that pages the model to disk) is a loser: fail it, don't wait.
PROBE_REQUEST_TIMEOUT_S = 900
_DEEP_TEXT: str | None = None


def deep_prompt() -> str:
    global _DEEP_TEXT
    if _DEEP_TEXT is None:
        _DEEP_TEXT = generate(seed=4242, target_tokens=DEEP_TOKENS)["code"]
    return _DEEP_TEXT


def turn_seconds(pp_tps: float, tg_tps: float) -> float:
    return 3000 / pp_tps + 500 / tg_tps


def run_probe(cfg: Config, model: ModelSpec, flags: dict, log_path: Path, label: str = "") -> dict:
    n_deep, n_short = model.probe_tokens
    out = {"label": label, "flags": dict(flags), "started": now_iso(), "error": None}
    server = LlamaServer(cfg, model, flags, log_path)
    try:
        out["server"] = server.start()
        c = Client(server.base_url, timeout=PROBE_REQUEST_TIMEOUT_S)
        c.completion("def warm_up():\n", n_predict=32, cache_prompt=False, ignore_eos=True)
        r1 = c.completion(deep_prompt(), n_predict=1, cache_prompt=False)
        t1 = r1.get("timings") or {}
        r2 = c.completion(deep_prompt(), n_predict=n_deep, cache_prompt=True, ignore_eos=True)
        t2 = r2.get("timings") or {}
        r3 = c.completion("# A small utility module\nimport os\n\n\ndef ", n_predict=n_short, cache_prompt=False,
                          ignore_eos=True)
        t3 = r3.get("timings") or {}
        pp, tg_deep, tg_short = t1.get("prompt_per_second"), t2.get("predicted_per_second"), t3.get("predicted_per_second")
        out.update({"prompt_tokens": t1.get("prompt_n"), "pp_tps": round(pp, 1), "tg_deep_tps": round(tg_deep, 2),
                    "tg_short_tps": round(tg_short, 2), "turn_s": round(turn_seconds(pp, tg_deep), 2)})
        log(f"probe {model.name} [{label}]: prefill {pp:.0f} tok/s, write@16K {tg_deep:.1f} tok/s, "
            f"write@0 {tg_short:.1f} tok/s -> {out['turn_s']:.1f} s/turn")
    except (ServerError, Exception) as e:  # a setting that fails to load is just a losing candidate
        out["error"] = f"{type(e).__name__}: {e}"[:1500]
        log(f"probe {model.name} [{label}] FAILED: {out['error'][:200]}")
    finally:
        server.stop()
        out["finished"] = now_iso()
    return out
