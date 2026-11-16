"""Smoke test: does MTP speculative decoding (--spec-type draft-mtp) load, keep answers sane, and help?

Runs each model twice (thinking off, to keep it short): without and with MTP, on the same two prompts
and seeds, and prints the answers, writing speed and the draft acceptance rate. Not part of the
benchmark; used once to decide whether MTP belongs in the search space.

    python scripts/smoke_spec.py ornith-35b qwen3.8-27b
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iterbench.client import Client  # noqa: E402
from iterbench.config import load_config  # noqa: E402
from iterbench.server import LlamaServer, ServerError  # noqa: E402
from iterbench.util import RUNS_DIR  # noqa: E402

PROMPTS = [
    "Write a Python function is_prime(n) that returns True for prime integers. Reply with only the code.",
    "In three sentences, explain what a hash map is and why lookups are fast on average.",
]


def run(cfg, model, flags, tag, max_tokens):
    out = []
    log = RUNS_DIR / "smoke" / f"{model.name}_{tag}.log"
    try:
        with LlamaServer(cfg, model, flags, log) as srv:
            c = Client(srv.base_url)
            for i, p in enumerate(PROMPTS):
                t0 = time.perf_counter()
                data = c._post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}],
                                                        "max_tokens": max_tokens, "seed": 1234 + i,
                                                        "stream": False})
                wall = time.perf_counter() - t0
                if "__error__" in data:
                    out.append({"error": data["__error__"]})
                    continue
                msg = data["choices"][0]["message"]
                tm = data.get("timings") or {}
                out.append({"content": msg.get("content") or "", "tps": tm.get("predicted_per_second"),
                            "n": tm.get("predicted_n"), "draft_n": tm.get("draft_n"),
                            "draft_ok": tm.get("draft_n_accepted"), "wall": wall,
                            "finish": data["choices"][0].get("finish_reason")})
    except ServerError as e:
        out.append({"error": f"server: {e}"[:600]})
    return out


def main(names):
    cfg = load_config()
    for name in names:
        model = cfg.models[name]
        base = dict(model.baseline, reasoning="off")
        max_tokens = 160 if model.suite == "lite" else 320
        for tag, flags in (("no-spec", base), ("mtp", dict(base, **{"spec-type": "draft-mtp"}))):
            print(f"\n===== {name} [{tag}]", flush=True)
            for i, r in enumerate(run(cfg, model, flags, tag, max_tokens)):
                if "error" in r:
                    print(f"  prompt {i}: ERROR {r['error']}")
                    continue
                acc = (f", draft accepted {r['draft_ok']}/{r['draft_n']}"
                       if r.get("draft_n") else "")
                print(f"  prompt {i}: {r['n']} tokens at {r['tps']:.2f} tok/s ({r['wall']:.1f}s, "
                      f"finish {r['finish']}{acc})")
                print("    " + r["content"].strip().replace("\n", "\n    ")[:700])


if __name__ == "__main__":
    main(sys.argv[1:] or ["ornith-35b"])
