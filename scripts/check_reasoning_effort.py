"""Does a request's `reasoning_effort` field change anything on llama-server for a tuned model?

OpenCode adds `reasoning_effort: "low"` to its title requests. This starts the model with its tuned
flags (from results/best_configs.json) on a spare port, then compares the rendered prompt and seeded
generations with and without the field. Identical output means the field has no effect.

    python scripts/check_reasoning_effort.py ornith-9b
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iterbench.config import load_config  # noqa: E402
from iterbench.server import LlamaServer  # noqa: E402
from iterbench.util import RUNS_DIR  # noqa: E402

MESSAGES = [{"role": "system", "content": "You write short titles for coding sessions."},
            {"role": "user", "content": "Write a title (at most 6 words) for this request: the login form "
                                        "accepts empty passwords; fix it and add a test."}]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "ornith-9b"
    cfg = load_config()
    cfg.port = 8097                                   # away from the launchers' 8080 and the harness port
    flags = json.loads((ROOT / "results" / "best_configs.json").read_text(encoding="utf-8"))[name]["flags"]
    with LlamaServer(cfg, cfg.models[name], flags, RUNS_DIR / "smoke" / f"{name}_reasoning_effort.log") as srv:
        plain = post(srv.base_url + "/apply-template", {"messages": MESSAGES})["prompt"]
        low = post(srv.base_url + "/apply-template", {"messages": MESSAGES, "reasoning_effort": "low"})["prompt"]
        print("rendered prompt identical with reasoning_effort=low:", plain == low)
        for extra in ({}, {"reasoning_effort": "low"}, {}):
            r = post(srv.base_url + "/v1/chat/completions",
                     {"messages": MESSAGES, "max_tokens": 3000, "seed": 7, **extra})
            msg, t = r["choices"][0]["message"], r.get("timings", {})
            print(f"{json.dumps(extra) if extra else '(no field)':28s} tokens written {t.get('predicted_n')}, "
                  f"thinking {len(msg.get('reasoning_content') or '')} chars, "
                  f"answer {(msg.get('content') or '').strip()[:60]!r}")


if __name__ == "__main__":
    main()
