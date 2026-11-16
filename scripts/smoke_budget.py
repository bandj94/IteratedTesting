"""Smoke test: does --reasoning-budget N stop thinking near N tokens, and does the model still answer?

The tuning loop relies on this: when the budget runs out, llama.cpp closes the thinking block and the
model has to write its answer. Starts the model's original launcher settings plus the budget, asks
for a function that invites long thinking, and prints how many tokens went to thinking and answering.

    python scripts/smoke_budget.py ornith-9b 300
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iterbench.client import Client  # noqa: E402
from iterbench.config import load_config  # noqa: E402
from iterbench.server import LlamaServer  # noqa: E402
from iterbench.util import RUNS_DIR  # noqa: E402

PROMPT = ("Write a Python function longest_palindrome(s) returning the longest palindromic substring "
          "(first one if tied). Think it through carefully, then give only the code.")


def main() -> None:
    cfg = load_config()
    model = cfg.models[sys.argv[1] if len(sys.argv) > 1 else "ornith-9b"]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    flags = dict(model.baseline, **{"reasoning-budget": budget})
    with LlamaServer(cfg, model, flags, RUNS_DIR / "smoke" / f"{model.name}_budget{budget}.log") as srv:
        c = Client(srv.base_url)
        r = c.chat([{"role": "user", "content": PROMPT}], max_tokens=budget + 1500, seed=7)
        print(f"finish={r.finish_reason} written={r.predicted_n} thinking~{c.tokenize(r.reasoning)} "
              f"answer~{c.tokenize(r.content)} tokens (budget {budget})")
        print("--- end of thinking:", r.reasoning[-300:].replace("\n", " | "))
        print("--- start of answer:", r.content[:500])


if __name__ == "__main__":
    main()
