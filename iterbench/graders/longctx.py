"""Long-context tasks: trace a value through a large, generated code base.

Each context is a deterministic, generated Python package (`shop`, roughly 8K-24K tokens). Every
module defines the same constant names (LIMIT, FACTOR, ...) with different values, and function
names repeat across modules, so the model has to resolve each name in the right module. A question
asks what one function returns for a given input; answering needs 2-3 hops across modules that sit
far apart in the prompt. The expected answer comes from executing the generated code.
"""
from __future__ import annotations

import random
import re
import types

MODULES = ["accounts", "audit", "billing", "cache", "catalog", "checkout", "exports", "fulfillment",
           "inventory", "ledger", "metrics", "notify", "orders", "payments", "pricing", "ratings",
           "refunds", "reports", "routing", "search", "sessions", "shipping", "storage", "support",
           "tax", "users", "vendors", "warehouse"]
NAMES = ["adjust_batch", "score_order", "weight_item", "level_route", "budget_window", "offset_tier",
         "factor_slot", "limit_unit", "grade_ticket", "rank_bucket", "quota_shard", "margin_cycle",
         "scale_lane", "boost_queue", "trim_ledger", "pace_cohort"]
NOUNS = ["Order", "Invoice", "Parcel", "Session", "Ticket", "Vendor", "Shelf", "Route", "Refund", "Report",
         "Account", "Coupon", "Batch", "Rating", "Payment", "Carrier"]
CHARS_PER_TOKEN = 3.1     # measured on Qwen3.5-family tokenizers for this generated code


def _consts(rng: random.Random) -> dict[str, int]:
    return {"LIMIT": rng.randint(20, 60), "FACTOR": rng.randint(2, 4), "OFFSET": rng.randint(1, 15),
            "BASE": rng.randint(1, 20), "CAP": rng.randint(40, 120), "STEP": rng.randint(1, 9)}


def _leaf(rng: random.Random) -> list[str]:
    kind = rng.choice(["lin", "lin2", "cond"])
    if kind == "lin":
        return ["return x * FACTOR + OFFSET"]
    if kind == "lin2":
        return ["y = x + BASE", "return y * FACTOR - STEP"]
    return ["if x > LIMIT:", "    return x - OFFSET", "return x + STEP"]


def _caller(rng: random.Random, callees: list[str]) -> list[str]:
    kind = rng.choice(["cap", "two", "half"]) if len(callees) > 1 else rng.choice(["cap", "half"])
    if kind == "cap":
        k = rng.randint(1, 9)
        return [f"y = {callees[0]}(x + {k})", "if y > CAP:", "    y = y - LIMIT", "return y + BASE"]
    if kind == "two":
        return [f"a = {callees[0]}(x)", f"b = {callees[1]}(x // 2)", "return max(a, b) - STEP"]
    return [f"y = {callees[0]}(x) // 2", "if y < OFFSET:", "    return OFFSET", "return y + STEP"]


def _filler(rng: random.Random, module: str) -> str:
    noun = rng.choice(NOUNS)
    low = noun.lower()
    kind = rng.randrange(4)
    if kind == 0:
        return (f'class {noun}Record:\n'
                f'    """A {low} record kept by the {module} service."""\n\n'
                f'    def __init__(self, ident, label, tags=None):\n'
                f'        self.ident = ident\n        self.label = label\n'
                f'        self.tags = list(tags or [])\n        self.history = []\n\n'
                f'    def touch(self, event):\n        """Remember an event for this {low}."""\n'
                f'        self.history.append(event)\n        return len(self.history)\n\n'
                f'    def to_dict(self):\n'
                f'        return {{"id": self.ident, "label": self.label, "tags": sorted(self.tags),\n'
                f'                "events": len(self.history)}}\n')
    if kind == 1:
        return (f'def summarize_{low}s(rows):\n'
                f'    """Group {low} rows by status and count them."""\n'
                f'    counts = {{}}\n    for row in rows:\n'
                f'        key = row.get("status", "unknown")\n'
                f'        counts[key] = counts.get(key, 0) + 1\n'
                f'    return dict(sorted(counts.items()))\n')
    if kind == 2:
        field = rng.choice(["created_at", "updated_at", "due_at", "closed_at"])
        return (f'def latest_{low}(items, field="{field}"):\n'
                f'    """Return the {low} with the most recent `{field}`, or None."""\n'
                f'    best = None\n    for item in items:\n'
                f'        stamp = item.get(field)\n'
                f'        if stamp is None:\n            continue\n'
                f'        if best is None or stamp > best.get(field):\n            best = item\n'
                f'    return best\n')
    width = rng.randint(8, 24)
    return (f'def format_{low}_line(name, amount, note=""):\n'
            f'    """Render one {low} line for the {module} report."""\n'
            f'    text = f"{{name:<{width}}} {{amount:>10.2f}}"\n'
            f'    if note:\n        text += "  # " + note\n'
            f'    return text.rstrip()\n')


def generate(seed: int, target_tokens: int) -> dict:
    """Build a code base and its question pool. Pure function of (seed, target_tokens)."""
    rng = random.Random(seed)
    n_mod = max(6, min(len(MODULES), target_tokens // 1100))
    mods = rng.sample(MODULES, n_mod)
    consts = {m: _consts(rng) for m in mods}
    funcs: dict[str, list[tuple[str, list[str], set[str]]]] = {m: [] for m in mods}
    depth: dict[str, int] = {}
    deps: dict[str, set[str]] = {}          # qualified name -> modules involved in its call tree
    order: list[str] = []                    # qualified names in creation order
    for round_no in range(3):
        for m in mods:
            used = {n for n, _, _ in funcs[m]}
            name = rng.choice([n for n in NAMES if n not in used])
            qual = f"{m}.{name}"
            candidates = [q for q in order if not q.startswith(m + ".")]
            if round_no == 0 or len(candidates) < 2 or rng.random() < 0.2:
                body, calls = _leaf(rng), set()
                depth[qual], deps[qual] = 0, {m}
            else:
                picks = rng.sample(candidates, 2)
                body = _caller(rng, picks)
                calls = {p for p in picks if p in "\n".join(body)}
                depth[qual] = 1 + max(depth[p] for p in calls)
                deps[qual] = {m}.union(*(deps[p] for p in calls))
            funcs[m].append((name, body, calls))
            order.append(qual)

    files: dict[str, str] = {}
    exec_src: dict[str, str] = {}
    for m in mods:
        imports = sorted({c.split(".")[0] for _, _, calls in funcs[m] for c in calls})
        blocks = []
        for name, body, _ in funcs[m]:
            doc = rng.choice(["Compute the adjusted value.", "Return the derived amount.",
                              "Apply this module's rules to x.", "Score x for downstream use."])
            blocks.append(f"def {name}(x):\n    \"\"\"{doc}\"\"\"\n" + "".join(f"    {line}\n" for line in body))
        head = f'"""{m.capitalize()} helpers for the shop package."""\n'
        imp = f"from shop import {', '.join(imports)}\n" if imports else ""
        const_lines = "".join(f"{k} = {v}\n" for k, v in consts[m].items())
        files[m] = [head, imp, const_lines, blocks]
        exec_src[m] = const_lines + "\n" + "\n".join(blocks)

    # Pad with filler until the target size is reached (round-robin over modules).
    target_chars = int(target_tokens * CHARS_PER_TOKEN)

    def render() -> str:
        parts = []
        for m in mods:
            head, imp, const_lines, blocks = files[m]
            parts.append(f"# ===== file: shop/{m}.py =====\n{head}{imp}\n{const_lines}\n\n" + "\n\n".join(blocks))
        return "\n\n".join(parts) + "\n"

    size = len(render())
    i = 0
    while size < target_chars:
        m = mods[i % len(mods)]
        block = _filler(rng, m)
        pos = rng.randint(0, len(files[m][3]))
        files[m][3].insert(pos, block)
        exec_src[m] += "\n\n" + block
        size += len(block) + 2
        i += 1
    code = render()

    namespaces = {m: types.ModuleType(f"shop.{m}") for m in mods}
    for m in mods:
        ns = namespaces[m].__dict__
        ns.update({k: v for k, v in namespaces.items()})
        exec(compile(exec_src[m], f"shop/{m}.py", "exec"), ns)

    pool = [q for q in order if depth[q] >= 2 and len(deps[q]) >= 3]
    pool += [q for q in order if depth[q] >= 2 and q not in pool]
    pool += [q for q in order if depth[q] == 1]
    questions = []
    for q in pool:
        m, name = q.split(".")
        x = rng.randint(5, 40)
        questions.append({"call": f"{m}.{name}({x})", "answer": int(getattr(namespaces[m], name)(x)),
                          "depth": depth[q], "modules": sorted(deps[q])})
    return {"code": code, "questions": questions, "modules": mods}


PROMPT = ("Below is the complete source code of a small Python package named `shop`. Each file starts "
          "with a `# ===== file: shop/<name>.py =====` line.\n\n{code}\n"
          "Question: using only the code above, what does `{call}` return? Work it out step by step. "
          "Every constant and function must be taken from the module it belongs to (the same names "
          "appear in several modules with different values). End your reply with a final line of the "
          "form `ANSWER: <integer>`.")

_ANSWER = re.compile(r"answer[\s*_`]*[:=][\s*_`$]*(-?\d[\d,]*)", re.I)


def build(data: dict) -> tuple[str, dict]:
    gen = generate(int(data["seed"]), int(data["target_tokens"]))
    q = gen["questions"][int(data["question_index"])]
    return PROMPT.format(code=gen["code"], call=q["call"]), q


def parse_answer(content: str) -> int | None:
    content = content or ""
    found = _ANSWER.findall(content)
    if found:
        return int(found[-1].replace(",", ""))
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    if lines:
        nums = re.findall(r"-?\d+", lines[-1])
        if len(nums) == 1:
            return int(nums[0])
    return None


def grade(question: dict, content: str) -> dict:
    got = parse_answer(content)
    ok = got is not None and got == question["answer"]
    return {"passed": ok, "partial": 1.0 if ok else 0.0, "answer": got, "expected": question["answer"],
            "detail": "" if ok else f"answered {got}, expected {question['answer']} for {question['call']}"}
