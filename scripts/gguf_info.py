"""Print a GGUF model's architecture metadata and what its built-in chat template reads (stdlib only).

Shows layer, expert and attention settings (these hybrid models keep a KV cache in only every fourth
layer, which is why KV-cache precision barely changes their speed) and whether the chat template uses
switches such as `enable_thinking` or `reasoning_effort` (what a client's request field can change).

    python scripts/gguf_info.py path/to/model.gguf [more.gguf ...]
"""
from __future__ import annotations

import struct
import sys

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
KEYS = ("general.", "context_length", "expert", "block_count", "sampling", "full_attention", "ssm",
        "head_count", "quantize", "nextn", "file_type")
TEMPLATE_WORDS = ("enable_thinking", "reasoning_effort", "thinking_budget", "<think>", "<tool_call>",
                  "<function=", "tool_calls")


def read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_val(f, t):
    if t in SCALARS:
        fmt = SCALARS[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et in SCALARS and n > 64:
            f.seek(n * struct.calcsize(SCALARS[et]), 1)
            return f"<array of {n}>"
        if et == 8 and n > 64:
            for _ in range(n):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            return f"<array of {n} strings>"
        return [read_val(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {t}")


def metadata(path: str) -> dict:
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        f.read(4)                                     # version
        _, n_kv = struct.unpack("<QQ", f.read(16))    # tensor count, key/value count
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            out[key] = read_val(f, t)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        m = metadata(path)
        template = m.pop("tokenizer.chat_template", "") or ""
        print("=" * 80)
        print(path)
        for k, v in m.items():
            if any(s in k for s in KEYS) and not k.startswith("tokenizer."):
                print(f"  {k} = {v}")
        print(f"  chat template: {len(template)} characters")
        for word in TEMPLATE_WORDS:
            print(f"    uses {word!r}: {template.count(word)} times")


if __name__ == "__main__":
    main()
