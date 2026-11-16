"""Minimal HTTP client for llama-server (OpenAI-compatible chat plus a few native endpoints)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ClientError(RuntimeError):
    """The server could not be reached or returned a server error (5xx)."""


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)     # {"id", "name", "arguments", "args"}
    raw_tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_n: int = 0           # prompt tokens processed by this request (cached ones excluded)
    cache_n: int = 0            # prompt tokens reused from the cache
    prompt_ms: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    wall_s: float = 0.0
    error: str | None = None    # request rejected (for example: prompt longer than the context)

    def metrics(self) -> dict:
        return {"prompt_n": self.prompt_n, "cache_n": self.cache_n, "prompt_ms": round(self.prompt_ms, 1),
                "predicted_n": self.predicted_n, "predicted_ms": round(self.predicted_ms, 1),
                "wall_s": round(self.wall_s, 2), "finish_reason": self.finish_reason, "error": self.error}


class Client:
    def __init__(self, base_url: str, timeout: float = 7200):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            if e.code >= 500:
                raise ClientError(f"HTTP {e.code}: {body}") from e
            return {"__error__": f"HTTP {e.code}: {body}"}
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise ClientError(f"request to {path} failed: {e}") from e

    def chat(self, messages: list[dict], *, max_tokens: int, seed: int,
             tools: list[dict] | None = None, extra: dict | None = None) -> ChatResult:
        payload: dict[str, Any] = {"messages": messages, "max_tokens": int(max_tokens), "seed": int(seed),
                                   "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if extra:
            payload.update(extra)
        t0 = time.perf_counter()
        data = self._post("/v1/chat/completions", payload)
        res = ChatResult(wall_s=time.perf_counter() - t0)
        if "__error__" in data:
            res.error = data["__error__"]
            return res
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        res.content = msg.get("content") or ""
        res.reasoning = msg.get("reasoning_content") or ""
        res.finish_reason = choice.get("finish_reason")
        res.raw_tool_calls = msg.get("tool_calls") or []
        for tc in res.raw_tool_calls:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                args, raw = raw, json.dumps(raw)
            else:
                try:
                    args = json.loads(raw) if raw else {}
                except (TypeError, ValueError):
                    args = None
            res.tool_calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": raw,
                                   "args": args if isinstance(args, dict) else None})
        tm = data.get("timings") or {}
        usage = data.get("usage") or {}
        res.prompt_n = int(tm.get("prompt_n") or 0)
        res.cache_n = int(tm.get("cache_n") or 0)
        res.prompt_ms = float(tm.get("prompt_ms") or 0.0)
        res.predicted_n = int(tm.get("predicted_n") or usage.get("completion_tokens") or 0)
        res.predicted_ms = float(tm.get("predicted_ms") or 0.0)
        return res

    def completion(self, prompt: str, *, n_predict: int, cache_prompt: bool = True,
                   ignore_eos: bool = False, temperature: float = 0.0, seed: int = 0) -> dict:
        payload = {"prompt": prompt, "n_predict": int(n_predict), "cache_prompt": cache_prompt,
                   "ignore_eos": ignore_eos, "temperature": temperature, "seed": seed, "stream": False}
        data = self._post("/completion", payload)
        if "__error__" in data:
            raise ClientError(data["__error__"])
        return data

    def tokenize(self, text: str) -> int:
        if not text:
            return 0
        data = self._post("/tokenize", {"content": text}, timeout=120)
        return len(data.get("tokens") or [])
