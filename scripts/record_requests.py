r"""Record what an OpenAI-compatible client (such as OpenCode) sends to a llama.cpp server.

A stub server: it answers every chat request with "ok" and appends the request's settings to a JSONL
file. Prompt text is never stored: for `messages` and `tools` only their count and roles/names are
kept. Binds 127.0.0.1 only.

Usage:
    python scripts/record_requests.py 8099 requests.jsonl

Then point the client at http://127.0.0.1:8099/v1 instead of llama-server. For OpenCode, without
touching its config or session history (Git Bash syntax; run it from an empty folder):

    OPENCODE_CONFIG_CONTENT='{"autoupdate":false,"provider":{"llamacpp":{"options":{"baseURL":"http://127.0.0.1:8099/v1"}}}}' \
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 \
    XDG_DATA_HOME=./oc-data XDG_STATE_HOME=./oc-state \
    opencode run -m llamacpp/ornith-9b "Reply with the single word ok."

Any sampling field in the log (temperature, top_p, ...) overrides the server's launch flags.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ("ornith-9b", "ornith-35b", "qwen3.6-35b", "qwen3.8-27b")


def summarize(body: dict) -> dict:
    out = {}
    for k, v in body.items():
        if k == "messages":
            out[k] = {"count": len(v), "roles": [m.get("role") for m in v]}
        elif k == "tools":
            out[k] = {"count": len(v), "names": [t.get("function", {}).get("name") for t in v][:40]}
        else:
            out[k] = v
    return out


def make_handler(log_path: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _record(self, entry: dict) -> None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        def _send_json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._record({"method": "GET", "path": self.path})
            if self.path.rstrip("/").endswith("/models"):
                self._send_json(200, {"object": "list",
                                      "data": [{"id": m, "object": "model", "owned_by": "stub"} for m in MODELS]})
            else:
                self._send_json(404, {})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"_unparsed_bytes": len(raw)}
            self._record({"method": "POST", "path": self.path, "body": summarize(body)})
            model, now = body.get("model", "stub"), int(time.time())
            usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunks = [
                    {"id": "c1", "object": "chat.completion.chunk", "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                                  "finish_reason": None}]},
                    {"id": "c1", "object": "chat.completion.chunk", "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
                ]
                for c in chunks:
                    self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._send_json(200, {"id": "c1", "object": "chat.completion", "created": now, "model": model,
                                      "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                   "finish_reason": "stop"}], "usage": usage})

    return Handler


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    port, log_path = int(sys.argv[1]), sys.argv[2]
    print(f"recording requests to {log_path}; listening on http://127.0.0.1:{port}/v1 (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(log_path)).serve_forever()


if __name__ == "__main__":
    main()
