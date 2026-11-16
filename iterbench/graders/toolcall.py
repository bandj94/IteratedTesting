"""Tool-calling tasks: does the model pick the right tool with the right arguments (or answer
directly when no tool is needed)? Uses the server's OpenAI-style `tools` support, so it also
tests that the model's tool-call format survives the server's parser under each configuration.
"""
from __future__ import annotations

import json
import re

SYSTEM = ("You are a coding assistant working in the user's software project. Use the provided tools "
          "when the request needs them; when you can answer directly, answer without calling a tool. "
          "Today's date is 2026-11-15.")

TOOLS: dict[str, dict] = {
    "read_file": {
        "description": "Read a text file from the project and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the project root"},
            "start_line": {"type": "integer", "description": "First line to return (1-based, optional)"},
            "end_line": {"type": "integer", "description": "Last line to return, inclusive (optional)"}},
            "required": ["path"]}},
    "write_file": {
        "description": "Create a file, or overwrite it, with exactly the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the project root"},
            "content": {"type": "string", "description": "The complete new file content"}},
            "required": ["path", "content"]}},
    "edit_file": {
        "description": "Replace one exact occurrence of old_text with new_text in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"]}},
    "run_command": {
        "description": "Run a shell command in the project root and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "Maximum run time in seconds (default 60)"}},
            "required": ["command"]}},
    "search_code": {
        "description": "Search the project's files for a text pattern. Returns matching lines with file "
                       "names and line numbers.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Text to look for"},
            "path": {"type": "string", "description": "Directory or file to search (default: whole project)"},
            "regex": {"type": "boolean", "description": "Treat the pattern as a regular expression"}},
            "required": ["pattern"]}},
    "list_dir": {
        "description": "List the files and folders inside a directory of the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path relative to the project root"}},
            "required": ["path"]}},
    "create_ticket": {
        "description": "Create an issue in the team's issue tracker.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "due_date": {"type": "string", "description": "Due date as YYYY-MM-DD"},
            "labels": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "priority"]}},
    "web_fetch": {
        "description": "Download a web page and return its text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
}


def tool_defs(names: list[str]) -> list[dict]:
    return [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in names]


def build_request(data: dict) -> tuple[list[dict], list[dict]]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    for m in data["messages"]:
        msg: dict = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                 for tc in m["tool_calls"]]
        if m["role"] == "tool":
            msg["tool_call_id"] = m["tool_call_id"]
        messages.append(msg)
    return messages, tool_defs(data.get("tools", list(TOOLS)))


def _norm_path(p) -> str:
    p = str(p).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") or p


def match_value(value, rule: dict) -> bool:
    text = value if isinstance(value, str) else json.dumps(value)
    if "path" in rule and not (isinstance(value, str) and _norm_path(value) == _norm_path(rule["path"])):
        return False
    if "equals" in rule and value != rule["equals"]:
        return False
    if "iequals" in rule and not (isinstance(value, str) and value.strip().lower() == rule["iequals"].strip().lower()):
        return False
    if "contains" in rule and not all(s.lower() in text.lower() for s in rule["contains"]):
        return False
    if "contains_any" in rule and not any(s.lower() in text.lower() for s in rule["contains_any"]):
        return False
    if "regex" in rule and not re.search(rule["regex"], text, re.I):
        return False
    if "number" in rule:
        if isinstance(value, bool):
            return False
        try:
            if float(value) != float(rule["number"]):
                return False
        except (TypeError, ValueError):
            return False
    if "set" in rule:
        if not isinstance(value, list):
            return False
        if {str(x).strip().lower() for x in value} != {s.lower() for s in rule["set"]}:
            return False
    if "json" in rule:
        try:
            if json.loads(value) != rule["json"]:
                return False
        except (TypeError, ValueError):
            return False
    if "text" in rule:
        if not isinstance(value, str):
            return False
        if value.replace("\r\n", "\n").rstrip("\n") != rule["text"].replace("\r\n", "\n").rstrip("\n"):
            return False
    return True


def match_call(call: dict, spec: dict) -> str | None:
    """None if the call matches the spec, otherwise the reason it does not."""
    if call.get("name") != spec["name"]:
        return f"called {call.get('name')} instead of {spec['name']}"
    args = call.get("args")
    if args is None:
        return "arguments were not valid JSON"
    for key, rule in spec.get("args", {}).items():
        if key not in args:
            return f"missing argument {key}"
        if not match_value(args[key], rule):
            return f"argument {key}={args[key]!r} does not match {rule}"
    return None


def grade(data: dict, content: str, tool_calls: list[dict]) -> dict:
    exp = data["expect"]
    calls = [{"name": c["name"], "args": c["args"]} for c in tool_calls]
    if exp["type"] == "answer":
        if tool_calls:
            return {"passed": False, "partial": 0.0, "detail": f"called a tool ({calls[0]['name']}) instead of answering",
                    "calls": calls}
        text = (content or "").lower()
        ok = all(s.lower() in text for s in exp.get("contains_all", []))
        if exp.get("contains_any"):
            ok = ok and any(s.lower() in text for s in exp["contains_any"])
        return {"passed": ok, "partial": 1.0 if ok else 0.0,
                "detail": "" if ok else f"answer missing expected text: {content[:200]!r}", "calls": calls}
    specs = exp.get("any_of") or [{"name": exp["name"], "args": exp.get("args", {})}]
    if not tool_calls:
        return {"passed": False, "partial": 0.0, "detail": f"no tool call; answered: {(content or '')[:200]!r}",
                "calls": calls}
    reasons = []
    for c in tool_calls:
        for s in specs:
            why = match_call(c, s)
            if why is None:
                return {"passed": True, "partial": 1.0, "detail": "", "calls": calls}
            reasons.append(why)
    return {"passed": False, "partial": 0.0, "detail": "; ".join(reasons[:3]), "calls": calls}
