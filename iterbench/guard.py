"""Run model-written Python with guard rails. This is NOT a security sandbox.

It blocks the obvious ways test code could reach outside its scratch folder: starting processes,
opening network sockets, loading native libraries through ctypes, the Windows registry, and
writing, deleting or renaming files outside ROOT. It uses a Python audit hook (PEP 578), which
cannot be removed once installed. The parent process also adds a timeout and a memory/process
limit. That is enough for benign benchmark tasks; do not use it for hostile code.

Usage (always started by iterbench.sandbox, never imported):
    python -I -B guard.py ROOT script FILE
    python -I -B guard.py ROOT unittest START_DIR [START_DIR ...]
"""
import json
import os
import sys

_BLOCKED = {
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile",
    "os.fork", "os.forkpty", "os.kill", "os.killpg", "pty.spawn", "webbrowser.open",
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect", "imaplib.open",
    "poplib.connect", "nntplib.connect", "telnetlib.Telnet.open",
}
_BLOCKED_PREFIXES = ("ctypes.", "winreg.", "_winapi.", "msvcrt.")
# event name -> indexes of path arguments that must stay inside ROOT
_PATH_EVENTS = {
    "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.truncate": (0,),
    "os.chmod": (0,), "os.chown": (0,), "os.link": (0, 1), "os.symlink": (0, 1), "os.utime": (0,),
    "os.chflags": (0,), "os.lchflags": (0,), "shutil.rmtree": (0,), "shutil.move": (0, 1),
    "shutil.copyfile": (1,), "shutil.copytree": (1,), "shutil.copymode": (1,), "shutil.copystat": (1,),
}


def install(root: str) -> None:
    root = os.path.normcase(os.path.abspath(root))

    def inside(path) -> bool:
        if path is None or isinstance(path, int):
            return True
        try:
            full = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        except Exception:
            return False
        return full == root or full.startswith(root + os.sep)

    def hook(event, args):
        if event in _BLOCKED or event.startswith(_BLOCKED_PREFIXES):
            raise PermissionError(f"blocked by the benchmark guard: {event}")
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if mode is not None:
                writing = any(c in str(mode) for c in "wax+")
            else:
                writing = bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
            if writing and not inside(path):
                raise PermissionError(f"blocked by the benchmark guard: writing outside the sandbox ({path})")
        elif event in _PATH_EVENTS:
            for i in _PATH_EVENTS[event]:
                if i < len(args) and not inside(args[i]):
                    raise PermissionError(f"blocked by the benchmark guard: {event} outside the sandbox")

    sys.addaudithook(hook)


def _run_unittest(root: str, start_dirs: list[str]) -> None:
    import unittest

    total = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "ok": True, "dirs": []}
    for start in start_dirs:
        start_path = os.path.join(root, start)
        if not os.path.isdir(start_path):
            continue
        loader = unittest.TestLoader()
        suite = loader.discover(start_path, pattern="test*.py", top_level_dir=start_path)
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
        result = runner.run(suite)
        errors = len(result.errors) + len(loader.errors)
        total["tests"] += result.testsRun
        total["failures"] += len(result.failures)
        total["errors"] += errors
        total["skipped"] += len(result.skipped)
        total["ok"] = total["ok"] and result.wasSuccessful() and not loader.errors
        total["dirs"].append({"dir": start, "tests": result.testsRun, "failures": len(result.failures),
                              "errors": errors})
    total["ok"] = total["ok"] and total["tests"] > 0
    print("@@RESULT@@" + json.dumps(total), flush=True)


def main() -> None:
    root, mode = sys.argv[1], sys.argv[2]
    root = os.path.abspath(root)
    os.chdir(root)
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    install(root)
    if mode == "script":
        import runpy

        runpy.run_path(sys.argv[3], run_name="__main__")
    elif mode == "unittest":
        _run_unittest(root, sys.argv[3:])
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
