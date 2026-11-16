"""Run guarded Python in a child process with a timeout, a memory cap and no child processes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

GUARD = Path(__file__).resolve().parent / "guard.py"
IS_WIN = os.name == "nt"
MARKER = "@@RESULT@@"


@dataclass
class GuardResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    result: dict | None          # the JSON after the last @@RESULT@@ marker, if any


def _windows_job(proc: subprocess.Popen, mem_mb: int):
    """Put the child in a job object: memory cap, no child processes, killed with the job."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    # PROCESS_MEMORY | ACTIVE_PROCESS | KILL_ON_JOB_CLOSE
    info.BasicLimitInformation.LimitFlags = 0x100 | 0x8 | 0x2000
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = mem_mb * 1024 * 1024
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    return job


def _close_job(job) -> None:
    if job:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(job)


def run_guarded(root: Path, args: list[str], timeout: float = 20, mem_mb: int = 1024) -> GuardResult:
    """Run `python -I -B guard.py ROOT *args` inside ROOT and return its output."""
    root = Path(root).resolve()
    # The base interpreter, not a venv redirector: a redirector starts a second process, which the
    # job object's one-process limit would block. Tasks only need the standard library.
    python = getattr(sys, "_base_executable", None) or sys.executable
    cmd = [python, "-I", "-B", str(GUARD), str(root), *args]
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "TEMP": str(root), "TMP": str(root),
           "TMPDIR": str(root)}
    for key in ("SYSTEMROOT", "SystemRoot", "WINDIR", "PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    kwargs: dict = {}
    if IS_WIN:
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    else:
        def _limits():  # pragma: no cover - POSIX only
            import resource
            limit = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        kwargs["preexec_fn"] = _limits
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, **kwargs)
    job = _windows_job(proc, mem_mb) if IS_WIN else None
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    finally:
        _close_job(job)
    out = out.decode("utf-8", "replace") if isinstance(out, bytes) else (out or "")
    err = err.decode("utf-8", "replace") if isinstance(err, bytes) else (err or "")
    result = None
    idx = out.rfind(MARKER)
    if idx >= 0:
        line = out[idx + len(MARKER):].splitlines()[0] if out[idx + len(MARKER):] else ""
        try:
            result = json.loads(line)
        except ValueError:
            result = None
    return GuardResult(proc.returncode, out, err, timed_out, result)
