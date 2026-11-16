"""Start and stop llama-server for one configuration."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config, ModelSpec, flags_to_args
from .telemetry import gpu_query, ram_available_mib
from .util import log

IS_WIN = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000
GPU_IDLE_MIB = 800          # more VRAM than this in use by others means the GPU is not free


class ServerError(RuntimeError):
    pass


def first_error_line(log_text: str) -> str:
    """The first log line that names an error (such as "CUDA error: out of memory"), without its timestamp."""
    for line in log_text.splitlines():
        if re.search(r"error|out of memory|failed", line, re.IGNORECASE):
            return re.sub(r"^[\d.]+ [A-Z] +", "", line).strip()[:160]
    return ""


def find_llama_servers() -> list[int]:
    """PIDs of running llama-server processes (ours or anyone else's)."""
    try:
        if IS_WIN:
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq llama-server.exe"],
                                 capture_output=True, text=True, timeout=20).stdout
            pids = []
            for line in out.splitlines():
                parts = [p.strip().strip('"') for p in line.split('","')]
                if len(parts) > 1 and parts[0].lower().startswith("llama-server"):
                    pids.append(int(parts[1]))
            return pids
        out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True, timeout=20).stdout
        return [int(x) for x in out.split()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def wait_gpu_idle(timeout_s: float = 120) -> float | None:
    """Wait until VRAM use drops below GPU_IDLE_MIB. Returns the last reading (MiB)."""
    t0 = time.time()
    used = None
    while True:
        q = gpu_query()
        if q is None:
            return None
        used = q.get("memory_used_mib")
        if used is None or used < GPU_IDLE_MIB:
            return used
        if time.time() - t0 > timeout_s:
            return used
        time.sleep(3)


def _kill_on_close_job(proc: subprocess.Popen):
    """Windows: put the server in a job object that is killed when this Python process exits, so a
    crashed or interrupted harness never leaves a model sitting in VRAM."""
    if not IS_WIN:
        return None
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", ctypes.c_ulonglong * 6),
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
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    return job


def _get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LlamaServer:
    """Context manager: `with LlamaServer(cfg, model, flags, log_path) as srv: ...`."""

    def __init__(self, cfg: Config, model: ModelSpec, flags: dict, log_path: Path):
        self.cfg, self.model, self.flags = cfg, model, dict(flags)
        self.log_path = Path(log_path)
        self.base_url = f"http://{cfg.host}:{cfg.port}"
        self.proc: subprocess.Popen | None = None
        self._logf = None
        self._job = None
        self.info: dict = {}

    def command(self) -> list[str]:
        return [str(self.cfg.server_exe), "-m", str(self.model.file), "--alias", self.model.name,
                "--host", self.cfg.host, "--port", str(self.cfg.port), "--metrics", "--no-webui",
                *flags_to_args(self.flags)]

    def start(self) -> dict:
        if not self.model.file.exists():
            raise ServerError(f"model file not found: {self.model.file}")
        others = find_llama_servers()
        if others:
            raise ServerError(f"another llama-server is already running (PID {others}); close it first "
                              "so it does not share the GPU with the benchmark")
        if port_in_use(self.cfg.host, self.cfg.port):
            raise ServerError(f"port {self.cfg.port} is already in use")
        used = wait_gpu_idle(120)
        if used is not None and used >= GPU_IDLE_MIB:
            raise ServerError(f"{used:.0f} MiB of VRAM is in use by another program; free the GPU first")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = self.command()
        self._logf.write("# " + subprocess.list2cmdline(cmd) + "\n")
        self._logf.flush()
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=self._logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     creationflags=CREATE_NO_WINDOW if IS_WIN else 0)
        self._job = _kill_on_close_job(self.proc)
        while True:
            if self.proc.poll() is not None:
                tail = self.log_tail()
                self._close_log()
                cause = first_error_line(tail)
                raise ServerError(f"llama-server exited with code {self.proc.returncode} while loading"
                                  f"{f' ({cause})' if cause else ''}:\n{tail}")
            if time.perf_counter() - t0 > self.cfg.startup_timeout_s:
                self.stop()
                raise ServerError("llama-server did not become ready in time")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1)
        load_s = time.perf_counter() - t0
        props = {}
        try:
            props = _get_json(self.base_url + "/props")
        except (urllib.error.URLError, OSError, ValueError):
            pass
        dgs = props.get("default_generation_settings") or {}
        gpu = gpu_query() or {}
        self.info = {
            "load_s": round(load_s, 1),
            "n_ctx": dgs.get("n_ctx") or props.get("n_ctx"),
            "build": props.get("build_info"),
            "vram_used_mib": gpu.get("memory_used_mib"),
            "ram_available_mib": round(ram_available_mib() or 0),
            "command": subprocess.list2cmdline(cmd),
        }
        log(f"server ready in {load_s:.0f}s (ctx {self.info['n_ctx']}, VRAM {self.info['vram_used_mib']} MiB, "
            f"free RAM {self.info['ram_available_mib']} MiB)")
        return self.info

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 25) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def _close_log(self) -> None:
        if self._logf:
            self._logf.close()
            self._logf = None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        if self._job:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self._job)
            self._job = None
        self._close_log()
        wait_gpu_idle(60)

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
