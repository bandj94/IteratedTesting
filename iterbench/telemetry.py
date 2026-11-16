"""GPU and RAM readings (nvidia-smi and OS calls) plus a background GPU sampler.

The sampler records temperature, SM clock, power and throttle reasons while a benchmark runs, so a
slow run caused by heat (laptops!) can be told apart from a slow configuration.
"""
from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

_FIELDS = ["temperature.gpu", "clocks.sm", "power.draw", "utilization.gpu", "memory.used",
           "memory.total"]
_REASON_FIELDS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]
_reason_field: str | None = _REASON_FIELDS[0]

# nvmlClocksEventReasons bits
POWER_CAP = 0x4
THERMAL = 0x8 | 0x40 | 0x80 | 0x100


def _num(text: str):
    text = text.strip()
    try:
        return int(text, 16) if text.startswith("0x") else float(text)
    except ValueError:
        return None


def gpu_query() -> dict | None:
    """One nvidia-smi reading, or None when there is no NVIDIA GPU."""
    global _reason_field
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    fields = _FIELDS + ([_reason_field] if _reason_field else [])
    try:
        out = subprocess.run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 and _reason_field:
        # Older drivers only know the "throttle" spelling, very old ones neither.
        idx = _REASON_FIELDS.index(_reason_field)
        _reason_field = _REASON_FIELDS[idx + 1] if idx + 1 < len(_REASON_FIELDS) else None
        return gpu_query()
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return None
    values = [_num(v) for v in line[0].split(",")]
    names = ["temp_c", "sm_clock_mhz", "power_w", "util_pct", "memory_used_mib", "memory_total_mib",
             "reasons"][:len(values)]
    return dict(zip(names, values))


def ram_available_mib() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / 2**20
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            return out or os.environ.get("PROCESSOR_IDENTIFIER", "?")
        except (OSError, subprocess.TimeoutExpired):
            return os.environ.get("PROCESSOR_IDENTIFIER", "?")
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def gpu_name() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "none"
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out.splitlines()[0] if out else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


class GpuSampler(threading.Thread):
    """Polls nvidia-smi every `interval` seconds until stop() is called."""

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._halt = threading.Event()     # not "_stop": threading.Thread uses that name internally

    def run(self) -> None:
        while not self._halt.is_set():
            s = gpu_query()
            if s:
                s["t"] = time.time()
                self.samples.append(s)
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=30)

    def summary(self) -> dict:
        busy = [s for s in self.samples if (s.get("util_pct") or 0) >= 20]
        if not self.samples:
            return {}

        def col(rows, key):
            return [r[key] for r in rows if r.get(key) is not None]

        reasons = [int(r["reasons"]) for r in busy if r.get("reasons") is not None]
        temps = col(self.samples, "temp_c")
        return {
            "samples": len(self.samples),
            "max_temp_c": max(temps) if temps else None,
            "mean_temp_busy_c": round(statistics.mean(col(busy, "temp_c")), 1) if busy else None,
            "mean_sm_clock_busy_mhz": round(statistics.mean(col(busy, "sm_clock_mhz")), 0) if busy else None,
            "mean_power_busy_w": round(statistics.mean(col(busy, "power_w")), 1) if busy else None,
            "max_vram_mib": max(col(self.samples, "memory_used_mib") or [0]),
            "frac_power_capped": round(sum(1 for r in reasons if r & POWER_CAP) / len(reasons), 3) if reasons else None,
            "frac_thermal": round(sum(1 for r in reasons if r & THERMAL) / len(reasons), 3) if reasons else None,
        }
