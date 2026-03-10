"""GPU monitoring via pynvml (NVML)."""

from typing import Any

_PYNVML_OK = False
try:
    import pynvml

    pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    pass


def gpu_metrics() -> dict[str, Any] | None:
    """Return GPU telemetry dict or None if pynvml unavailable."""
    if not _PYNVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "name": name,
            "temp_c": temp,
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "vram_used_gb": round(mem.used / (1024**3), 2),
            "vram_total_gb": round(mem.total / (1024**3), 2),
            "vram_pct": round(mem.used / mem.total * 100, 1),
        }
    except Exception:
        return None
