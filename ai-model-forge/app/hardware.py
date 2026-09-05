"""Hardware detection: CPU / NVIDIA GPU / VRAM / CUDA / RAM / disk.

The training engine will later adapt batch sizes, precision and methods to
this spec. Detection is lazy so the API and model builder never pay for it
unless something asks.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareSpec:
    device: str                    # "cuda" | "cpu" (mps ignored: not targeted initially)
    backend: str                   # torch backend label, e.g. "cpu", "cuda:0"
    gpu_name: str | None = None
    gpu_vram_bytes: int | None = None
    cuda_version: str | None = None
    ram_bytes: int = 0
    cpus: int = 0
    disk_free_bytes: int = 0
    torch_version: str = ""
    has_gpu: bool = False
    supports_flash_attn: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:  # pragma: no cover - broken driver edge case
        return 0


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareSpec:
    """Detect the runtime environment once per process."""
    cpus = os.cpu_count() or 0
    try:
        import resource

        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        ram = 0
    if ram <= 0:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        ram = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass

    disk_free = 0
    try:
        disk_free = shutil.disk_usage("/").free
    except OSError:
        pass

    torch_version = torch.__version__
    count = _cuda_device_count()
    if count:
        try:
            name = torch.cuda.get_device_name(0)
            vram = int(torch.cuda.get_device_properties(0).total_memory)
            cuda = torch.version.cuda or "unknown"
        except Exception:  # pragma: no cover
            name, vram, cuda = "nvidia-gpu", None, None
        return HardwareSpec(
            device="cuda",
            backend="cuda:0",
            gpu_name=name,
            gpu_vram_bytes=vram,
            cuda_version=cuda,
            ram_bytes=ram,
            cpus=cpus,
            disk_free_bytes=disk_free,
            torch_version=torch_version,
            has_gpu=True,
            supports_flash_attn=True,
        )

    return HardwareSpec(
        device="cpu",
        backend="cpu",
        ram_bytes=ram,
        cpus=cpus,
        disk_free_bytes=disk_free,
        torch_version=torch_version,
    )


def log_hardware_summary() -> None:
    spec = detect_hardware()
    if spec.has_gpu:
        log.info("Hardware: NVIDIA GPU '%s' | VRAM %s MiB | RAM %s MiB | CUDA %s",
                 spec.gpu_name, (spec.gpu_vram_bytes or 0) // 2**20, spec.ram_bytes // 2**20, spec.cuda_version)
    else:
        log.info("Hardware: CPU only (%s cores, %s MiB RAM) — CUDA not available", spec.cpus, spec.ram_bytes // 2**20)
