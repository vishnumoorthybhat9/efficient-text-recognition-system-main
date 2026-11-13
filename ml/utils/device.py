"""Hardware detection with a guaranteed CPU fallback.

The project is developed on a 4 GB laptop GPU, so every call site has to cope
with CUDA being absent, busy, or out of memory.  Nothing in this module raises
on a machine without a GPU.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

__all__ = ["DeviceInfo", "resolve_device", "describe_hardware", "autocast_dtype", "empty_cache"]

DevicePreference = Literal["auto", "cuda", "cpu"]


@dataclass(slots=True)
class GpuInfo:
    """Static description of one CUDA device."""

    index: int
    name: str
    total_memory_mb: int
    capability: str
    multi_processor_count: int


@dataclass(slots=True)
class DeviceInfo:
    """Everything the UI and the trainers need to know about the host."""

    device: str
    device_type: Literal["cuda", "cpu"]
    cuda_available: bool
    amp_supported: bool
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    gpus: list[GpuInfo] = field(default_factory=list)
    cpu_name: str = ""
    cpu_count: int = 0
    total_ram_mb: int = 0
    platform: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpus"] = [asdict(g) if not isinstance(g, dict) else g for g in self.gpus]
        return payload


def resolve_device(preference: DevicePreference | str = "auto") -> torch.device:
    """Turn a user preference into a usable :class:`torch.device`.

    ``"cuda"`` is honoured only when CUDA really is available; otherwise a
    warning is logged and the CPU is used.  This keeps the application runnable
    on a machine with no GPU without any configuration change.
    """
    pref = (preference or "auto").strip().lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("device='cuda' requested but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")

    if pref != "auto":
        logger.warning("Unknown device preference %r; treating it as 'auto'", preference)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Pick the autocast dtype for mixed-precision training on *device*.

    bfloat16 is preferred when supported because it needs no loss scaling;
    Turing-class cards (such as the RTX 2050 this was developed on) fall back
    to float16.
    """
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # pragma: no cover - depends on the driver
        logger.debug("bf16 support probe failed; assuming fp16", exc_info=True)
    return torch.float16


def empty_cache(device: torch.device | None = None) -> None:
    """Release cached CUDA blocks.  A no-op on CPU."""
    if device is None or device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cpu_name() -> str:
    """Best-effort CPU model string across platforms."""
    name = platform.processor() or ""
    if name and not name.lower().startswith(("x86", "amd64", "i386")):
        return name

    if platform.system() == "Windows":
        value = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if value:
            return value
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Darwin" and shutil.which("sysctl"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return name or platform.machine()


def _total_ram_mb() -> int:
    try:
        import psutil  # imported lazily: optional dependency

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:  # pragma: no cover - psutil missing or unsupported
        return 0


@lru_cache(maxsize=4)
def describe_hardware(preference: DevicePreference | str = "auto") -> DeviceInfo:
    """Collect a full hardware report.  Cached: the hardware does not change."""
    device = resolve_device(preference)
    gpus: list[GpuInfo] = []

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    GpuInfo(
                        index=index,
                        name=props.name,
                        total_memory_mb=int(props.total_memory / (1024 * 1024)),
                        capability=f"{props.major}.{props.minor}",
                        multi_processor_count=props.multi_processor_count,
                    )
                )
            except Exception:  # pragma: no cover - driver hiccup
                logger.warning("Could not query CUDA device %d", index, exc_info=True)

    return DeviceInfo(
        device=str(device),
        device_type=device.type,  # type: ignore[arg-type]
        cuda_available=torch.cuda.is_available(),
        amp_supported=device.type == "cuda",
        # str(): torch.__version__ is a TorchVersion instance, not a plain str.
        # This report is embedded in checkpoints, and a non-builtin type there
        # makes them unloadable under torch.load(weights_only=True).
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        cudnn_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        gpus=gpus,
        cpu_name=_cpu_name(),
        cpu_count=os.cpu_count() or 0,
        total_ram_mb=_total_ram_mb(),
        platform=f"{platform.system()} {platform.release()}",
    )
