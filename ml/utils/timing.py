"""Wall-clock measurement helpers.

Timing a CUDA model needs an explicit synchronise, otherwise the measurement
only captures the time to *enqueue* the kernels.  Every inference-time number
reported by this project goes through here.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = ["Stopwatch", "stopwatch", "synchronize", "TimedResult", "time_call"]

T = TypeVar("T")


def synchronize(device: "torch.device | str | None" = None) -> None:
    """Block until queued CUDA work on *device* has finished.

    ``device=None`` means "this is CPU work" and does **nothing**.  That is
    deliberate and matters a great deal: a full ``torch.cuda.synchronize()``
    costs milliseconds, and timing a purely CPU stage (image decoding,
    OpenCV preprocessing) must not pay it.  Measured at ~22 ms per image in
    the preprocessing pipeline before this was made explicit -- twenty times
    the cost of the work being timed.

    Callers timing GPU work know their device and pass it.

    torch is imported lazily so that the preprocessing pipeline -- which times
    its stages through this module -- stays importable in a plain
    OpenCV + NumPy environment with no deep-learning stack installed.
    """
    if device is None:
        return

    try:
        import torch as _torch
    except ImportError:
        return

    if not _torch.cuda.is_available():
        return

    resolved: Any = _torch.device(device)
    if resolved.type == "cuda":
        _torch.cuda.synchronize(resolved)


@dataclass(slots=True)
class Stopwatch:
    """Accumulating stopwatch in milliseconds."""

    elapsed_ms: float = 0.0
    _start: float | None = None

    def start(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Stopwatch.stop() called before start()")
        self.elapsed_ms += (time.perf_counter() - self._start) * 1000.0
        self._start = None
        return self.elapsed_ms

    def reset(self) -> None:
        self.elapsed_ms = 0.0
        self._start = None


@contextmanager
def stopwatch(device: "torch.device | str | None" = None) -> Iterator[Stopwatch]:
    """Context manager yielding a :class:`Stopwatch`.

    CUDA work is synchronised on both entry and exit so the measurement covers
    only the body of the ``with`` block.

    >>> with stopwatch(device) as sw:
    ...     logits = model(batch)
    >>> sw.elapsed_ms
    """
    synchronize(device)
    watch = Stopwatch().start()
    try:
        yield watch
    finally:
        synchronize(device)
        watch.stop()


@dataclass(slots=True)
class TimedResult:
    """A function's return value together with how long it took."""

    value: object
    elapsed_ms: float


def time_call(
    fn: Callable[..., T],
    *args: object,
    device: "torch.device | str | None" = None,
    **kwargs: object,
) -> tuple[T, float]:
    """Call ``fn(*args, **kwargs)`` and return ``(result, elapsed_ms)``."""
    synchronize(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0
