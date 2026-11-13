"""Stop Windows from sleeping while a long training run is in progress.

Started in the background by TRAIN_MODELS.bat. It asks Windows to keep the
system awake (the screen may still switch off) for as long as the flag file
exists, and exits on its own once the batch file deletes it -- or after 24
hours at most, so it can never keep a machine awake forever.

Closing the laptop lid can still send it to sleep, depending on the power
settings, so keep the lid open while training.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
MAX_SECONDS = 24 * 3600


def main() -> int:
    flag = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/logs/training.running")
    if sys.platform != "win32":
        return 0
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    started = time.time()
    try:
        while flag.exists() and time.time() - started < MAX_SECONDS:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(60)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
