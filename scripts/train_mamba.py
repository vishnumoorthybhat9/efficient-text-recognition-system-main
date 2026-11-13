"""Train the CNN + bidirectional Mamba + CTC recogniser."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_crnn import main as train_ctc_model  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in arguments:
        arguments = ["--config", str(PROJECT_ROOT / "configs" / "mamba.yaml"), *arguments]
    return train_ctc_model(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
