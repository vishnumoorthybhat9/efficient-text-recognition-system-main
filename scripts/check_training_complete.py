"""Verify that a training run completed and saved its best checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: check_training_complete.py CHECKPOINT_DIRECTORY", file=sys.stderr)
        return 2

    directory = Path(args[0])
    history_path = directory / "history.json"
    checkpoint_path = directory / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        print(
            f"Training is incomplete: expected {history_path} and {checkpoint_path}.",
            file=sys.stderr,
        )
        return 1

    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not verify training history at {history_path}: {exc}", file=sys.stderr)
        return 1

    if history.get("finished_at") is None or int(history.get("epochs_completed", 0)) < 1:
        print(f"Training history does not record a completed epoch: {history_path}.", file=sys.stderr)
        return 1

    print(f"Verified completed training run in {directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
