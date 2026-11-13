from __future__ import annotations

import json
from pathlib import Path

from scripts.check_training_complete import main


def test_completed_run_requires_finished_history_and_best_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(
        json.dumps({"finished_at": 123.0, "epochs_completed": 1}),
        encoding="utf-8",
    )
    (tmp_path / "best.pt").touch()

    assert main([str(tmp_path)]) == 0


def test_incomplete_run_is_not_accepted(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(
        json.dumps({"finished_at": None, "epochs_completed": 1}),
        encoding="utf-8",
    )
    (tmp_path / "best.pt").touch()

    assert main([str(tmp_path)]) == 1


def test_missing_checkpoint_is_not_accepted(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(
        json.dumps({"finished_at": 123.0, "epochs_completed": 1}),
        encoding="utf-8",
    )

    assert main([str(tmp_path)]) == 1
