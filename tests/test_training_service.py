from __future__ import annotations

from pathlib import Path

from backend.services import training_service


def test_resolve_encoder_checkpoint_uses_configured_path(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "encoder_best.pt"
    checkpoint.touch()
    monkeypatch.setattr(training_service, "resolve_path", lambda _: checkpoint)

    assert training_service.resolve_encoder_checkpoint("encoder_best.pt") == checkpoint


def test_resolve_encoder_checkpoint_supports_legacy_nested_output(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "simclr_pretrain" / "encoder_best.pt"
    legacy = configured.parent / configured.parent.name / configured.name
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    monkeypatch.setattr(training_service, "resolve_path", lambda _: configured)

    assert training_service.resolve_encoder_checkpoint("encoder_best.pt") == legacy
