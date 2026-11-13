"""Project path resolution.

Every other module asks this one where the project root is, so that scripts
work no matter which directory they are launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "resolve_path", "ensure_dir", "relative_to_root"]


def _detect_project_root() -> Path:
    """Return the project root directory.

    Resolution order:

    1. the ``OCR_PROJECT_ROOT`` environment variable, when set;
    2. the grandparent of this file (``<root>/ml/utils/paths.py``).
    """
    env_root = os.environ.get("OCR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _detect_project_root()


def resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Resolve *value* against :data:`PROJECT_ROOT` unless it is absolute.

    ``None`` is passed through so optional config entries stay optional.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(value: str | os.PathLike[str]) -> Path:
    """Resolve *value*, create it as a directory if missing, and return it."""
    path = resolve_path(value)
    assert path is not None  # resolve_path only returns None for a None input
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_root(value: str | os.PathLike[str]) -> str:
    """Return *value* as a POSIX path relative to the project root when possible.

    Used for log and API output so absolute Windows paths do not leak into the
    UI or the database.
    """
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
