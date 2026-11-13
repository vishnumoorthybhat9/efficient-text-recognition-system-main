"""YAML configuration loader with single-level inheritance and deep merge.

An experiment file declares ``extends: default.yaml`` and overrides only the
keys it cares about::

    extends: default.yaml
    name: baseline_scratch
    train:
      epochs: 25        # every other `train` key comes from default.yaml

Values are reachable by attribute, by key, or by dotted path::

    cfg.train.epochs
    cfg["train"]["epochs"]
    cfg.get("train.epochs", 10)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from ml.utils.paths import PROJECT_ROOT, resolve_path

__all__ = ["Config", "load_config", "deep_merge", "ConfigError"]

CONFIG_DIR = PROJECT_ROOT / "configs"

# Guards against an `extends` cycle; one level is all the project needs, but a
# small limit is cheap insurance.
_MAX_EXTENDS_DEPTH = 8


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed or circular."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested mappings are merged key by key; every other type (including lists)
    is replaced wholesale, which is what you want for e.g. ``channels``.
    """
    merged: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config(Mapping[str, Any]):
    """Read-only, attribute-accessible view over a nested dict."""

    __slots__ = ("_data", "_source")

    def __init__(self, data: Mapping[str, Any], source: str | None = None) -> None:
        self._data: dict[str, Any] = dict(data)
        self._source = source

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value, self._source) if isinstance(value, Mapping) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Ergonomics ---------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails, so __slots__ members
        # never reach here.
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key {name!r}"
                + (f" (loaded from {self._source})" if self._source else "")
            ) from exc

    def get(self, path: str, default: Any = None) -> Any:  # type: ignore[override]
        """Look up a dotted *path*, returning *default* when any step is absent.

        >>> cfg.get("train.epochs", 10)
        """
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return Config(node, self._source) if isinstance(node, Mapping) else node

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy as plain Python containers (JSON serialisable)."""
        return copy.deepcopy(self._data)

    def merged_with(self, override: Mapping[str, Any]) -> "Config":
        """Return a new Config with *override* deep-merged on top."""
        return Config(deep_merge(self._data, override), self._source)

    @property
    def source(self) -> str | None:
        """Path of the file this config was loaded from, if any."""
        return self._source

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({json.dumps(self._data, default=str, sort_keys=True)[:200]}...)"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _locate(reference: str | Path, relative_to: Path | None) -> Path:
    """Resolve a config reference to a real file.

    Tried in order: as given / relative to the referencing file / relative to
    ``configs/`` / relative to the project root.
    """
    candidate = Path(reference).expanduser()
    if candidate.is_absolute():
        return candidate

    searched: list[Path] = []
    if relative_to is not None:
        searched.append(relative_to / candidate)
    searched.append(CONFIG_DIR / candidate)
    searched.append(PROJECT_ROOT / candidate)

    for option in searched:
        if option.is_file():
            return option
    # Return the most likely location so the error message is useful.
    return searched[0]


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    _depth: int = 0,
) -> Config:
    """Load a YAML config, applying ``extends`` inheritance then *overrides*.

    Args:
        path: file to load, absolute or relative to ``configs/``.
        overrides: optional mapping deep-merged last (used for CLI flags).

    Raises:
        ConfigError: the file is missing, malformed, or the inheritance chain
            is too deep.
    """
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ConfigError(
            f"`extends` chain deeper than {_MAX_EXTENDS_DEPTH} levels while loading {path}; "
            "this usually means two files extend each other."
        )

    config_path = _locate(path, relative_to=None)
    raw = _read_yaml(config_path)

    parent_ref = raw.pop("extends", None)
    if parent_ref is not None:
        if not isinstance(parent_ref, str):
            raise ConfigError(f"`extends` in {config_path} must be a string, got {parent_ref!r}")
        parent_path = _locate(parent_ref, relative_to=config_path.parent)
        parent = load_config(parent_path, _depth=_depth + 1)
        raw = deep_merge(parent.to_dict(), raw)

    if overrides:
        raw = deep_merge(raw, overrides)

    # Record where this came from; experiment scripts write it into the report.
    raw.setdefault("_config_path", str(config_path))
    return Config(raw, source=str(config_path))


def resolve_config_paths(cfg: Config, keys: tuple[str, ...]) -> dict[str, Path | None]:
    """Resolve a set of dotted config *keys* to absolute paths.

    Convenience for scripts that need several directories at once.
    """
    return {key: resolve_path(cfg.get(key)) for key in keys}
