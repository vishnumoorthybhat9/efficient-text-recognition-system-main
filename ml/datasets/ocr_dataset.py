"""PyTorch datasets and collate functions.

Two datasets, one for each training stage:

* :class:`OCRLineDataset` -- labeled images for supervised CTC training.
* :class:`UnlabeledLineDataset` -- unlabeled images yielding two augmented
  views for SimCLR.

Both read the manifest format from :mod:`ml.datasets.manifest` and share the
same preprocessing and geometry, so the encoder sees an identically
distributed input in both stages.  That is a prerequisite for the transferred
weights to be useful at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.datasets.manifest import ManifestEntry, read_manifest, read_unlabeled_manifest
from ml.preprocessing.augment import (
    AugmentOptions,
    OCRAugmentor,
    SimCLRViewGenerator,
    SimCLRViewOptions,
)
from ml.preprocessing.pipeline import PreprocessOptions, PreprocessPipeline, prepare_model_input
from ml.text.charset import Charset
from ml.utils.image_io import ImageDecodeError, read_image
from ml.utils.seed import worker_init_fn

__all__ = [
    "ImageGeometry",
    "OCRSample",
    "OCRBatch",
    "SimCLRBatch",
    "OCRLineDataset",
    "UnlabeledLineDataset",
    "collate_ocr",
    "collate_simclr",
    "build_ocr_dataloader",
    "build_simclr_dataloader",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageGeometry:
    """Fixed input geometry of the network."""

    height: int = 32
    width: int = 256
    channels: int = 1
    keep_aspect_ratio: bool = True
    pad_value: int = 255

    @classmethod
    def from_config(cls, cfg: Any) -> "ImageGeometry":
        """Build from the ``image`` block of a config."""
        if cfg is None:
            return cls()
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: d)
        return cls(
            height=int(get("height", 32) or 32),
            width=int(get("width", 256) or 256),
            channels=int(get("channels", 1) or 1),
            keep_aspect_ratio=bool(get("keep_aspect_ratio", True)),
            pad_value=int(get("pad_value", 255)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "keep_aspect_ratio": self.keep_aspect_ratio,
            "pad_value": self.pad_value,
        }


@dataclass(slots=True)
class OCRSample:
    """One labeled training sample."""

    image: torch.Tensor          # (C, H, W) float32
    target: torch.Tensor         # (L,) int64 class indices
    text: str
    width_ratio: float           # fraction of the canvas holding real content
    sample_id: str


@dataclass(slots=True)
class OCRBatch:
    """A collated supervised batch, shaped for :class:`torch.nn.CTCLoss`.

    ``targets`` is the *concatenated*, unpadded form CTC accepts, which avoids
    needing a padding index that the loss would then have to ignore.
    """

    images: torch.Tensor          # (B, C, H, W)
    targets: torch.Tensor         # (sum(target_lengths),)
    target_lengths: torch.Tensor  # (B,)
    width_ratios: torch.Tensor    # (B,) float
    texts: list[str]
    sample_ids: list[str]

    def __len__(self) -> int:
        return self.images.size(0)

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "OCRBatch":
        """Move the tensors to *device*, leaving the Python fields alone."""
        return OCRBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            target_lengths=self.target_lengths.to(device, non_blocking=non_blocking),
            width_ratios=self.width_ratios.to(device, non_blocking=non_blocking),
            texts=self.texts,
            sample_ids=self.sample_ids,
        )

    def input_lengths(self, time_steps: int) -> torch.Tensor:
        """Valid time steps per sample, given the encoder emits *time_steps*.

        Derived from how much of the padded canvas each sample actually
        occupies.  Two guarantees are enforced:

        * at least one frame, so ``pack_padded_sequence`` does not raise;
        * at least ``target_length`` frames, because CTC cannot emit a label
          longer than the input and would otherwise return ``inf``.
        """
        scaled = torch.ceil(self.width_ratios.to(torch.float32) * float(time_steps))
        lengths = scaled.to(torch.long).clamp(min=1, max=int(time_steps))
        floor = self.target_lengths.to(lengths.device, dtype=torch.long)
        return torch.maximum(lengths, floor).clamp(max=int(time_steps))


@dataclass(slots=True)
class SimCLRBatch:
    """A collated contrastive batch: two views of the same images."""

    view_i: torch.Tensor  # (B, C, H, W)
    view_j: torch.Tensor  # (B, C, H, W)
    sample_ids: list[str]

    def __len__(self) -> int:
        return self.view_i.size(0)

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "SimCLRBatch":
        return SimCLRBatch(
            view_i=self.view_i.to(device, non_blocking=non_blocking),
            view_j=self.view_j.to(device, non_blocking=non_blocking),
            sample_ids=self.sample_ids,
        )


class _BaseLineDataset(Dataset):
    """Shared image loading and shaping."""

    def __init__(
        self,
        *,
        geometry: ImageGeometry,
        preprocess: PreprocessOptions | None,
    ) -> None:
        self.geometry = geometry
        # Grayscale is forced: the shaping step decides the channel count, and
        # letting the pipeline keep colour here would silently break it.
        base = preprocess or PreprocessOptions()
        self._pipeline = PreprocessPipeline(base.with_overrides(grayscale=True))

    def _load_and_enhance(self, path: Path) -> np.ndarray:
        """Read an image and run the enhancement pipeline over it."""
        image = read_image(path, grayscale=True)
        return self._pipeline.run(image).image

    def _shape(self, image: np.ndarray) -> tuple[torch.Tensor, float]:
        """Fit *image* to the network geometry.

        Returns:
            ``(chw_tensor, width_ratio)``.
        """
        geometry = self.geometry
        array, info = prepare_model_input(
            image,
            height=geometry.height,
            width=geometry.width,
            channels=geometry.channels,
            keep_aspect_ratio=geometry.keep_aspect_ratio,
            pad_value=geometry.pad_value,
            normalize=True,
        )
        return torch.from_numpy(array), float(info.content_ratio)


class OCRLineDataset(_BaseLineDataset):
    """Labeled text-line images for supervised CTC training.

    Args:
        entries: manifest entries, or pass *manifest* instead.
        charset: encodes labels into CTC targets.
        geometry: network input geometry.
        preprocess: enhancement applied before augmentation.
        augment: augmentation options; ignored unless *train*.
        train: enables augmentation and marks the split in logs.
        max_samples: keep only the first N entries.  This is how the low-label
            regime of the research experiment is created, and it takes a
            *deterministic* prefix of a seeded shuffle so both experiment arms
            see the identical subset.
        seed: controls the subset shuffle and the augmentation RNG.
        drop_unencodable: discard entries whose label has no representation in
            the charset, rather than training on a silently truncated target.

    Raises:
        ValueError: neither *entries* nor *manifest* was given, or nothing
            usable remains after filtering.
    """

    def __init__(
        self,
        *,
        charset: Charset,
        entries: Sequence[ManifestEntry] | None = None,
        manifest: str | Path | None = None,
        geometry: ImageGeometry | None = None,
        preprocess: PreprocessOptions | None = None,
        augment: AugmentOptions | None = None,
        train: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
        drop_unencodable: bool = True,
    ) -> None:
        super().__init__(geometry=geometry or ImageGeometry(), preprocess=preprocess)

        if entries is None:
            if manifest is None:
                raise ValueError("OCRLineDataset needs either `entries` or `manifest`")
            entries = read_manifest(manifest, skip_missing=True)

        self.charset = charset
        self.train = bool(train)
        self.seed = int(seed)

        usable = self._filter(list(entries), drop_unencodable=drop_unencodable)
        self.entries = self._subsample(usable, max_samples)

        self._augmentor = (
            OCRAugmentor(augment or AugmentOptions(), seed=self.seed) if self.train else None
        )

        logger.info(
            "OCRLineDataset(%s): %d sample(s)%s",
            "train" if self.train else "eval",
            len(self.entries),
            f" (capped from {len(usable)})" if max_samples and len(usable) > len(self.entries) else "",
        )

    def _filter(self, entries: list[ManifestEntry], *, drop_unencodable: bool) -> list[ManifestEntry]:
        if not drop_unencodable:
            return entries

        kept: list[ManifestEntry] = []
        dropped_characters: dict[str, int] = {}
        for entry in entries:
            missing = self.charset.unsupported(entry.label)
            if missing and not self.charset.encode(entry.label):
                # Nothing at all survives encoding: the sample is unusable.
                for character in missing:
                    dropped_characters[character] = dropped_characters.get(character, 0) + 1
                continue
            kept.append(entry)

        if len(kept) != len(entries):
            logger.warning(
                "Dropped %d sample(s) whose labels are entirely outside the charset "
                "(characters: %s)",
                len(entries) - len(kept),
                sorted(dropped_characters)[:20],
            )
        if not kept:
            raise ValueError(
                "No sample survived charset filtering. The manifest labels and "
                "charset.characters are incompatible."
            )
        return kept

    def _subsample(self, entries: list[ManifestEntry], max_samples: int | None) -> list[ManifestEntry]:
        if max_samples is None or max_samples >= len(entries):
            return entries
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

        # Seeded permutation, not a raw slice: a manifest is usually written in
        # generation order, so the first N entries could be biased by style.
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(entries))[:max_samples]
        return [entries[int(i)] for i in sorted(order)]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> OCRSample:
        entry = self.entries[index]

        try:
            image = self._load_and_enhance(entry.path)
        except (FileNotFoundError, ImageDecodeError) as exc:
            # A DataLoader worker raising here produces an unreadable traceback,
            # so make the message name the offending file explicitly.
            raise RuntimeError(f"Could not load sample {entry.relative_path!r}: {exc}") from exc

        if self._augmentor is not None:
            image = self._augmentor(image)

        tensor, width_ratio = self._shape(image)
        encoded = self.charset.encode(entry.label)

        return OCRSample(
            image=tensor,
            target=torch.tensor(encoded, dtype=torch.long),
            text=self.charset.normalize(entry.label),
            width_ratio=width_ratio,
            sample_id=entry.relative_path,
        )

    @property
    def labels(self) -> list[str]:
        """The ground-truth strings, in dataset order."""
        return [self.charset.normalize(entry.label) for entry in self.entries]


class UnlabeledLineDataset(_BaseLineDataset):
    """Unlabeled images yielding two augmented views for SimCLR.

    Args:
        paths: image paths, or pass *manifest* instead.
        geometry: network input geometry -- must match the recogniser's.
        preprocess: enhancement applied before view generation.
        views: strength of the contrastive augmentation.
        max_samples: cap the pool size.
        seed: RNG seed for view generation.
    """

    def __init__(
        self,
        *,
        paths: Sequence[Path] | None = None,
        manifest: str | Path | None = None,
        geometry: ImageGeometry | None = None,
        preprocess: PreprocessOptions | None = None,
        views: SimCLRViewOptions | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(geometry=geometry or ImageGeometry(), preprocess=preprocess)

        if paths is None:
            if manifest is None:
                raise ValueError("UnlabeledLineDataset needs either `paths` or `manifest`")
            paths = read_unlabeled_manifest(manifest, skip_missing=True)

        resolved = list(paths)
        if max_samples is not None and 0 < max_samples < len(resolved):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(resolved))[:max_samples]
            resolved = [resolved[int(i)] for i in sorted(order)]

        if not resolved:
            raise ValueError("UnlabeledLineDataset received no usable image paths")

        self.paths = resolved
        self.seed = int(seed)
        self._views = SimCLRViewGenerator(views or SimCLRViewOptions(), seed=self.seed)

        logger.info("UnlabeledLineDataset: %d unlabeled image(s)", len(self.paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        try:
            image = self._load_and_enhance(path)
        except (FileNotFoundError, ImageDecodeError) as exc:
            raise RuntimeError(f"Could not load unlabeled image {path.name!r}: {exc}") from exc

        view_a, view_b = self._views(image)
        tensor_a, _ = self._shape(view_a)
        tensor_b, _ = self._shape(view_b)
        return tensor_a, tensor_b, path.name


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------
def collate_ocr(samples: Sequence[OCRSample]) -> OCRBatch:
    """Collate :class:`OCRSample` items into an :class:`OCRBatch`."""
    if not samples:
        raise ValueError("collate_ocr received an empty batch")

    return OCRBatch(
        images=torch.stack([s.image for s in samples], dim=0),
        targets=torch.cat([s.target for s in samples], dim=0),
        target_lengths=torch.tensor([s.target.numel() for s in samples], dtype=torch.long),
        width_ratios=torch.tensor([s.width_ratio for s in samples], dtype=torch.float32),
        texts=[s.text for s in samples],
        sample_ids=[s.sample_id for s in samples],
    )


def collate_simclr(samples: Sequence[tuple[torch.Tensor, torch.Tensor, str]]) -> SimCLRBatch:
    """Collate two-view items into a :class:`SimCLRBatch`."""
    if not samples:
        raise ValueError("collate_simclr received an empty batch")

    return SimCLRBatch(
        view_i=torch.stack([s[0] for s in samples], dim=0),
        view_j=torch.stack([s[1] for s in samples], dim=0),
        sample_ids=[s[2] for s in samples],
    )


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------
def _loader_kwargs(num_workers: int, device: torch.device | None) -> dict[str, Any]:
    """DataLoader keyword arguments that are safe on Windows.

    ``num_workers > 0`` uses spawn on Windows, which re-imports the whole
    module tree in every worker; it only pays off for large datasets, so the
    project defaults to 0.  ``persistent_workers`` and ``prefetch_factor`` are
    invalid when there are no workers, hence the conditional.
    """
    kwargs: dict[str, Any] = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(device is not None and device.type == "cuda"),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = worker_init_fn
    return kwargs


def build_ocr_dataloader(
    dataset: OCRLineDataset,
    *,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for supervised training or evaluation.

    *shuffle* and *drop_last* default to ``dataset.train``: shuffling an
    evaluation set would scramble the per-sample report, and dropping the last
    partial batch would quietly discard test samples.
    """
    is_train = dataset.train
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=is_train if shuffle is None else bool(shuffle),
        drop_last=is_train if drop_last is None else bool(drop_last),
        collate_fn=collate_ocr,
        **_loader_kwargs(num_workers, device),
    )


def build_simclr_dataloader(
    dataset: UnlabeledLineDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    """Build a DataLoader for contrastive pretraining.

    ``drop_last`` is forced on: NT-Xent's loss scale depends on the batch size,
    so a short final batch would make the epoch's mean loss jump for reasons
    that have nothing to do with learning.
    """
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_simclr,
        **_loader_kwargs(num_workers, device),
    )
