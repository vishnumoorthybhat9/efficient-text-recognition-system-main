"""Download the IAM handwriting database (line level) and add it to training.

Source
------
``Teklia/IAM-line`` on the Hugging Face Hub (MIT licence): the IAM Handwriting
Database cut into 10,373 text lines -- real English sentences written by
hundreds of different people -- with the standard train / validation / test
split (6,482 / 976 / 2,915). IAM is the benchmark most handwriting papers
report on, so results on its test split can be set against the literature.

What it produces
----------------
``data/processed/iam/``           IAM on its own: images + train/val/test.tsv
``data/processed/combined_iam/``  synthetic + Kaggle handwriting + IAM, the
                                  training set for the final two models
``data/processed/iam/prepare_report.json``  what was kept and dropped, and why

Image handling
--------------
Lines are stored grayscale at 64 px high. The CNN reads a 32 x 512 canvas
(16 : 1) and its training loader *crops* anything wider -- for a full sentence
that silently cuts words off while the label still contains them, which
teaches the model wrong answers. Lines wider than 16 : 1 are therefore
narrowed horizontally to exactly 16 : 1 here (at most ~35 %), so the whole
line always reaches the model. TrOCR resizes every image to 384 x 384 itself
and is unaffected.

Usage
-----
::

    python scripts/prepare_iam_hf.py            # download + prepare (skips if done)
    python scripts/prepare_iam_hf.py --force    # rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml  # noqa: E402,F401  (keeps the Hugging Face cache on this drive)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ml.config import load_config  # noqa: E402
from ml.datasets.manifest import read_manifest, write_manifest  # noqa: E402
from ml.text.charset import Charset  # noqa: E402
from ml.utils.image_io import ImageDecodeError, decode_image_bytes, write_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import ensure_dir, resolve_path  # noqa: E402

logger = logging.getLogger("prepare_iam")

REPO_ID = "Teklia/IAM-line"
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))  # (hub name, ours)
STORE_HEIGHT = 64
MAX_ASPECT = 16.0           # CNN canvas 512 / 32
MAX_LABEL_CHARS = 100       # CTC budget is 127 steps; leave room for repeats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IAM (line level) and build combined_iam.")
    parser.add_argument("--out", default="data/processed/iam")
    parser.add_argument(
        "--base",
        default=None,
        help="dataset to combine with IAM (auto-detects combined, then synthetic)",
    )
    parser.add_argument("--combined-out", default="data/processed/combined_iam")
    parser.add_argument("--force", action="store_true", help="rebuild even if already prepared")
    return parser.parse_args(argv)


def resolve_base_dataset(base: Path | None, data_root: Path) -> Path | None:
    """Choose a prepared dataset that has all labeled splits needed for merging."""
    candidates = [base] if base is not None else [
        data_root / "combined",
        data_root / "synthetic",
    ]
    required = ("train.tsv", "val.tsv", "test.tsv")
    for candidate in candidates:
        if candidate is not None and all((candidate / split).is_file() for split in required):
            return candidate
    return None


def fit_image(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Scale to STORE_HEIGHT and cap the aspect ratio at MAX_ASPECT."""
    h, w = gray.shape[:2]
    scale = STORE_HEIGHT / float(h)
    new_w = max(1, int(round(w * scale)))
    narrowed = False
    max_w = int(STORE_HEIGHT * MAX_ASPECT)
    if new_w > max_w:
        new_w, narrowed = max_w, True
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, (new_w, STORE_HEIGHT), interpolation=interpolation), narrowed


def prepare_iam(out: Path, charset: Charset) -> dict:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    report: dict = {"source": f"https://huggingface.co/datasets/{REPO_ID}", "licence": "MIT",
                    "splits": {}, "dropped_characters": {}}
    for hub_split, ours in SPLITS:
        logger.info("Downloading %s (%s split)...", REPO_ID, hub_split)
        parquet = hf_hub_download(REPO_ID, f"data/{hub_split}.parquet", repo_type="dataset")
        table = pq.read_table(parquet, columns=["image", "text"])
        image_dir = ensure_dir(out / "images" / ours)

        records: list[tuple[str, str]] = []
        stats = {"source_rows": table.num_rows, "kept": 0, "narrowed": 0,
                 "dropped_unsupported_chars": 0, "dropped_too_long": 0,
                 "dropped_empty": 0, "dropped_bad_image": 0}
        index = 0
        for batch in table.to_batches(max_chunksize=256):
            images = batch.column("image").to_pylist()
            texts = batch.column("text").to_pylist()
            for image, text in zip(images, texts):
                index += 1
                label = " ".join(str(text or "").split())
                if not label:
                    stats["dropped_empty"] += 1
                    continue
                bad = charset.unsupported(label)
                if bad:
                    for ch in bad:
                        report["dropped_characters"][ch] = report["dropped_characters"].get(ch, 0) + 1
                    stats["dropped_unsupported_chars"] += 1
                    continue
                if len(label) > MAX_LABEL_CHARS:
                    stats["dropped_too_long"] += 1
                    continue
                data = image.get("bytes") if isinstance(image, dict) else None
                if not data:
                    stats["dropped_bad_image"] += 1
                    continue
                try:
                    gray = decode_image_bytes(data, grayscale=True)
                except ImageDecodeError:
                    stats["dropped_bad_image"] += 1
                    continue
                if gray is None or gray.size == 0 or min(gray.shape[:2]) < 6:
                    stats["dropped_bad_image"] += 1
                    continue
                fitted, narrowed = fit_image(gray)
                stats["narrowed"] += int(narrowed)
                name = f"iam_{ours}_{index:06d}.png"
                write_image(image_dir / name, fitted)
                records.append((f"images/{ours}/{name}", label))

        write_manifest(out / f"{ours}.tsv", records,
                       header_comment=f"IAM handwriting, line level ({REPO_ID}, MIT) -- {ours} ({len(records)})")
        stats["kept"] = len(records)
        report["splits"][ours] = stats
        logger.info("IAM %s: kept %d of %d (%d narrowed to 16:1, %d dropped for unsupported characters, "
                    "%d too long)", ours, stats["kept"], stats["source_rows"], stats["narrowed"],
                    stats["dropped_unsupported_chars"], stats["dropped_too_long"])
    return report


def build_combined(base: Path, iam: Path, out: Path) -> dict:
    """Concatenate the existing combined manifests with IAM, paths relative to *out*."""
    ensure_dir(out)
    counts = {}
    for split in ("train", "val", "test"):
        entries = read_manifest(base / f"{split}.tsv") + read_manifest(iam / f"{split}.tsv")
        rows = [(Path(os.path.relpath(e.path, out)).as_posix(), e.label) for e in entries]
        write_manifest(out / f"{split}.tsv", rows,
                       header_comment=f"synthetic + Kaggle handwriting + IAM -- {split} ({len(rows)})")
        counts[split] = len(rows)
    logger.info("combined_iam: %s", counts)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("INFO", log_file=ensure_dir("artifacts/logs") / "prepare_iam.log")

    out = resolve_path(args.out) or Path(args.out)
    requested_base = resolve_path(args.base) if args.base else None
    data_root = resolve_path("data/processed")
    assert data_root is not None
    base = resolve_base_dataset(requested_base, data_root)
    combined_out = resolve_path(args.combined_out) or Path(args.combined_out)

    if not args.force and (combined_out / "test.tsv").is_file() and (out / "test.tsv").is_file():
        logger.info("IAM and combined_iam are already prepared; nothing to do (use --force to rebuild).")
        return 0
    if base is None:
        logger.error(
            "No prepared base dataset found under %s. Generate one with "
            "`python scripts/make_synthetic_data.py`, or pass --base with a "
            "dataset containing train.tsv, val.tsv and test.tsv.",
            data_root,
        )
        return 1
    if requested_base is None and base.name == "synthetic":
        logger.info("Using the prepared synthetic dataset as the base: %s", base)

    config = load_config("default.yaml")
    charset = Charset.from_config(config.charset)

    try:
        report = prepare_iam(ensure_dir(out), charset)
    except Exception as exc:  # network, missing pyarrow, ...
        logger.error("Could not download/prepare IAM: %s", exc)
        return 1

    report["combined_iam"] = build_combined(base, out, combined_out)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Done. Report: %s", out / "prepare_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
