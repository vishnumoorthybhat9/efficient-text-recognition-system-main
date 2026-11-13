"""Generate the synthetic OCR corpus.

Renders labeled text-line images from the typefaces installed on this machine,
so the project is trainable with no dataset download.  See
:mod:`ml.datasets.synthetic` for what this data is and is not.

Usage
-----
::

    # ~22k images, the default used for the reported experiments (~2 min)
    python scripts/make_synthetic_data.py

    # ~2.9k images, enough to smoke-test the whole pipeline (~20 s)
    python scripts/make_synthetic_data.py --preset quick

    # custom
    python scripts/make_synthetic_data.py --train 20000 --unlabeled 30000

Outputs (under ``data/processed/synthetic/`` by default)::

    images/            rendered PNGs
    unlabeled/         rendered PNGs for self-supervised pretraining
    train.tsv          <relative path>\\t<label>
    val.tsv
    test.tsv
    unlabeled.txt      one relative path per line, no labels
    dataset_info.json  provenance: counts, seed, fonts used, config

Experimental hygiene
--------------------
The unlabeled pool is rendered from its own independent text sample and
written to its own directory, so **no image file is shared between the
unlabeled pool and the validation or test splits**.  Pretraining therefore
cannot have seen an evaluation image.  Lexical overlap between splits does
exist -- the lexicon is finite, and real corpora reuse words too.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Make the project importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import load_config  # noqa: E402
from ml.datasets.lexicon import TextSampler, load_lexicon  # noqa: E402
from ml.datasets.manifest import (  # noqa: E402
    validate_manifest,
    write_manifest,
    write_unlabeled_manifest,
)
from ml.datasets.synthetic import SyntheticRenderer, TextKind, discover_fonts  # noqa: E402
from ml.text.charset import Charset  # noqa: E402
from ml.utils.image_io import write_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import PROJECT_ROOT, ensure_dir, relative_to_root  # noqa: E402

logger = logging.getLogger("make_synthetic_data")


@dataclass(slots=True)
class Preset:
    """A named set of split sizes."""

    train: int
    val: int
    test: int
    unlabeled: int


PRESETS: dict[str, Preset] = {
    # Fast end-to-end check of the whole pipeline.
    "quick": Preset(train=1000, val=200, test=200, unlabeled=1500),
    # Used for the numbers reported in the README.
    "default": Preset(train=8000, val=1000, test=1000, unlabeled=12000),
    # Better accuracy, roughly 4x the generation time and disk.
    "large": Preset(train=30000, val=3000, test=3000, unlabeled=45000),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a synthetic OCR dataset from installed system fonts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="default.yaml", help="config file (under configs/)")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="default", help="split sizes to use"
    )
    parser.add_argument("--train", type=int, default=None, help="override train split size")
    parser.add_argument("--val", type=int, default=None, help="override validation split size")
    parser.add_argument("--test", type=int, default=None, help="override test split size")
    parser.add_argument(
        "--unlabeled", type=int, default=None, help="override unlabeled pool size"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: data/processed/synthetic from the config)",
    )
    parser.add_argument(
        "--handwritten-ratio",
        type=float,
        default=0.35,
        help="fraction of samples rendered with a handwriting typeface",
    )
    parser.add_argument("--height", type=int, default=64, help="rendered image height in pixels")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument("--lexicon", default=None, help="optional word list, one word per line")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing dataset directory"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _resolve_counts(args: argparse.Namespace) -> Preset:
    preset = PRESETS[args.preset]
    return Preset(
        train=args.train if args.train is not None else preset.train,
        val=args.val if args.val is not None else preset.val,
        test=args.test if args.test is not None else preset.test,
        unlabeled=args.unlabeled if args.unlabeled is not None else preset.unlabeled,
    )


def _max_label_length(charset: Charset, width: int) -> int:
    """Longest label CTC can emit, given the encoder's time-step budget.

    The encoder downsamples width by 4 and the final valid convolution removes
    one column, so ``T = width // 4 - 1``.  CTC needs a blank between repeated
    characters, so the safe cap is a little under ``T``; 60% leaves comfortable
    headroom and still permits long lines.
    """
    time_steps = max(1, width // 4 - 1)
    return max(4, int(time_steps * 0.6))


def _kinds_for(count: int, handwritten_ratio: float, seed: int) -> list[TextKind]:
    """Deterministic, exactly-proportioned mix of printed and handwritten."""
    import numpy as np

    handwritten_count = int(round(count * max(0.0, min(1.0, handwritten_ratio))))
    kinds: list[TextKind] = ["handwritten"] * handwritten_count + ["printed"] * (
        count - handwritten_count
    )
    np.random.default_rng(seed).shuffle(kinds)  # type: ignore[arg-type]
    return kinds


def _render_split(
    *,
    name: str,
    count: int,
    texts: Sequence[str],
    kinds: Sequence[TextKind],
    renderer: SyntheticRenderer,
    image_dir: Path,
    prefix: str,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Render one split to disk.

    Returns:
        ``(manifest_rows, font_usage)`` where a row is
        ``(path_relative_to_the_manifest, label)``.
    """
    ensure_dir(image_dir)
    rows: list[tuple[str, str]] = []
    font_usage: dict[str, int] = {}
    failures = 0
    started = time.perf_counter()

    for index in range(count):
        text = texts[index]
        kind = kinds[index]
        try:
            image, style = renderer.render(text, kind=kind)
        except Exception as exc:  # a single bad glyph must not abort the run
            failures += 1
            logger.debug("Failed to render %r (%s): %s", text, kind, exc)
            continue

        filename = f"{prefix}_{index:07d}.png"
        write_image(image_dir / filename, image)

        # Path recorded relative to the manifest's own directory.
        rows.append((f"{image_dir.name}/{filename}", text))
        font_usage[style.font_name] = font_usage.get(style.font_name, 0) + 1

        if (index + 1) % 2000 == 0:
            rate = (index + 1) / max(1e-6, time.perf_counter() - started)
            logger.info("  %s: %d/%d rendered (%.0f img/s)", name, index + 1, count, rate)

    elapsed = time.perf_counter() - started
    logger.info(
        "%s: %d image(s) in %.1fs (%.0f img/s)%s",
        name,
        len(rows),
        elapsed,
        len(rows) / max(1e-6, elapsed),
        f", {failures} render failure(s)" if failures else "",
    )
    return rows, font_usage


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = load_config(args.config)
    charset = Charset.from_config(config.charset)
    counts = _resolve_counts(args)

    output_dir = Path(args.out) if args.out else Path(str(config.get("data.processed_dir", "data/processed"))) / "synthetic"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        logger.error(
            "%s already exists and is not empty. Re-run with --force to overwrite it.",
            relative_to_root(output_dir),
        )
        return 1

    ensure_dir(output_dir)

    # --- fonts --------------------------------------------------------------
    catalogue = discover_fonts()
    if catalogue.used_fallback:
        logger.warning(
            "Rendering with Pillow's built-in font only. A model trained on this will "
            "not generalise to other typefaces; install some TrueType fonts for a "
            "meaningful dataset."
        )

    renderer = SyntheticRenderer(catalogue, height=args.height, seed=args.seed)

    # --- text ---------------------------------------------------------------
    words = load_lexicon(args.lexicon)
    max_length = _max_label_length(charset, int(config.get("image.width", 256)))
    logger.info(
        "Charset: %d classes (incl. blank) | max label length: %d characters",
        charset.num_classes,
        max_length,
    )

    total_labeled = counts.train + counts.val + counts.test
    logger.info(
        "Generating %d labeled + %d unlabeled image(s) into %s",
        total_labeled,
        counts.unlabeled,
        relative_to_root(output_dir),
    )

    # One sampler for the labeled splits; a separately seeded sampler for the
    # unlabeled pool, so the two text streams are independent.
    labeled_sampler = TextSampler(
        words, charset_characters=charset.characters, max_length=max_length, seed=args.seed
    )
    unlabeled_sampler = TextSampler(
        words,
        charset_characters=charset.characters,
        max_length=max_length,
        seed=args.seed + 99991,
    )

    labeled_texts = labeled_sampler.sample_many(total_labeled)
    splits = {
        "train": labeled_texts[: counts.train],
        "val": labeled_texts[counts.train : counts.train + counts.val],
        "test": labeled_texts[counts.train + counts.val :],
    }

    # --- render labeled splits ---------------------------------------------
    image_dir = output_dir / "images"
    font_usage: dict[str, int] = {}
    manifests: dict[str, Path] = {}
    offset = 0

    for split_name, split_texts in splits.items():
        if not split_texts:
            logger.warning("Split %r is empty; skipping", split_name)
            continue

        kinds = _kinds_for(len(split_texts), args.handwritten_ratio, args.seed + offset)
        rows, usage = _render_split(
            name=split_name,
            count=len(split_texts),
            texts=split_texts,
            kinds=kinds,
            renderer=renderer,
            image_dir=image_dir,
            prefix=split_name,
        )
        offset += len(split_texts)
        for font, n in usage.items():
            font_usage[font] = font_usage.get(font, 0) + n

        manifests[split_name] = write_manifest(
            output_dir / f"{split_name}.tsv",
            rows,
            header_comment=(
                f"Synthetic OCR dataset -- {split_name} split\n"
                f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} | seed={args.seed} | "
                f"{len(rows)} samples\n"
                "format: <relative image path>\\t<label>"
            ),
        )

    # --- render the unlabeled pool -----------------------------------------
    if counts.unlabeled > 0:
        unlabeled_dir = output_dir / "unlabeled"
        unlabeled_texts = unlabeled_sampler.sample_many(counts.unlabeled)
        kinds = _kinds_for(counts.unlabeled, args.handwritten_ratio, args.seed + 7777)

        rows, usage = _render_split(
            name="unlabeled",
            count=counts.unlabeled,
            texts=unlabeled_texts,
            kinds=kinds,
            renderer=renderer,
            image_dir=unlabeled_dir,
            prefix="unlab",
        )
        for font, n in usage.items():
            font_usage[font] = font_usage.get(font, 0) + n

        write_unlabeled_manifest(
            output_dir / "unlabeled.txt",
            [path for path, _ in rows],
            header_comment=(
                "Unlabeled pool for SimCLR self-supervised pretraining.\n"
                "Rendered from an independent text sample; shares no image file with "
                "the val or test splits.\n"
                f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} | seed={args.seed + 99991}"
            ),
        )

    # --- demo images --------------------------------------------------------
    demo_dir = ensure_dir(config.get("data.demo_dir", "data/demo"))
    demo_entries: list[dict[str, str]] = []
    demo_texts = [
        ("Efficient Text Recognition", "printed"),
        ("Deep Learning", "printed"),
        ("Self Supervised Learning", "handwritten"),
        ("Invoice Total 1,250.00", "printed"),
        ("INV-44718", "printed"),
        ("handwritten sample text", "handwritten"),
        ("22 Mar 2026", "handwritten"),
        ("Department of ISE", "printed"),
    ]
    for index, (text, kind) in enumerate(demo_texts):
        try:
            image, _ = renderer.render(text, kind=kind)  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("Could not render demo image %r: %s", text, exc)
            continue
        filename = f"demo_{index:02d}_{kind}.png"
        write_image(demo_dir / filename, image)
        demo_entries.append({"file": filename, "text": text, "kind": kind})

    (demo_dir / "demo_index.json").write_text(
        json.dumps(demo_entries, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d demo image(s) to %s", len(demo_entries), relative_to_root(demo_dir))

    # --- validate ----------------------------------------------------------
    validations: dict[str, dict[str, object]] = {}
    all_ok = True
    for split_name, manifest_path in manifests.items():
        report = validate_manifest(manifest_path, charset, max_label_length=max_length)
        validations[split_name] = report.to_dict()
        logger.info("%s", report.summary())
        all_ok = all_ok and report.ok

    # --- provenance --------------------------------------------------------
    info = {
        "name": "synthetic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "scripts/make_synthetic_data.py",
        "seed": args.seed,
        "preset": args.preset,
        "counts": {
            "train": counts.train,
            "val": counts.val,
            "test": counts.test,
            "unlabeled": counts.unlabeled,
        },
        "render_height": args.height,
        "handwritten_ratio": args.handwritten_ratio,
        "max_label_length": max_length,
        "charset": charset.to_dict(),
        "fonts": catalogue.to_dict(),
        "font_usage": dict(sorted(font_usage.items(), key=lambda kv: -kv[1])),
        "validation": validations,
        "notes": (
            "Synthetic data rendered from system typefaces. 'handwritten' samples use "
            "script typefaces with per-character baseline jitter, which is more regular "
            "than real cursive: use scripts/prepare_dataset.py with IAM for handwriting "
            "results that are comparable to the literature."
        ),
    }
    info_path = output_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Dataset provenance written to %s", relative_to_root(info_path))
    logger.info(
        "Done. Next: python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml"
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
