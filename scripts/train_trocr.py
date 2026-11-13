"""Fine-tune the Transformer OCR model (TrOCR) on this project's own data.

Until now the Transformer arm of the comparison used Microsoft's pretrained
weights unchanged, so its numbers measured transfer to an unseen domain rather
than what a Transformer recogniser achieves on *our* data. This script closes
that gap: it fine-tunes TrOCR on exactly the labeled lines the final CNN model
(``crnn_full_v2``) is trained on -- synthetic + Kaggle handwriting + IAM
(``data/processed/combined_iam``) -- so the two compare like for like.

Starting point
--------------
``microsoft/trocr-small-handwritten`` (~62 M parameters): a ViT image encoder
and a Transformer text decoder, pretrained by its authors on printed and
handwritten text. The *small* variant is the only one that trains in 4 GB of
VRAM at a usable batch size; ``trocr-base`` (334 M) does not fit.

Memory plan for a 4 GB card
---------------------------
* bf16 autocast (Ampere and later), else fp16 with loss scaling.
* micro-batch 8, gradient accumulation 4 -> effective batch 32.
* if the first steps run out of memory, gradient checkpointing is switched on
  automatically and training continues.

Resuming
--------
Every epoch saves ``last/`` (weights + optimizer + scheduler state). Running
the script again with ``--resume`` continues from there, and does nothing if
training already finished -- so an interrupted overnight run (sleep, power
cut) loses at most one epoch.

Output (``artifacts/checkpoints/trocr_finetuned/``)
----------------------------------------------------
``model/``        best weights + processor, loadable with ``from_pretrained``
``last/``         resume state
``history.json``  per-epoch metrics, read by the app's model registry

Usage
-----
::

    python scripts/train_trocr.py                  # full run
    python scripts/train_trocr.py --resume         # continue / skip if done
    python scripts/train_trocr.py --epochs 1 --max-samples 200 --val-samples 50   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml  # noqa: E402,F401  (sets HF_HOME before transformers is imported)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ml.datasets.manifest import ManifestEntry, read_manifest  # noqa: E402
from ml.evaluation.metrics import evaluate_predictions  # noqa: E402
from ml.utils.image_io import read_image  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.paths import ensure_dir, relative_to_root, resolve_path  # noqa: E402
from ml.utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger("train_trocr")

RUN_NAME = "trocr_finetuned"


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on the project's labeled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--train-manifest", default="data/processed/combined_iam/train.tsv")
    parser.add_argument("--val-manifest", default="data/processed/combined_iam/val.tsv")
    parser.add_argument("--output-dir", default=f"artifacts/checkpoints/{RUN_NAME}")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8, help="micro-batch per step")
    parser.add_argument("--grad-accum", type=int, default=4, help="steps per optimizer update")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-label-tokens", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=0, help="cap training samples (0 = all)")
    parser.add_argument("--val-samples", type=int, default=1000, help="validation subset size (0 = all)")
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (0 = safest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-checkpointing", action="store_true", help="start with it on")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", help="continue from last/ if present")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class LineDataset(Dataset):
    """Labeled line images -> (RGB uint8 image, token ids)."""

    def __init__(self, entries: Sequence[ManifestEntry], tokenizer: Any, eos_id: int, max_tokens: int):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.eos_id = int(eos_id)
        self.max_tokens = int(max_tokens)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[int]]:
        entry = self.entries[index]
        gray = read_image(entry.path, grayscale=True)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Target format used by TrOCR's own pretraining: text tokens then </s>;
        # the decoder start token is supplied by the model when shifting.
        ids = self.tokenizer(entry.label, add_special_tokens=False).input_ids
        ids = list(ids)[: self.max_tokens - 1] + [self.eos_id]
        return rgb, ids


class Collate:
    """Batch images through the TrOCR processor and pad labels with -100."""

    def __init__(self, image_processor: Any):
        self.image_processor = image_processor

    def __call__(self, batch: list[tuple[np.ndarray, list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = [item[0] for item in batch]
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values
        longest = max(len(item[1]) for item in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for row, (_, ids) in enumerate(batch):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return pixel_values, labels


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


def amp_settings(device: torch.device, disabled: bool) -> tuple[bool, torch.dtype]:
    if disabled or device.type != "cuda":
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def lr_lambda_factory(total_updates: int, warmup_updates: int):
    def factor(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(0.0, 1.0 - progress)  # linear decay, the standard for TrOCR

    return factor


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


@torch.no_grad()
def validate(model: Any, processor: Any, entries: Sequence[ManifestEntry], device: torch.device,
             batch_size: int, max_new_tokens: int, amp: bool, amp_dtype: torch.dtype) -> dict[str, float]:
    """Greedy-decode the validation subset and score it with the project's metrics."""
    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        images = [cv2.cvtColor(read_image(e.path, grayscale=True), cv2.COLOR_GRAY2RGB) for e in chunk]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            generated = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=1)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(" ".join(t.split()) for t in texts)
        references.extend(e.label for e in chunk)
    metrics, samples = evaluate_predictions(references, predictions, return_samples=True)
    model.train()
    examples = [
        {"reference": s.reference, "prediction": s.prediction}
        for s in samples[:5]
    ] if samples else []
    return {
        "val_cer": float(metrics.cer),
        "val_wer": float(metrics.wer),
        "val_accuracy": float(metrics.sequence_accuracy),
        "val_cer_normalized": float(metrics.cer_normalized),
        "_examples": examples,  # type: ignore[dict-item]
    }


def save_model(model: Any, processor: Any, directory: Path) -> None:
    tmp = directory.with_name(directory.name + "_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    processor.save_pretrained(tmp)
    if directory.exists():
        shutil.rmtree(directory)
    tmp.rename(directory)


def write_history(path: Path, history: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("INFO", log_file=ensure_dir("artifacts/logs") / f"train_{RUN_NAME}.log")
    seed_everything(args.seed)
    random.seed(args.seed)

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = resolve_device(args.device)
    amp, amp_dtype = amp_settings(device, args.no_amp)
    output_dir = ensure_dir(resolve_path(args.output_dir) or Path(args.output_dir))
    best_dir, last_dir = output_dir / "model", output_dir / "last"
    history_path = output_dir / "history.json"

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%d MB) | amp=%s %s", props.name, props.total_memory // 2**20, amp, amp_dtype)
    else:
        logger.warning("No GPU: training on CPU will be extremely slow (days).")

    # --- resume state ------------------------------------------------------
    history: dict[str, Any] = {}
    start_epoch = 1
    resume_state: dict[str, Any] | None = None
    if args.resume and (last_dir / "trainer_state.pt").is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        resume_state = torch.load(last_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > args.epochs:
            logger.info("Already trained for %d epoch(s); nothing to do.", args.epochs)
            return 0
        logger.info("Resuming from epoch %d", start_epoch)

    source = str(last_dir) if resume_state else args.base_model
    logger.info("Loading %s", source)
    processor = TrOCRProcessor.from_pretrained(source)
    model = VisionEncoderDecoderModel.from_pretrained(source)
    tokenizer = processor.tokenizer

    # The decoder start token is kept exactly as the authors pretrained it;
    # only fill gaps so that label shifting and generation agree.
    # (also check the generation config: some checkpoints keep the ids there)
    gen0 = model.generation_config
    for key in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
        if getattr(model.config, key, None) is None and getattr(gen0, key, None) is not None:
            setattr(model.config, key, getattr(gen0, key))
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    gen = model.generation_config
    gen.decoder_start_token_id = model.config.decoder_start_token_id
    gen.pad_token_id = model.config.pad_token_id
    gen.eos_token_id = model.config.eos_token_id
    gen.max_new_tokens = args.max_label_tokens
    gen.no_repeat_ngram_size = 0  # a trained model should not need it; it would hurt "AA", "1111"
    model.to(device)
    parameters = int(sum(p.numel() for p in model.parameters()))
    logger.info("Model: %.1f M parameters", parameters / 1e6)

    use_checkpointing = bool(args.grad_checkpointing or (resume_state or {}).get("grad_checkpointing"))
    if use_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: on")

    # --- data --------------------------------------------------------------
    train_entries = read_manifest(resolve_path(args.train_manifest), skip_missing=True)
    val_entries = read_manifest(resolve_path(args.val_manifest), skip_missing=True)
    if args.max_samples and args.max_samples > 0:
        rng = random.Random(args.seed)
        train_entries = rng.sample(train_entries, min(args.max_samples, len(train_entries)))
    if args.val_samples and args.val_samples < len(val_entries):
        # fixed subset, stratified only by order of a seeded shuffle
        rng = random.Random(args.seed)
        val_entries = rng.sample(val_entries, args.val_samples)
    logger.info("Data: %d train, %d val", len(train_entries), len(val_entries))

    dataset = LineDataset(train_entries, tokenizer, tokenizer.eos_token_id, args.max_label_tokens)
    generator = torch.Generator().manual_seed(args.seed + start_epoch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=Collate(processor.image_processor),
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
        generator=generator,
    )

    # --- optimisation ------------------------------------------------------
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    updates_per_epoch = max(1, len(loader) // args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_updates, int(total_updates * args.warmup_ratio))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)

    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])

    if not history:
        history = {
            "run_name": RUN_NAME,
            "family": "trocr",
            "base_model": args.base_model,
            "monitor": "val_cer",
            "mode": "min",
            "best_epoch": None,
            "best_value": None,
            "epochs": [],
            "epochs_completed": 0,
            "total_duration_s": 0.0,
            "device": str(device),
            "notes": {
                "train_samples": len(train_entries),
                "val_samples": len(val_entries),
                "parameters": {"total": parameters},
                "effective_batch_size": args.batch_size * args.grad_accum,
                "learning_rate": args.lr,
                "train_manifest": args.train_manifest,
                "provenance": "fine-tuned in this project from " + args.base_model,
            },
        }

    logger.info(
        "Training: %d epoch(s) x %d updates (micro-batch %d x accum %d), lr %.1e",
        args.epochs, updates_per_epoch, args.batch_size, args.grad_accum, args.lr,
    )

    # --- loop --------------------------------------------------------------
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        steps = len(loader)

        step = 0
        iterator = iter(loader)
        while step < steps:
            try:
                pixel_values, labels = next(iterator)
            except StopIteration:
                break
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    loss = model(pixel_values=pixel_values, labels=labels).loss
                scaler.scale(loss / args.grad_accum).backward()
                running += float(loss.detach())
                seen += 1
                step += 1
                # the first optimizer step allocates AdamW state, so it sits
                # inside the out-of-memory guard too
                if step % args.grad_accum == 0 or step == steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                if not (is_oom(exc) and not use_checkpointing and epoch == start_epoch and step < 50):
                    raise
                hit_oom = True
            else:
                hit_oom = False
            if hit_oom:
                # handled outside the except block so the exception's frames
                # (which pin the activations) are already released
                logger.warning("Out of GPU memory; switching gradient checkpointing on and continuing.")
                loss = pixel_values = labels = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                model.gradient_checkpointing_enable()
                use_checkpointing = True
                continue

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - step_started
                rate = step / max(elapsed, 1e-6)
                eta_min = (steps - step) / max(rate, 1e-6) / 60
                logger.info(
                    "epoch %d/%d | step %d/%d | loss %.4f | %.2f it/s | lr %.2e | epoch ETA %.0f min",
                    epoch, args.epochs, step, steps, running / max(seen, 1), rate,
                    scheduler.get_last_lr()[0], eta_min,
                )

        train_loss = running / max(seen, 1)
        logger.info("epoch %d: train_loss %.4f -- validating on %d samples", epoch, train_loss, len(val_entries))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metrics = validate(model, processor, val_entries, device, args.val_batch_size,
                           args.max_label_tokens, amp, amp_dtype)
        examples = metrics.pop("_examples", [])
        duration = time.perf_counter() - epoch_started

        record = {"epoch": epoch, "train_loss": round(train_loss, 6),
                  **{k: round(v, 6) for k, v in metrics.items()},
                  "lr": scheduler.get_last_lr()[0], "duration_s": round(duration, 1)}
        history["epochs"].append(record)
        history["epochs_completed"] = epoch
        history["total_duration_s"] = round(float(history.get("total_duration_s", 0.0)) + duration, 1)

        improved = history["best_value"] is None or metrics["val_cer"] < history["best_value"]
        if improved:
            history["best_value"] = round(metrics["val_cer"], 6)
            history["best_epoch"] = epoch
            save_model(model, processor, best_dir)
            logger.info("New best: val_cer=%.4f at epoch %d -> %s", metrics["val_cer"], epoch,
                        relative_to_root(best_dir))

        logger.info(
            "epoch %d/%d | train_loss=%.4f  val_cer=%.4f  val_wer=%.4f  val_accuracy=%.4f | %.0fs",
            epoch, args.epochs, train_loss, metrics["val_cer"], metrics["val_wer"],
            metrics["val_accuracy"], duration,
        )
        for ex in examples[:3]:
            logger.info("   ref: %-40s pred: %s", ex["reference"][:40], ex["prediction"][:60])

        # resume state last, so a crash mid-save never leaves a newer
        # trainer_state.pt pointing at older weights
        save_model(model, processor, last_dir)
        torch.save(
            {"epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "grad_checkpointing": use_checkpointing},
            last_dir / "trainer_state.pt",
        )
        write_history(history_path, history)

    logger.info("-" * 72)
    logger.info("Training complete: %s", RUN_NAME)
    logger.info("  best val CER   : %.4f (epoch %s)", history["best_value"], history["best_epoch"])
    logger.info("  training time  : %.1f min", history["total_duration_s"] / 60)
    logger.info("  model          : %s", relative_to_root(best_dir))
    logger.info("Evaluate it with:  python scripts/evaluate.py --models %s --dataset combined --split test",
                RUN_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
