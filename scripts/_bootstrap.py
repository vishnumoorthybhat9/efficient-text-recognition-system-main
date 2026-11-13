"""Shared start-up code for the command-line scripts.

Every script under ``scripts/`` needs the same four things: the project root on
``sys.path`` (so it can be run as ``python scripts/foo.py`` from anywhere),
logging configured, a config loaded with CLI overrides applied, and a device
resolved.  Putting it here keeps each script about its own job.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import Config, load_config  # noqa: E402
from ml.utils.device import describe_hardware, resolve_device  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.seed import seed_everything  # noqa: E402

__all__ = [
    "PROJECT_ROOT",
    "add_common_arguments",
    "bootstrap",
    "BootstrapResult",
    "log_hardware",
]

logger = logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every training/evaluation script accepts."""
    parser.add_argument("--config", required=True, help="config file (absolute, or under configs/)")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="override train.learning_rate")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers. 0 is safest on Windows; 4 is usually faster.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap the number of training samples (overrides data.max_train_samples)",
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="disable mixed precision even on a GPU"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic cuDNN kernels (slower, exactly reproducible)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="override train.checkpoint_dir"
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--run-name", default=None, help="override the run name used for logs and checkpoints"
    )
    return parser


class BootstrapResult:
    """What :func:`bootstrap` produced."""

    __slots__ = ("config", "device", "seed", "args")

    def __init__(self, config: Config, device: Any, seed: int, args: argparse.Namespace) -> None:
        self.config = config
        self.device = device
        self.seed = seed
        self.args = args


def _overrides_from(args: argparse.Namespace) -> dict[str, Any]:
    """Turn CLI flags into a config override mapping."""
    train: dict[str, Any] = {}
    data: dict[str, Any] = {}

    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.lr is not None:
        train["learning_rate"] = args.lr
    if args.seed is not None:
        train["seed"] = args.seed
    if args.workers is not None:
        train["num_workers"] = args.workers
    if getattr(args, "no_amp", False):
        train["amp"] = False
    if getattr(args, "checkpoint_dir", None):
        train["checkpoint_dir"] = args.checkpoint_dir
    if getattr(args, "max_samples", None) is not None:
        data["max_train_samples"] = args.max_samples

    overrides: dict[str, Any] = {}
    if train:
        overrides["train"] = train
    if data:
        overrides["data"] = data
    return overrides


def bootstrap(args: argparse.Namespace, *, log_file: str | Path | None = None) -> BootstrapResult:
    """Configure logging, load the config, seed everything, resolve the device."""
    setup_logging(args.log_level, log_file=log_file)

    config = load_config(args.config, overrides=_overrides_from(args))
    if getattr(args, "run_name", None):
        config = config.merged_with({"name": args.run_name})

    seed = int(config.get("train.seed", 42))
    seed_everything(seed, deterministic=bool(getattr(args, "deterministic", False)))

    device = resolve_device(args.device or "auto")

    logger.info("Config: %s", config.source)
    logger.info("Run name: %s | seed: %d | device: %s", config.get("name", "unnamed"), seed, device)
    return BootstrapResult(config=config, device=device, seed=seed, args=args)


def log_hardware(device: Any) -> dict[str, Any]:
    """Log, and return, the hardware report recorded in the training history."""
    info = describe_hardware("auto")
    if info.gpus:
        gpu = info.gpus[0]
        logger.info(
            "GPU: %s (%d MB, compute %s) | CUDA %s | torch %s",
            gpu.name,
            gpu.total_memory_mb,
            gpu.capability,
            info.cuda_version,
            info.torch_version,
        )
    else:
        logger.info(
            "No CUDA device: running on CPU (%s, %d core(s)) | torch %s",
            info.cpu_name,
            info.cpu_count,
            info.torch_version,
        )
    if device.type == "cpu" and info.cuda_available:
        logger.warning("A CUDA device is available but the run was forced onto the CPU.")
    return info.to_dict()
