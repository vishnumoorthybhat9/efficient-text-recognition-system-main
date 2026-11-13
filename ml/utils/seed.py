"""Reproducibility helpers.

The SimCLR-vs-scratch comparison is only meaningful if both arms see the same
data in the same order, so seeding is part of the experimental protocol rather
than a convenience.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["seed_everything", "worker_init_fn"]


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Args:
        seed: the seed value.
        deterministic: when true, ask cuDNN for deterministic kernels.  This
            costs roughly 10-25% throughput, so it is off by default and
            enabled explicitly for the comparison runs.

    Returns:
        The seed that was applied, so callers can log it.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.debug("Seeded everything with %d (deterministic=%s)", seed, deterministic)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that gives each worker a distinct seed."""
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
