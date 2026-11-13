"""Machine-learning library for the text-recognition project.

Layout
------
``ml.config``        YAML configuration loading with ``extends`` inheritance
``ml.text``          character sets and label encoding
``ml.preprocessing`` OpenCV enhancement pipeline and augmentation
``ml.datasets``      synthetic rendering, external importers, manifests, Datasets
``ml.models``        CNN encoders, attention, the CRNN recogniser, SimCLR, TrOCR
``ml.decoding``      CTC greedy and prefix beam search
``ml.engine``        training loops
``ml.evaluation``    CER/WER metrics and benchmarking
``ml.inference``     segmentation, predictors, model registry
``ml.utils``         device, logging, paths, timing, image IO, serialisation

Importing this package sets two environment defaults *before* anything can
import ``transformers`` or ``huggingface_hub``, because both read these at
import time and ignore later changes.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

__all__ = ["__version__"]

__version__ = "1.0.0"

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent


def _configure_huggingface_cache() -> None:
    """Keep the Hugging Face cache inside the project directory.

    TrOCR's weights are a few hundred megabytes and the default cache lives
    under the user profile on ``C:``. The machine this was developed on had
    about 6 GB free there, so the download would have been a real risk.
    Pointing the cache at ``<project>/.cache/huggingface`` puts it on the same
    drive as the rest of the project's artefacts.

    ``setdefault`` throughout: an explicitly configured cache always wins.
    """
    _os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))
    # Windows without Developer Mode cannot create the symlinks the cache
    # prefers. It falls back correctly; the warning is just noise.
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


_configure_huggingface_cache()
