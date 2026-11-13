"""TrOCR integration -- the Transformer-based OCR arm of the comparison.

TrOCR (Li et al., arXiv:2109.10282, cited in the synopsis) is an encoder-decoder
transformer: a ViT image encoder feeding a text decoder that generates the
transcription token by token.  It contrasts with the project's own recogniser
in exactly the way the comparison needs:

===================  ==========================  =============================
                     CRNN + BiLSTM + Attention   TrOCR
===================  ==========================  =============================
Alignment            CTC, monotonic              autoregressive generation
Language modelling   none                        implicit, in the decoder
Supervision          trained here, on our data   pretrained on ~700M lines
Parameters           ~8 M                        ~62 M (small) upward
===================  ==========================  =============================

Honesty note
------------
TrOCR weights are **downloaded, pretrained, third-party** weights.  They were
not trained by this project and its numbers must never be presented as a
result of the self-supervised experiment.  Every model this project exposes
carries a ``provenance`` field for precisely this reason, and the UI shows it.

Loading is lazy and failure is non-fatal: on a machine with no network and an
empty cache the rest of the application keeps working, and this model simply
reports itself unavailable with the reason.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from ml.preprocessing import ops

__all__ = ["TrOCRRecognizer", "TrOCRUnavailable", "TrOCRPrediction"]

logger = logging.getLogger(__name__)


class TrOCRUnavailable(RuntimeError):
    """Raised when the TrOCR model cannot be loaded, with the reason."""


@dataclass(slots=True)
class TrOCRPrediction:
    """One generated transcription."""

    text: str
    confidence: float = 0.0
    num_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "num_tokens": self.num_tokens,
        }


@dataclass(slots=True)
class _LoadState:
    """Tracks the outcome of the lazy load so it is attempted only once."""

    attempted: bool = False
    error: str | None = None
    loaded_at: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class TrOCRRecognizer:
    """Lazy wrapper around a Hugging Face TrOCR checkpoint.

    Args:
        model_id: Hugging Face id, e.g. ``microsoft/trocr-small-printed``.
            The *small* variants are the default: ``trocr-base`` needs ~1.4 GB
            of VRAM for inference, which does not leave room alongside the
            project's own model on a 4 GB card.
        device: target device.
        max_new_tokens: generation cap.  TrOCR has no natural stopping signal
            on a non-text image and will happily hallucinate until the cap.
        num_beams: 1 for greedy decoding; higher is slower but a little better.
        no_repeat_ngram_size: forbid repeating an n-gram during generation.
            Autoregressive decoders degenerate into loops on out-of-domain
            input -- on cropped form fields this model emitted
            "000 000 000 ..." until it hit the token cap. Blocking repeated
            trigrams is the standard remedy and is applied here so the
            baseline is judged on its recognition, not on a decoding artefact.
        allow_download: when false, only an already-cached model is used.

    >>> recogniser = TrOCRRecognizer("microsoft/trocr-small-printed", device)
    >>> recogniser.is_available        # does not load anything
    True
    >>> recogniser.predict([image])    # loads on first use
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        max_new_tokens: int = 48,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 3,
        allow_download: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = max(1, int(num_beams))
        self.no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
        self.allow_download = bool(allow_download)

        self._processor: Any = None
        self._model: Any = None
        self._state = _LoadState()
        # Loading is not thread safe and FastAPI serves requests concurrently.
        self._lock = threading.Lock()

    # -- availability -------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._state.error

    @property
    def is_available(self) -> bool:
        """Whether a load is plausible, without actually performing one.

        Returns False only when ``transformers`` is missing or a previous load
        already failed.  A model that merely has not been downloaded yet still
        counts as available, because the download will be attempted on use.
        """
        if self.is_loaded:
            return True
        if self._state.error is not None:
            return False
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _offline_requested(self) -> bool:
        if not self.allow_download:
            return True
        return os.environ.get("TRANSFORMERS_OFFLINE", "0").strip().lower() in ("1", "true", "yes")

    # -- loading ------------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load the processor and model, once.

        Raises:
            TrOCRUnavailable: loading failed; the message states why and what
                to do about it.
        """
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:  # another thread won the race
                return
            if self._state.error is not None:
                raise TrOCRUnavailable(self._state.error)

            self._state.attempted = True
            try:
                self._load()
            except TrOCRUnavailable:
                raise
            except Exception as exc:
                self._state.error = self._explain(exc)
                logger.warning("TrOCR %s unavailable: %s", self.model_id, self._state.error)
                raise TrOCRUnavailable(self._state.error) from exc

    def _load(self) -> None:
        import time

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise TrOCRUnavailable(
                "The `transformers` package is not installed, so the Transformer-OCR "
                "model is unavailable. Install it with:  pip install transformers"
            ) from exc

        offline = self._offline_requested()
        logger.info(
            "Loading TrOCR %s onto %s (%s)",
            self.model_id,
            self.device,
            "cache only" if offline else "downloading if not cached",
        )

        started = time.perf_counter()
        kwargs: dict[str, Any] = {"local_files_only": offline}

        self._processor = TrOCRProcessor.from_pretrained(self.model_id, **kwargs)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_id, **kwargs)

        model.eval()
        model.to(self.device)
        self._model = model

        self._state.loaded_at = time.time()
        self._state.extras = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "load_seconds": round(time.perf_counter() - started, 2),
            "offline": offline,
        }
        logger.info(
            "TrOCR %s ready: %.1f M parameters, loaded in %.1fs",
            self.model_id,
            self._state.extras["parameters"] / 1e6,
            self._state.extras["load_seconds"],
        )

    def _explain(self, exc: Exception) -> str:
        """Turn a loading exception into something actionable."""
        text = str(exc)
        lowered = text.lower()

        if "local_files_only" in lowered or "not found in the cached files" in lowered or "offline" in lowered:
            return (
                f"TrOCR model {self.model_id!r} is not in the local cache and downloads are "
                "disabled. Either allow downloads (unset TRANSFORMERS_OFFLINE / set "
                "trocr.allow_download: true) or pre-download it with:\n"
                f"  python -c \"from transformers import VisionEncoderDecoderModel as M; "
                f"M.from_pretrained('{self.model_id}')\""
            )
        if any(marker in lowered for marker in ("connection", "network", "timed out", "resolve", "proxy")):
            return (
                f"Could not reach huggingface.co to download {self.model_id!r}. The rest of "
                f"the application works offline; only Transformer OCR needs this. ({text[:160]})"
            )
        if "out of memory" in lowered:
            return (
                f"Not enough memory to load {self.model_id!r} on {self.device}. Try a smaller "
                "variant (microsoft/trocr-small-printed) or set OCR_DEVICE=cpu."
            )
        return f"Could not load {self.model_id!r}: {text[:300]}"

    def unload(self) -> None:
        """Release the model, freeing VRAM.

        Useful on a 4 GB card when switching between models mid-session.
        """
        with self._lock:
            self._model = None
            self._processor = None
            self._state = _LoadState()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded TrOCR %s", self.model_id)

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: Sequence[np.ndarray]) -> list[TrOCRPrediction]:
        """Recognise a batch of text-line images.

        Args:
            images: gray or BGR ``uint8`` arrays.  TrOCR's processor handles
                resizing to 384x384 itself, so no shaping is needed here --
                it must *not* receive the 32x256 canvas the CRNN uses.

        Returns:
            One :class:`TrOCRPrediction` per input, in order.

        Raises:
            TrOCRUnavailable: the model could not be loaded.
        """
        if not images:
            return []

        self.ensure_loaded()
        assert self._model is not None and self._processor is not None

        # The processor expects RGB. cv2.cvtColor is used rather than a
        # `[..., ::-1]` slice: the slice produces a negative-stride view, and
        # torch refuses to build a tensor from one.
        rgb_images = [cv2.cvtColor(ops.to_bgr(image), cv2.COLOR_BGR2RGB) for image in images]

        inputs = self._processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if self.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        generated = self._model.generate(pixel_values, **generate_kwargs)

        sequences = generated.sequences
        texts = self._processor.batch_decode(sequences, skip_special_tokens=True)
        confidences = self._confidences(generated, sequences)

        results: list[TrOCRPrediction] = []
        for index, text in enumerate(texts):
            cleaned = " ".join(str(text).split())
            token_count = int((sequences[index] != self._pad_id()).sum().item())
            results.append(
                TrOCRPrediction(
                    text=cleaned,
                    confidence=float(confidences[index]),
                    num_tokens=token_count,
                )
            )
        return results

    def _pad_id(self) -> int:
        tokenizer = getattr(self._processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        return int(pad) if pad is not None else -1

    def _confidences(self, generated: Any, sequences: torch.Tensor) -> list[float]:
        """Derive a per-sequence confidence from the generation scores.

        Beam search already reports a length-normalised sequence score.  For
        greedy decoding the mean log-probability of the chosen tokens is
        computed here, which is the comparable quantity.  Returns zeros rather
        than failing if the transformers version does not expose scores.
        """
        batch = sequences.size(0)

        beam_scores = getattr(generated, "sequences_scores", None)
        if beam_scores is not None:
            return [float(np.clip(np.exp(float(s)), 0.0, 1.0)) for s in beam_scores]

        scores = getattr(generated, "scores", None)
        if not scores:
            return [0.0] * batch

        pad_id = self._pad_id()
        # `sequences` includes the decoder start token, which has no score.
        generated_tokens = sequences[:, sequences.size(1) - len(scores) :]

        totals = torch.zeros(batch, dtype=torch.float32, device=sequences.device)
        counts = torch.zeros(batch, dtype=torch.float32, device=sequences.device)

        for step, step_scores in enumerate(scores):
            log_probs = torch.log_softmax(step_scores.float(), dim=-1)
            tokens = generated_tokens[:, step]
            chosen = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
            valid = (tokens != pad_id).float()
            # `chosen` can be -inf for a masked token; zero it out before summing.
            totals += torch.where(valid.bool(), chosen, torch.zeros_like(chosen)) * valid
            counts += valid

        mean_log_prob = totals / counts.clamp(min=1.0)
        return [float(np.clip(np.exp(float(v)), 0.0, 1.0)) for v in mean_log_prob]

    # -- introspection ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Metadata for the API's model list."""
        return {
            "model_id": self.model_id,
            "architecture": "TrOCR (ViT encoder + text decoder)",
            "loaded": self.is_loaded,
            "available": self.is_available,
            "error": self._state.error,
            "device": str(self.device),
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "provenance": "pretrained_third_party",
            **self._state.extras,
        }
