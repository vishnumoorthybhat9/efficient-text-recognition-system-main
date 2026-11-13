"""End-to-end recognition: image bytes in, text out.

A predictor owns one model and performs the whole pipeline:

1. enhance the image (:mod:`ml.preprocessing.pipeline`);
2. find the text lines (:mod:`ml.inference.segmentation`);
3. recognise each line with its model;
4. assemble the text and a full timing breakdown.

Steps 1-3 are shared by every model, which is what makes the model comparison
fair: two predictors differ only in step 3.  The timing breakdown is reported
separately for exactly the same reason -- quoting a single end-to-end latency
would hide the fact that preprocessing and segmentation are common cost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ml.config import Config
from ml.datasets.ocr_dataset import ImageGeometry
from ml.decoding.ctc import decode as ctc_decode
from ml.inference.handwriting import clean_handwritten_page, detect_handwritten_lines
from ml.inference.segmentation import (
    SegmentationResult,
    TextRegion,
    crop_region,
    detect_text_lines,
    group_into_rows,
)
from ml.models.cnn_transformer_ocr import CNNTransformerOCR
from ml.models.crnn_attention import CRNNAttentionOCR
from ml.models.mamba_ocr import MambaOCR
from ml.models.trocr_wrapper import TrOCRRecognizer, TrOCRUnavailable
from ml.text.postprocess import normalize_recognised_text
from ml.preprocessing.pipeline import (
    MAX_INFERENCE_WIDTH,
    PreprocessOptions,
    PreprocessPipeline,
    prepare_model_input,
    prepare_variable_width_batch,
)
from ml.text.charset import Charset
from ml.utils.serialization import safe_load
from ml.utils.timing import stopwatch

__all__ = [
    "LinePrediction",
    "RecognitionResult",
    "RecognizeOptions",
    "BasePredictor",
    "CRNNPredictor",
    "TrOCRPredictor",
]

logger = logging.getLogger(__name__)

#: Lines are recognised in chunks of this size to bound peak VRAM on a 4 GB card.
DEFAULT_BATCH_SIZE = 16


@dataclass(slots=True)
class RecognizeOptions:
    """Per-request recognition settings."""

    preprocess: PreprocessOptions = field(default_factory=PreprocessOptions)
    #: Treat the input as a single pre-cropped line and skip segmentation.
    single_line: bool = False
    #: CTC decoding strategy; ignored by models that generate autoregressively.
    decode_strategy: str = "greedy"
    beam_width: int = 10
    max_lines: int = 200
    batch_size: int = DEFAULT_BATCH_SIZE
    #: Drop lines whose confidence falls below this (0 keeps everything).
    min_confidence: float = 0.0
    #: The page is handwritten (notebook, letter, prescription): flatten
    #: shadows, erase ruled lines and use the handwriting line detector.
    handwritten: bool = True
    #: Line detector for handwritten pages: auto | projection | doctr.
    line_detector: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preprocess"] = self.preprocess.to_dict()
        return payload


@dataclass(slots=True)
class LinePrediction:
    """One recognised text line."""

    index: int
    text: str
    confidence: float
    region: TextRegion | None = None
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "region": self.region.to_dict() if self.region else None,
            "inference_ms": round(self.inference_ms, 3),
        }


@dataclass(slots=True)
class RecognitionResult:
    """Everything one recognition request produced."""

    text: str
    lines: list[LinePrediction]
    model_id: str
    model_name: str
    confidence: float
    timings_ms: dict[str, float]
    segmentation: dict[str, Any]
    preprocessing: dict[str, Any]
    device: str
    image_size: tuple[int, int]

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "model_id": self.model_id,
            "model_name": self.model_name,
            "confidence": round(self.confidence, 6),
            "line_count": self.line_count,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "segmentation": self.segmentation,
            "preprocessing": self.preprocessing,
            "device": self.device,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
        }


class BasePredictor(ABC):
    """Shared recognition pipeline.

    Subclasses implement :meth:`_recognize_crops` and :meth:`describe`.
    """

    def __init__(self, *, model_id: str, model_name: str, device: torch.device) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.device = device

    # -- to implement -------------------------------------------------------
    @abstractmethod
    def _recognize_crops(
        self, crops: Sequence[np.ndarray], options: RecognizeOptions
    ) -> list[tuple[str, float]]:
        """Recognise cropped line images, returning ``(text, confidence)``."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Metadata for the API's model list."""

    @staticmethod
    def _assemble_text(lines: Sequence[LinePrediction], segmentation: SegmentationResult) -> str:
        """Join per-region text into one string, one line per visual row."""
        regions = [line.region for line in lines]
        if not lines or any(region is None for region in regions):
            return "\n".join(line.text for line in lines)

        rows = group_into_rows(
            [r for r in regions if r is not None],
            gutters=segmentation.column_gutters,
        )
        return "\n".join(
            " ".join(lines[index].text for index in row).strip() for row in rows
        ).strip()

    def recognize_lines(
        self,
        crops: Sequence[np.ndarray],
        options: RecognizeOptions | None = None,
    ) -> list[tuple[str, float]]:
        """Recognise already-cropped line images as one batch.

        The public entry point for benchmarking and for any caller that has
        done its own segmentation.  It skips enhancement and line detection,
        so the measured time is model inference and decoding only -- which is
        what an "inference time" column in a results table should mean.

        Args:
            crops: gray or BGR line images.
            options: decoding settings; defaults are used when omitted.

        Returns:
            ``(text, confidence)`` per crop, in order.
        """
        if not crops:
            return []
        return self._recognize_crops(crops, options or RecognizeOptions())

    def warmup(self) -> None:
        """Run a dummy inference so the first real request is not an outlier.

        The first CUDA forward pass pays for kernel autotuning and memory-pool
        setup, which can be 10-100x the steady-state latency.  Timing numbers
        shown to a user would otherwise be badly misleading.
        """
        try:
            dummy = np.full((48, 256), 255, dtype=np.uint8)
            dummy[16:32, 20:220] = 40
            self._recognize_crops([dummy], RecognizeOptions())
            logger.debug("Warmed up predictor %s", self.model_id)
        except Exception as exc:  # warm-up must never break startup
            logger.debug("Warm-up of %s failed (harmless): %s", self.model_id, exc)

    # -- pipeline -----------------------------------------------------------
    def recognize(
        self,
        image: np.ndarray,
        options: RecognizeOptions | None = None,
    ) -> RecognitionResult:
        """Run the full pipeline on *image*.

        Args:
            image: gray or BGR ``uint8`` array.
            options: per-request settings.

        Returns:
            A :class:`RecognitionResult`.

        Raises:
            ValueError: the image is empty.
        """
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Cannot recognise an empty image")

        opts = options or RecognizeOptions()
        height, width = image.shape[:2]
        timings: dict[str, float] = {}

        with stopwatch(self.device) as total_timer:
            # 1. enhancement
            with stopwatch() as pre_timer:
                pipeline = PreprocessPipeline(opts.preprocess)
                processed = pipeline.run(image)
            timings["preprocess"] = pre_timer.elapsed_ms

            # 2. segmentation (handwritten pages are cleaned first)
            page = processed.image
            handwriting_info: dict[str, Any] = {}
            if opts.handwritten and not opts.single_line:
                with stopwatch() as clean_timer:
                    cleaned = clean_handwritten_page(processed.image)
                timings["clean"] = clean_timer.elapsed_ms
                page = cleaned.image
                handwriting_info = cleaned.metadata()

            with stopwatch() as seg_timer:
                if opts.single_line:
                    segmentation = SegmentationResult(
                        regions=[
                            TextRegion(
                                0,
                                0,
                                processed.image.shape[1],
                                processed.image.shape[0],
                                index=0,
                            )
                        ],
                        used_whole_image=True,
                    )
                elif opts.handwritten:
                    segmentation, detector_used = detect_handwritten_lines(
                        page, detector=opts.line_detector, max_lines=opts.max_lines
                    )
                    handwriting_info["line_detector"] = detector_used
                else:
                    segmentation = detect_text_lines(page, max_lines=opts.max_lines)
            timings["segment"] = seg_timer.elapsed_ms

            crops = [crop_region(page, region) for region in segmentation.regions]

            # 3. recognition, in bounded batches
            with stopwatch(self.device) as inf_timer:
                outputs: list[tuple[str, float]] = []
                batch_size = max(1, int(opts.batch_size))
                for start in range(0, len(crops), batch_size):
                    outputs.extend(self._recognize_crops(crops[start : start + batch_size], opts))
            timings["inference"] = inf_timer.elapsed_ms

        timings["total"] = total_timer.elapsed_ms
        per_line_ms = timings["inference"] / max(1, len(outputs))

        # 4. assemble
        lines: list[LinePrediction] = []
        for index, (text, confidence) in enumerate(outputs):
            if not text.strip():
                continue
            if opts.min_confidence > 0 and confidence < opts.min_confidence:
                logger.debug(
                    "Dropping line %d: confidence %.3f < %.3f", index, confidence, opts.min_confidence
                )
                continue
            lines.append(
                LinePrediction(
                    index=len(lines),
                    text=text,
                    confidence=confidence,
                    region=segmentation.regions[index] if index < len(segmentation.regions) else None,
                    inference_ms=per_line_ms,
                )
            )

        # Rebuild visual rows. Segmentation keeps a table's columns apart so
        # each is a tight crop the model reads well; this joins them back with
        # spaces, so `Laptop stand | 2 | 51.25` comes out as one line rather
        # than three. A column gutter prevents joining across it, so a
        # two-column page stays two columns.
        full_text = self._assemble_text(lines, segmentation)
        # Weight each line's confidence by its length: a one-character line
        # recognised confidently should not outvote a long uncertain one.
        total_chars = sum(len(line.text) for line in lines)
        overall = (
            sum(line.confidence * len(line.text) for line in lines) / total_chars
            if total_chars
            else 0.0
        )

        segmentation_info = segmentation.to_dict()
        if opts.handwritten:
            segmentation_info["handwriting"] = handwriting_info

        return RecognitionResult(
            text=full_text,
            lines=lines,
            model_id=self.model_id,
            model_name=self.model_name,
            confidence=overall,
            timings_ms=timings,
            segmentation=segmentation_info,
            preprocessing=processed.metadata(),
            device=str(self.device),
            image_size=(width, height),
        )


class CRNNPredictor(BasePredictor):
    """Predictor for the project's own CNN + BiLSTM + Attention recogniser.

    Args:
        model: a loaded recogniser in eval mode.
        charset: the charset the model was trained with.
        geometry: input geometry used during training -- changing it at
            inference time silently degrades accuracy, which is why it travels
            inside the checkpoint rather than coming from the current config.
        model_id/model_name: identifiers surfaced by the API.
        provenance: ``"trained_in_project"`` or ``"pretrained_third_party"``.
        metadata: extra fields for :meth:`describe` (training metrics, etc.).
    """

    def __init__(
        self,
        model: CRNNAttentionOCR | CNNTransformerOCR | MambaOCR,
        charset: Charset,
        *,
        geometry: ImageGeometry,
        device: torch.device,
        model_id: str,
        model_name: str,
        provenance: str = "trained_in_project",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, model_name=model_name, device=device)
        self.model = model.to(device).eval()
        self.charset = charset
        self.geometry = geometry
        self.provenance = provenance
        self.metadata = metadata or {}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device,
        model_id: str | None = None,
        model_name: str | None = None,
        provenance: str = "trained_in_project",
    ) -> "CRNNPredictor":
        """Load a predictor from a checkpoint written by :class:`~ml.engine.ctc_trainer.CTCTrainer`.

        Raises:
            FileNotFoundError: *path* does not exist.
            RuntimeError: the checkpoint is missing required fields or its
                weights do not match the architecture it declares.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        payload = safe_load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(
                f"{checkpoint_path} is not a recogniser checkpoint "
                "(no 'model_state_dict' key)."
            )

        charset_payload = payload.get("charset")
        if not charset_payload:
            raise RuntimeError(
                f"{checkpoint_path} does not carry its charset, so its outputs cannot be "
                "mapped back to characters. Retrain with the current trainer."
            )
        charset = Charset.from_dict(charset_payload)

        config_payload = payload.get("config") or {}
        if config_payload:
            config = Config(config_payload)
            family = config.get("model.family", "crnn_attention")
            if family == "cnn_transformer":
                model = CNNTransformerOCR.from_config(config, charset)
            elif family == "mamba_ocr":
                model = MambaOCR.from_config(config, charset)
            else:
                model = CRNNAttentionOCR.from_config(config, charset)
        else:
            logger.warning(
                "%s has no config snapshot; rebuilding with default hyperparameters. "
                "This works only if the model was trained with the defaults.",
                checkpoint_path.name,
            )
            model = CRNNAttentionOCR(num_classes=charset.num_classes)

        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"The weights in {checkpoint_path.name} do not match the architecture it "
                f"declares. The checkpoint is likely from an older version. ({exc})"
            ) from exc

        geometry_payload = payload.get("geometry") or {}
        geometry = ImageGeometry(
            height=int(geometry_payload.get("height", 32)),
            width=int(geometry_payload.get("width", 256)),
            channels=int(geometry_payload.get("channels", 1)),
            keep_aspect_ratio=bool(geometry_payload.get("keep_aspect_ratio", True)),
            pad_value=int(geometry_payload.get("pad_value", 255)),
        )

        run_name = str(payload.get("run_name") or checkpoint_path.parent.name)
        metrics = payload.get("metrics") or {}

        return cls(
            model,
            charset,
            geometry=geometry,
            device=device,
            model_id=model_id or run_name,
            model_name=model_name or run_name.replace("_", " ").title(),
            provenance=provenance,
            metadata={
                "checkpoint": str(checkpoint_path),
                "epoch": payload.get("epoch"),
                "training_metrics": {
                    k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
                },
                # Named distinctly: "architecture" in the public model schema is
                # a human-readable string, and this is the checkpoint's
                # structured description. Reusing the name let this dict
                # overwrite the string and break GET /api/models.
                "architecture_detail": payload.get("architecture", {}),
            },
        )

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _recognize_crops(
        self, crops: Sequence[np.ndarray], options: RecognizeOptions
    ) -> list[tuple[str, float]]:
        if not crops:
            return []

        # Variable-width canvas, not the fixed training width.
        #
        # The network is fully convolutional and recurrent along width, so it
        # accepts any input width at inference. Forcing the 256 px training
        # canvas here would crop every long line -- a full document sentence
        # loses more than half its characters that way, and the truncated
        # result looks like a complete transcription.
        batch, ratios, infos = prepare_variable_width_batch(
            crops,
            height=self.geometry.height,
            channels=self.geometry.channels,
            pad_value=self.geometry.pad_value,
            normalize=True,
            min_width=self.geometry.width,
        )

        truncated = sum(1 for info in infos if info.truncated)
        if truncated:
            logger.warning(
                "%d of %d line(s) were still too wide for the %d px cap and were "
                "compressed to fit.",
                truncated, len(infos), MAX_INFERENCE_WIDTH,
            )

        images = torch.from_numpy(batch).to(self.device)
        time_steps = self.model.encoder.output_length(images.size(-1))
        lengths = torch.tensor(
            [max(1, min(time_steps, int(np.ceil(r * time_steps)))) for r in ratios],
            dtype=torch.long,
        )
        if isinstance(self.model, (MambaOCR, CNNTransformerOCR)):
            output = self.model(images, input_lengths=lengths)
        else:
            output = self.model(images)

        decoded = ctc_decode(
            output.log_probs,
            self.charset,
            strategy=options.decode_strategy,
            input_lengths=lengths,
            beam_width=options.beam_width,
        )
        return [(result.text, result.confidence) for result in decoded]

    def describe(self) -> dict[str, Any]:
        # Metadata first so the explicit keys below always win: a checkpoint
        # field must never be able to redefine a key the API schema types.
        return {
            **self.metadata,
            "id": self.model_id,
            "name": self.model_name,
            "family": (
                "mamba_ocr"
                if isinstance(self.model, MambaOCR)
                else "cnn_transformer"
                if isinstance(self.model, CNNTransformerOCR)
                else "crnn_attention"
            ),
            "architecture": self.model.describe()["architecture"],
            "provenance": self.provenance,
            "device": str(self.device),
            "available": True,
            "loaded": True,
            "charset_size": self.charset.num_classes,
            "geometry": self.geometry.to_dict(),
            "parameters": self.model.parameter_breakdown(),
            "supports_beam_search": True,
        }


class MambaPredictor(CRNNPredictor):
    """Checkpoint-backed predictor for the CNN + bidirectional Mamba OCR model."""

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device,
        model_id: str | None = None,
        model_name: str | None = None,
        provenance: str = "trained_in_project",
    ) -> "MambaPredictor":
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        payload = safe_load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(
                f"{checkpoint_path} is not a recogniser checkpoint "
                "(no 'model_state_dict' key)."
            )
        charset_payload = payload.get("charset")
        if not charset_payload:
            raise RuntimeError(f"{checkpoint_path} does not carry its charset.")

        charset = Charset.from_dict(charset_payload)
        config_payload = payload.get("config") or {}
        if config_payload:
            model = MambaOCR.from_config(Config(config_payload), charset)
        else:
            logger.warning(
                "%s has no config snapshot; rebuilding Mamba with default hyperparameters.",
                checkpoint_path.name,
            )
            model = MambaOCR(num_classes=charset.num_classes)

        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"The weights in {checkpoint_path.name} do not match its Mamba architecture. ({exc})"
            ) from exc

        geometry_payload = payload.get("geometry") or {}
        geometry = ImageGeometry(
            height=int(geometry_payload.get("height", 32)),
            width=int(geometry_payload.get("width", 256)),
            channels=int(geometry_payload.get("channels", 1)),
            keep_aspect_ratio=bool(geometry_payload.get("keep_aspect_ratio", True)),
            pad_value=int(geometry_payload.get("pad_value", 255)),
        )
        run_name = str(payload.get("run_name") or checkpoint_path.parent.name)
        metrics = payload.get("metrics") or {}
        return cls(
            model,
            charset,
            geometry=geometry,
            device=device,
            model_id=model_id or run_name,
            model_name=model_name or run_name.replace("_", " ").title(),
            provenance=provenance,
            metadata={
                "checkpoint": str(checkpoint_path),
                "epoch": payload.get("epoch"),
                "training_metrics": {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in metrics.items()
                },
                "architecture_detail": payload.get("architecture", {}),
            },
        )

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["family"] = "mamba_ocr"
        description["architecture"] = self.model.describe()["architecture"]
        return description


class TrOCRPredictor(BasePredictor):
    """Predictor wrapping a pretrained :class:`~ml.models.trocr_wrapper.TrOCRRecognizer`."""

    def __init__(
        self,
        recognizer: TrOCRRecognizer,
        *,
        model_id: str,
        model_name: str,
        device: torch.device,
        description: str = "",
        postprocess: str = "detokenize",
        provenance: str = "pretrained_third_party",
    ) -> None:
        super().__init__(model_id=model_id, model_name=model_name, device=device)
        self.recognizer = recognizer
        self.description = description
        # "trained_in_project" for a TrOCR fine-tuned by scripts/train_trocr.py
        self.provenance = provenance
        # TrOCR's document models were fine-tuned on IAM, whose transcriptions
        # separate punctuation with spaces ("Estate , Mangalore"). That is
        # faithful to the training data but not to the page, so it is undone
        # here. Set to "none" to see the model's raw output.
        self.postprocess = postprocess

    def _recognize_crops(
        self, crops: Sequence[np.ndarray], options: RecognizeOptions
    ) -> list[tuple[str, float]]:
        if not crops:
            return []
        try:
            predictions = self.recognizer.predict(crops)
        except TrOCRUnavailable:
            raise
        return [
            (normalize_recognised_text(p.text, mode=self.postprocess), p.confidence)
            for p in predictions
        ]

    def warmup(self) -> None:
        """Skip warm-up for TrOCR.

        Warm-up would trigger the (potentially multi-hundred-megabyte)
        download at application start, blocking it on the network for a model
        the user may never select.  TrOCR loads on first real use instead.
        """
        return

    def describe(self) -> dict[str, Any]:
        info = self.recognizer.describe()
        return {
            "id": self.model_id,
            "name": self.model_name,
            "family": "trocr",
            "architecture": info["architecture"],
            "provenance": self.provenance,
            "description": self.description
            or (
                "Pretrained Transformer OCR from Microsoft. Downloaded weights -- not "
                "trained by this project, and not part of the self-supervised experiment."
            ),
            "device": str(self.device),
            "available": info["available"],
            "loaded": info["loaded"],
            "error": info.get("error"),
            "supports_beam_search": False,
            "postprocess": self.postprocess,
            "hf_model_id": info["model_id"],
            "parameters": {"total": info.get("parameters")} if info.get("parameters") else {},
        }
