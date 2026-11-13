"""CNN + BiLSTM + Self-Attention + CTC text recogniser.

This is model (A) of the project: the architecture named in the synopsis
methodology ("Fine-tune deep learning models (CNN + BiLSTM + Attention)").

Pipeline
--------
::

    image (B, 1, 32, 256)
      -> CNNEncoder                -> (B, 63, 512)   visual column features
      -> BiLSTM x2                 -> (B, 63, 512)   local bidirectional context
      -> Multi-head self-attention -> (B, 63, 512)   global context, length preserving
      -> Linear                    -> (B, 63, V+1)   per-frame class logits
      -> CTC loss / CTC decoding

The CNN stage is a :class:`ml.models.backbone.CNNEncoder`, so its weights can
be loaded straight from a SimCLR pretraining run -- see
:meth:`CRNNAttentionOCR.load_encoder_weights`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ml.models.attention import AttentionEncoder
from ml.models.backbone import CNNEncoder, build_encoder
from ml.text.charset import Charset
from ml.utils.serialization import safe_load

__all__ = ["CRNNAttentionOCR", "RecognizerOutput", "EncoderTransferReport"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecognizerOutput:
    """Everything a forward pass produces.

    Attributes:
        logits: ``(B, T, num_classes)`` raw scores.
        log_probs: ``(B, T, num_classes)`` log-softmax of *logits*, which is
            exactly what :class:`torch.nn.CTCLoss` and the decoders expect.
        input_lengths: ``(B,)`` valid time steps per sample.
        attention: ``(B, T, T)`` head-averaged map from the last attention
            layer, or ``None`` when it was not requested.
    """

    logits: torch.Tensor
    log_probs: torch.Tensor
    input_lengths: torch.Tensor
    attention: torch.Tensor | None = None


@dataclass(slots=True)
class EncoderTransferReport:
    """Outcome of loading pretrained encoder weights.

    Written into the training history so the report can state precisely how
    much of the encoder actually transferred, rather than assuming it did.
    """

    source: str
    loaded: int
    total: int
    missing: list[str]
    unexpected: list[str]

    @property
    def fraction(self) -> float:
        return self.loaded / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"transferred {self.loaded}/{self.total} encoder tensors "
            f"({self.fraction:.0%}) from {self.source}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "loaded": self.loaded,
            "total": self.total,
            "fraction": round(self.fraction, 4),
            "missing": self.missing,
            "unexpected": self.unexpected,
        }


class CRNNAttentionOCR(nn.Module):
    """The project's primary recogniser.

    Args:
        num_classes: CTC output units, i.e. ``len(charset) + 1``.
        in_channels: 1 for grayscale input.
        encoder: a prebuilt encoder; built from *encoder_cfg* when omitted.
        encoder_cfg: ``crnn.encoder`` config block.
        rnn_hidden: LSTM hidden size **per direction**.
        rnn_layers: number of stacked BiLSTM layers.
        rnn_dropout: dropout between LSTM layers (ignored when ``rnn_layers``
            is 1, as PyTorch itself warns).
        attention_enabled: insert the self-attention stack.
        attention_*: the ``crnn.attention`` config block.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 1,
        encoder: CNNEncoder | None = None,
        encoder_cfg: Mapping[str, Any] | Any = None,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.2,
        attention_enabled: bool = True,
        attention_heads: int = 8,
        attention_layers: int = 1,
        attention_ff_dim: int = 1024,
        attention_dropout: float = 0.1,
        attention_max_positions: int = 512,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2 (blank + 1 symbol), got {num_classes}")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        self.encoder = encoder if encoder is not None else build_encoder(encoder_cfg, in_channels=in_channels)
        feature_dim = self.encoder.feature_dim

        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = int(rnn_layers)
        self.rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=float(rnn_dropout) if self.rnn_layers > 1 else 0.0,
        )
        sequence_dim = self.rnn_hidden * 2

        self.attention_enabled = bool(attention_enabled)
        self.attention: AttentionEncoder | None = (
            AttentionEncoder(
                sequence_dim,
                layers=int(attention_layers),
                heads=int(attention_heads),
                ff_dim=int(attention_ff_dim),
                dropout=float(attention_dropout),
                max_positions=int(attention_max_positions),
            )
            if self.attention_enabled
            else None
        )

        self.classifier = nn.Linear(sequence_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self._init_rnn()

    def _init_rnn(self) -> None:
        """Orthogonal recurrent weights, and the forget-gate bias set to 1.

        The forget-gate trick (Jozefowicz et al., 2015) makes the LSTM retain
        information by default, which noticeably shortens the early-epoch
        plateau that CTC training otherwise suffers from.
        """
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                for chunk in param.chunk(4, dim=0):
                    nn.init.orthogonal_(chunk)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Gate order in PyTorch is (input, forget, cell, output).
                hidden = self.rnn_hidden
                param.data[hidden : 2 * hidden].fill_(1.0)

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any, charset: Charset) -> "CRNNAttentionOCR":
        """Build from a loaded :class:`ml.config.Config` and a charset."""
        crnn = cfg.get("crnn", {}) or {}
        encoder_cfg = crnn.get("encoder", {}) if hasattr(crnn, "get") else {}
        sequence = crnn.get("sequence", {}) if hasattr(crnn, "get") else {}
        attention = crnn.get("attention", {}) if hasattr(crnn, "get") else {}

        def pick(block: Any, key: str, default: Any) -> Any:
            if block is None:
                return default
            value = block.get(key, default) if hasattr(block, "get") else default
            return default if value is None else value

        return cls(
            num_classes=charset.num_classes,
            in_channels=int(cfg.get("image.channels", 1)),
            encoder_cfg=encoder_cfg,
            rnn_hidden=int(pick(sequence, "rnn_hidden", 256)),
            rnn_layers=int(pick(sequence, "rnn_layers", 2)),
            rnn_dropout=float(pick(sequence, "rnn_dropout", 0.2)),
            attention_enabled=bool(pick(attention, "enabled", True)),
            attention_heads=int(pick(attention, "heads", 8)),
            attention_layers=int(pick(attention, "layers", 1)),
            attention_ff_dim=int(pick(attention, "ff_dim", 1024)),
            attention_dropout=float(pick(attention, "dropout", 0.1)),
            attention_max_positions=int(pick(attention, "max_positions", 512)),
        )

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
        need_attention: bool = False,
    ) -> RecognizerOutput:
        """Run the recogniser.

        Args:
            images: ``(B, C, H, W)`` normalised float tensor.
            input_lengths: ``(B,)`` number of *valid* time steps per sample.
                Supplying this matters: the canvas is right-padded, and
                without it the backward LSTM pass starts inside the padding
                and the attention averages over blank columns.  Defaults to
                the full width for every sample.
            need_attention: also return the attention map.

        Returns:
            A :class:`RecognizerOutput`.

        Raises:
            ValueError: *images* is not 4-D, or its channel count does not
                match the encoder.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected a 4-D image batch (B, C, H, W), got {tuple(images.shape)}")
        if images.size(1) != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channel(s) but received {images.size(1)}"
            )

        features = self.encoder(images)  # (B, T, feature_dim)
        batch, seq_len, _ = features.shape

        lengths = self._clean_lengths(input_lengths, batch, seq_len, features.device)
        sequence = self._run_rnn(features, lengths)

        attention_map: torch.Tensor | None = None
        if self.attention is not None:
            sequence, attention_map = self.attention(
                sequence, lengths=lengths, need_weights=need_attention
            )

        logits = self.classifier(sequence)
        # CTCLoss needs log-probabilities in float32; autocast would otherwise
        # leave these in fp16 and the loss becomes numerically unstable.
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        return RecognizerOutput(
            logits=logits,
            log_probs=log_probs,
            input_lengths=lengths,
            attention=attention_map,
        )

    def _clean_lengths(
        self,
        input_lengths: torch.Tensor | None,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if input_lengths is None:
            return torch.full((batch,), seq_len, dtype=torch.long, device=device)

        lengths = input_lengths.to(device=device, dtype=torch.long).reshape(-1)
        if lengths.numel() != batch:
            raise ValueError(
                f"input_lengths has {lengths.numel()} entries but the batch size is {batch}"
            )
        # A length of 0 would make pack_padded_sequence raise; CTC also needs
        # at least one frame per sample.
        return lengths.clamp(min=1, max=seq_len)

    def _run_rnn(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """BiLSTM over *features*, packing when the batch is actually ragged."""
        seq_len = features.size(1)

        # cuDNN's packed path needs lengths on the CPU.
        if bool((lengths < seq_len).any()):
            packed = pack_padded_sequence(
                features,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
            return output

        output, _ = self.rnn(features)
        return output

    # -- transfer learning -------------------------------------------------
    def load_encoder_weights(
        self,
        path: str | Path,
        *,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> EncoderTransferReport:
        """Load SimCLR-pretrained weights into the CNN encoder.

        Accepts either a bare encoder ``state_dict`` or a full checkpoint
        written by :mod:`ml.engine.simclr_trainer` (which nests it under
        ``"encoder_state_dict"`` or ``"state_dict"``, optionally prefixed with
        ``"encoder."`` from the SimCLR wrapper).

        Args:
            path: checkpoint file.
            strict: require every encoder tensor to be present.
            map_location: passed to :func:`torch.load`.

        Returns:
            An :class:`EncoderTransferReport` describing what transferred.

        Raises:
            FileNotFoundError: *path* does not exist.
            RuntimeError: *strict* is set and tensors are missing.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

        # Restricted unpickler: see ml.utils.serialization for why this
        # project never falls back to weights_only=False.
        payload = safe_load(checkpoint_path, map_location=map_location)

        state: Mapping[str, torch.Tensor]
        if isinstance(payload, Mapping) and "encoder_state_dict" in payload:
            state = payload["encoder_state_dict"]
        elif isinstance(payload, Mapping) and "state_dict" in payload:
            state = payload["state_dict"]
        elif isinstance(payload, Mapping):
            state = payload  # type: ignore[assignment]
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format in {checkpoint_path}: {type(payload).__name__}"
            )

        # Strip a leading "encoder." left by the SimCLR wrapper, and drop the
        # projection head, which has no counterpart in the recogniser.
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("projection"):
                continue
            cleaned[key[len("encoder.") :] if key.startswith("encoder.") else key] = value

        target_keys = set(self.encoder.state_dict().keys())
        usable = {k: v for k, v in cleaned.items() if k in target_keys}

        # Guard against a silent shape mismatch (e.g. a different channel config).
        own_state = self.encoder.state_dict()
        mismatched = [k for k, v in usable.items() if own_state[k].shape != v.shape]
        for key in mismatched:
            logger.warning(
                "Skipping encoder tensor %r: checkpoint shape %s != model shape %s",
                key,
                tuple(usable[key].shape),
                tuple(own_state[key].shape),
            )
            usable.pop(key)

        result = self.encoder.load_state_dict(usable, strict=False)
        missing = sorted(set(result.missing_keys) | set(mismatched))
        unexpected = sorted(set(cleaned) - target_keys)

        report = EncoderTransferReport(
            source=str(checkpoint_path),
            loaded=len(usable),
            total=len(target_keys),
            missing=missing,
            unexpected=unexpected,
        )

        if strict and missing:
            raise RuntimeError(
                f"Strict encoder transfer failed: {len(missing)} tensor(s) missing, "
                f"first few are {missing[:5]}"
            )

        logger.info("Encoder transfer: %s", report.summary())
        if unexpected:
            logger.debug("Checkpoint had %d tensor(s) with no match in the encoder", len(unexpected))
        return report

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the CNN encoder (see the fine-tuning schedule)."""
        self.encoder.set_frozen(frozen)
        logger.info("CNN encoder %s", "frozen" if frozen else "unfrozen")

    # -- introspection -----------------------------------------------------
    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameters per stage, used by the model-comparison table."""
        breakdown = {
            "encoder": self.encoder.num_parameters(),
            "bilstm": sum(p.numel() for p in self.rnn.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
        }
        if self.attention is not None:
            breakdown["attention"] = sum(p.numel() for p in self.attention.parameters())
        breakdown["total"] = self.num_parameters()
        return breakdown

    @torch.no_grad()
    def time_steps_for_width(self, width: int, *, height: int = 32) -> int:
        """Time steps the encoder emits for an input of *width* pixels."""
        return self.encoder.output_length(width, input_height=height, channels=self.in_channels)

    def describe(self) -> dict[str, Any]:
        """A JSON-serialisable architecture summary for the API."""
        return {
            "architecture": "CNN + BiLSTM + Self-Attention + CTC",
            "encoder": type(self.encoder).__name__,
            "feature_dim": self.encoder.feature_dim,
            "rnn_hidden_per_direction": self.rnn_hidden,
            "rnn_layers": self.rnn_layers,
            "attention_enabled": self.attention_enabled,
            "num_classes": self.num_classes,
            "parameters": self.parameter_breakdown(),
        }

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"num_classes={self.num_classes}, attention={self.attention_enabled}"
