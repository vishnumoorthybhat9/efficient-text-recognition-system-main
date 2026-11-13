"""CNN visual encoder + Transformer sequence encoder + CTC recogniser."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ml.models.attention import AttentionEncoder
from ml.models.backbone import CNNEncoder, build_encoder
from ml.models.crnn_attention import RecognizerOutput
from ml.text.charset import Charset

__all__ = ["CNNTransformerOCR"]


class CNNTransformerOCR(nn.Module):
    """Extract visual columns with a CNN and model their context with a Transformer."""

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 1,
        encoder: CNNEncoder | None = None,
        encoder_cfg: Mapping[str, Any] | Any = None,
        d_model: int = 256,
        layers: int = 2,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2 (blank + 1 symbol), got {num_classes}")
        if d_model < 2 or d_model % 2:
            raise ValueError(f"d_model must be a positive even number, got {d_model}")
        if heads < 1 or d_model % heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.encoder = encoder if encoder is not None else build_encoder(encoder_cfg, in_channels=in_channels)
        self.input_projection = nn.Linear(self.encoder.feature_dim, int(d_model))
        self.transformer = AttentionEncoder(
            int(d_model),
            layers=int(layers),
            heads=int(heads),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            max_positions=int(max_positions),
        )
        self.classifier = nn.Linear(int(d_model), self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_config(cls, cfg: Any, charset: Charset) -> "CNNTransformerOCR":
        block = cfg.get("cnn_transformer", {}) or {}

        def pick(key: str, default: Any) -> Any:
            value = block.get(key, default) if hasattr(block, "get") else default
            return default if value is None else value

        return cls(
            num_classes=charset.num_classes,
            in_channels=int(cfg.get("image.channels", 1)),
            encoder_cfg=cfg.get("crnn.encoder", {}),
            d_model=int(pick("d_model", 256)),
            layers=int(pick("layers", 2)),
            heads=int(pick("heads", 8)),
            ff_dim=int(pick("ff_dim", 1024)),
            dropout=float(pick("dropout", 0.1)),
            max_positions=int(pick("max_positions", 512)),
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
    ) -> RecognizerOutput:
        if images.dim() != 4:
            raise ValueError(f"Expected a 4-D image batch (B, C, H, W), got {tuple(images.shape)}")
        if images.size(1) != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channel(s) but received {images.size(1)}"
            )

        sequence = self.input_projection(self.encoder(images))
        batch, time_steps, _ = sequence.shape
        if input_lengths is None:
            lengths = torch.full((batch,), time_steps, dtype=torch.long, device=sequence.device)
        else:
            lengths = input_lengths.to(device=sequence.device, dtype=torch.long).reshape(-1)
            if lengths.numel() != batch:
                raise ValueError(
                    f"input_lengths has {lengths.numel()} entries but the batch size is {batch}"
                )
            lengths = lengths.clamp(min=1, max=time_steps)

        sequence, attention = self.transformer(sequence, lengths=lengths)
        logits = self.classifier(sequence)
        return RecognizerOutput(
            logits=logits,
            log_probs=torch.log_softmax(logits.float(), dim=-1),
            input_lengths=lengths,
            attention=attention,
        )

    def freeze_encoder(self, frozen: bool = True) -> None:
        self.encoder.set_frozen(frozen)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def parameter_breakdown(self) -> dict[str, int]:
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        transformer = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(("input_projection.", "transformer."))
        )
        classifier = sum(parameter.numel() for parameter in self.classifier.parameters())
        return {
            "encoder": encoder,
            "transformer": transformer,
            "classifier": classifier,
            "total": encoder + transformer + classifier,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "CNN + Transformer encoder + CTC",
            "parameters": self.parameter_breakdown(),
            "transformer_layers": len(self.transformer.blocks),
            "sequence_dimension": self.input_projection.out_features,
            "attention_heads": self.transformer.blocks[0].attention.heads,
        }
