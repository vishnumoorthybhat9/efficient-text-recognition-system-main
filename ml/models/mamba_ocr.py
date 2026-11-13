"""CNN + bidirectional selective state-space + CTC text recogniser."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.backbone import CNNEncoder, build_encoder
from ml.models.crnn_attention import RecognizerOutput
from ml.text.charset import Charset

__all__ = ["MambaOCR", "MambaSequenceLayer", "SelectiveStateSpace"]


class SelectiveStateSpace(nn.Module):
    """A Mamba-style input-selective state-space block with a causal convolution."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        conv_kernel: int = 4,
        expansion: int = 1,
    ) -> None:
        super().__init__()
        if min(d_model, d_state, conv_kernel, expansion) < 1:
            raise ValueError("Mamba dimensions and convolution kernel must be positive")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = self.d_model * int(expansion)
        self.dt_rank = max(1, math.ceil(self.d_model / 16))

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=int(conv_kernel),
            groups=self.d_inner,
            padding=0,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        state_values = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(state_values.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt_scale = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_scale, dt_scale)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        *,
        reverse_padding: bool = False,
    ) -> torch.Tensor:
        batch, time_steps, _ = sequence.shape
        positions = torch.arange(time_steps, device=sequence.device)[None, :]
        valid = (
            positions >= (time_steps - lengths[:, None])
            if reverse_padding
            else positions < lengths[:, None]
        )

        sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0.0)
        x, gate = self.in_proj(sequence).chunk(2, dim=-1)
        x = self.conv1d(F.pad(x.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0)))
        x = F.silu(x.transpose(1, 2)).masked_fill(~valid.unsqueeze(-1), 0.0)
        selective = self.x_proj(x)
        dt_low, input_state, output_state = torch.split(
            selective, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(dt_low).float()).clamp(max=20.0)
        input_state = input_state.float()
        output_state = output_state.float()
        x_float = x.float()
        decay = -torch.exp(self.A_log.float())
        state = x_float.new_zeros((batch, self.d_inner, self.d_state))
        outputs: list[torch.Tensor] = []
        for index in range(time_steps):
            step_valid = valid[:, index, None, None]
            step_delta = delta[:, index]
            candidate = (
                torch.exp(step_delta.unsqueeze(-1) * decay) * state
                + step_delta.unsqueeze(-1)
                * input_state[:, index, None, :]
                * x_float[:, index, :, None]
            )
            state = torch.where(step_valid, candidate, state)
            emitted = (state * output_state[:, index, None, :]).sum(dim=-1)
            emitted = emitted + self.D.float() * x_float[:, index]
            outputs.append(emitted * valid[:, index, None])

        result = torch.stack(outputs, dim=1).to(dtype=gate.dtype)
        return self.out_proj(result * F.silu(gate))


class MambaSequenceLayer(nn.Module):
    """Bidirectional selective SSM layer with a residual connection."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        conv_kernel: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_ssm = SelectiveStateSpace(
            d_model, d_state=d_state, conv_kernel=conv_kernel, expansion=expansion
        )
        self.backward_ssm = SelectiveStateSpace(
            d_model, d_state=d_state, conv_kernel=conv_kernel, expansion=expansion
        )
        self.merge = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.size(1), device=sequence.device)[None, :]
        valid = positions < lengths[:, None]
        normalized = self.norm(sequence).masked_fill(~valid.unsqueeze(-1), 0.0)
        forward = self.forward_ssm(normalized, lengths)
        backward = self.backward_ssm(
            normalized.flip(1), lengths, reverse_padding=True
        ).flip(1)
        return sequence + self.dropout(self.merge(torch.cat((forward, backward), dim=-1)))


class MambaOCR(nn.Module):
    """CNN visual encoder followed by bidirectional Mamba-style sequence layers and CTC."""

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 1,
        encoder: CNNEncoder | None = None,
        encoder_cfg: Mapping[str, Any] | Any = None,
        d_model: int = 256,
        d_state: int = 16,
        conv_kernel: int = 4,
        layers: int = 2,
        expansion: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2 (blank + 1 symbol), got {num_classes}")
        if layers < 1:
            raise ValueError(f"layers must be at least 1, got {layers}")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.encoder = encoder if encoder is not None else build_encoder(encoder_cfg, in_channels=in_channels)
        feature_dim = self.encoder.feature_dim
        self.input_projection = nn.Linear(feature_dim, int(d_model))
        self.sequence_layers = nn.ModuleList(
            MambaSequenceLayer(
                int(d_model),
                d_state=d_state,
                conv_kernel=conv_kernel,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.classifier = nn.Linear(int(d_model), self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_config(cls, cfg: Any, charset: Charset) -> "MambaOCR":
        block = cfg.get("mamba", {}) or {}

        def pick(key: str, default: Any) -> Any:
            value = block.get(key, default) if hasattr(block, "get") else default
            return default if value is None else value

        return cls(
            num_classes=charset.num_classes,
            in_channels=int(cfg.get("image.channels", 1)),
            encoder_cfg=cfg.get("crnn.encoder", {}),
            d_model=int(pick("d_model", 256)),
            d_state=int(pick("d_state", 16)),
            conv_kernel=int(pick("conv_kernel", 4)),
            layers=int(pick("layers", 2)),
            expansion=int(pick("expansion", 1)),
            dropout=float(pick("dropout", 0.1)),
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

        for layer in self.sequence_layers:
            sequence = layer(sequence, lengths)

        logits = self.classifier(sequence)
        return RecognizerOutput(
            logits=logits,
            log_probs=torch.log_softmax(logits.float(), dim=-1),
            input_lengths=lengths,
        )

    def freeze_encoder(self, frozen: bool = True) -> None:
        self.encoder.set_frozen(frozen)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        return sum(parameter.numel() for parameter in parameters if not trainable_only or parameter.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        mamba = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(("input_projection.", "sequence_layers."))
        )
        classifier = sum(parameter.numel() for parameter in self.classifier.parameters())
        return {
            "encoder": encoder,
            "mamba": mamba,
            "classifier": classifier,
            "total": encoder + mamba + classifier,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "CNN + bidirectional selective state-space (Mamba) + CTC",
            "parameters": self.parameter_breakdown(),
            "sequence_layers": len(self.sequence_layers),
            "sequence_dimension": self.input_projection.out_features,
            "state_size": self.sequence_layers[0].forward_ssm.d_state,
        }
