"""CNN encoders shared by the OCR recogniser and by SimCLR pretraining.

This module is the hinge of the whole project.  The *same* encoder object is:

* pretrained contrastively by :mod:`ml.models.simclr` (via :meth:`embed`), and
* used as the visual front-end of the recogniser in
  :mod:`ml.models.crnn_attention` (via :meth:`forward`).

Because both paths share one ``state_dict``, transferring self-supervised
weights into the supervised model is an exact load rather than a hand-written
mapping -- which is what makes the "with vs. without pretraining" comparison
trustworthy.

Output contract
---------------
``forward(x)`` takes ``(B, C_in, H, W)`` and returns ``(B, T, feature_dim)``:
a left-to-right sequence of column features, as CTC requires.  Height is
collapsed to 1 and width is downsampled by 4 (minus one for the final valid
2x2 convolution), so ``T = W // 4 - 1`` for the default encoder.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

__all__ = ["CNNEncoder", "VGGEncoder", "ResNetSmallEncoder", "build_encoder", "ENCODER_TYPES"]

logger = logging.getLogger(__name__)

ENCODER_TYPES = ("crnn_vgg", "resnet_small")

DEFAULT_CHANNELS: tuple[int, ...] = (64, 128, 256, 256, 512, 512, 512)


class CNNEncoder(nn.Module):
    """Base class defining the contract every encoder must satisfy."""

    #: Channel count of the emitted sequence; set by subclasses.
    feature_dim: int

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw feature map ``(B, C, H', W')``."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a width-major sequence ``(B, T, feature_dim)``.

        Any residual height is average-pooled away, so the encoder tolerates an
        input height other than the one it was configured for instead of
        failing with a shape error.
        """
        features = self.forward_features(x)
        if features.dim() != 4:
            raise RuntimeError(f"Encoder features must be 4-D, got shape {tuple(features.shape)}")

        if features.size(2) != 1:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, features.size(3)))

        # (B, C, 1, W') -> (B, W', C)
        return features.squeeze(2).permute(0, 2, 1).contiguous()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return one global vector per image ``(B, feature_dim)``.

        This is the representation SimCLR contrasts.  Averaging over width
        (rather than taking a fixed position) keeps it independent of how much
        of the canvas the text occupies.
        """
        features = self.forward_features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(pooled, 1)

    @torch.no_grad()
    def output_length(self, input_width: int, *, input_height: int = 32, channels: int = 1) -> int:
        """Number of time steps produced for an input of *input_width*.

        Measured with a single dummy forward pass rather than derived from a
        formula, so it stays correct if the architecture changes.
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless encoder
            device = torch.device("cpu")
        try:
            dummy = torch.zeros(1, channels, int(input_height), int(input_width), device=device)
            length = int(self.forward(dummy).size(1))
        finally:
            if was_training:
                self.train()
        return length

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """Parameter count, reported in the model-comparison table."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze every parameter.

        Used by the SimCLR fine-tuning schedule, which holds the transferred
        encoder fixed for the first few epochs so the randomly initialised
        recognition head cannot wash out the pretrained features.
        """
        for param in self.parameters():
            param.requires_grad = not frozen
        # Keep BatchNorm running statistics stable while frozen.
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = not frozen


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    batch_norm: bool = False,
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not batch_norm,  # a following BatchNorm makes the bias redundant
        )
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


class VGGEncoder(CNNEncoder):
    """The CRNN convolutional front-end (Shi et al., 2015), modernised.

    Differences from the original paper, all deliberate:

    * the two height-only pooling layers use a ``(2, 1)`` kernel with no
      padding instead of ``(2, 2)`` with asymmetric padding, which makes the
      width reduction exactly ``//4`` and the time-step count easy to reason
      about;
    * BatchNorm on the deeper blocks (the paper only had it on two), which
      matters because SimCLR pretraining uses large batches;
    * optional dropout before the final projection.

    For a ``32 x 256`` grayscale input this emits 63 time steps of 512 features.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(channels) != 7:
            raise ValueError(
                f"VGGEncoder expects exactly 7 channel widths, got {len(channels)}: {list(channels)}"
            )

        c1, c2, c3, c4, c5, c6, c7 = (int(c) for c in channels)
        self.in_channels = int(in_channels)
        self.feature_dim = c7

        layers: list[nn.Module] = []

        # Block 1: H/2, W/2
        layers += _conv_block(self.in_channels, c1)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 2: H/4, W/4
        layers += _conv_block(c1, c2)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 3: two convolutions, then pool height only -> H/8, W/4
        layers += _conv_block(c2, c3, batch_norm=True)
        layers += _conv_block(c3, c4)
        layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

        # Block 4: two convolutions, then pool height only -> H/16, W/4
        layers += _conv_block(c4, c5, batch_norm=True)
        layers += _conv_block(c5, c6)
        layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

        # Block 5: valid 2x2 convolution collapses the last height rows and
        # trims one column -> H/32 (= 1 for H=32), W/4 - 1
        layers += _conv_block(c6, c7, kernel_size=2, stride=1, padding=0, batch_norm=True)

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.features(x))


class _ResidualBlock(nn.Module):
    """Pre-activation residual block with an optional stride."""

    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        needs_projection = in_channels != out_channels or stride != 1
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetSmallEncoder(CNNEncoder):
    """A compact residual alternative to :class:`VGGEncoder`.

    Offered because the synopsis targets *lower computational complexity*:
    this encoder reaches the same 512-d sequence with roughly a third of the
    parameters, which makes it the interesting arm when the comparison table
    is read for efficiency rather than accuracy.  Same output contract, so it
    is a drop-in swap via ``crnn.encoder.type: resnet_small``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        feature_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(
                f"ResNetSmallEncoder expects exactly 4 channel widths, got {len(channels)}"
            )
        c1, c2, c3, c4 = (int(c) for c in channels)
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
        )
        self.stage1 = _ResidualBlock(c1, c2, stride=2)            # H/4,  W/4
        self.stage2 = _ResidualBlock(c2, c3, stride=(2, 1))       # H/8,  W/4
        self.stage3 = _ResidualBlock(c3, c4, stride=(2, 1))       # H/16, W/4
        self.head = nn.Sequential(
            nn.Conv2d(c4, self.feature_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.dropout(self.head(out))


def build_encoder(cfg: Mapping[str, Any] | Any = None, *, in_channels: int = 1) -> CNNEncoder:
    """Construct an encoder from the ``crnn.encoder`` config block.

    Args:
        cfg: mapping with optional ``type``, ``channels``, ``feature_dim`` and
            ``dropout`` keys.  ``None`` builds the default VGG encoder.
        in_channels: 1 for grayscale input, 3 for colour.

    Raises:
        ValueError: ``type`` is not one of :data:`ENCODER_TYPES`.
    """
    data: dict[str, Any] = {}
    if cfg is not None:
        data = dict(cfg) if isinstance(cfg, Mapping) else dict(getattr(cfg, "to_dict", dict)())

    encoder_type = str(data.get("type", "crnn_vgg")).strip().lower()
    dropout = float(data.get("dropout", 0.1))

    if encoder_type == "crnn_vgg":
        channels = tuple(data.get("channels") or DEFAULT_CHANNELS)
        encoder = VGGEncoder(in_channels=in_channels, channels=channels, dropout=dropout)
    elif encoder_type == "resnet_small":
        channels = tuple(data.get("channels") or (32, 64, 128, 256))
        if len(channels) == 7:
            # A `channels` list inherited from the VGG default: take a sensible
            # 4-stage subset instead of refusing to build.
            channels = (channels[0], channels[1], channels[2], channels[4])
            logger.debug("Adapted 7-entry channel list to 4 stages for resnet_small: %s", channels)
        encoder = ResNetSmallEncoder(
            in_channels=in_channels,
            channels=channels,
            feature_dim=int(data.get("feature_dim", 512)),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder type {encoder_type!r}; expected one of {ENCODER_TYPES}")

    declared = data.get("feature_dim")
    if declared is not None and int(declared) != encoder.feature_dim:
        logger.warning(
            "Config declares crnn.encoder.feature_dim=%s but the %s encoder emits %d; "
            "using the encoder's actual value.",
            declared,
            encoder_type,
            encoder.feature_dim,
        )

    return encoder
