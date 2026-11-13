"""SimCLR self-supervised pretraining (Chen et al., 2020), adapted to text.

Role in the project
-------------------
This is requirement (C) and (D) of the project: pretrain the CNN encoder on
*unlabeled* text crops, then transfer those weights into the supervised
recogniser.  The encoder here is the very same
:class:`ml.models.backbone.CNNEncoder` the recogniser uses, so the transfer in
:meth:`ml.models.crnn_attention.CRNNAttentionOCR.load_encoder_weights` is an
exact ``state_dict`` load.

What is adapted for OCR
-----------------------
* **Views.** The canonical recipe (random resized crop + colour jitter +
  horizontal flip) is wrong for text: flipping or aggressively cropping a word
  changes what it says.  :class:`ml.preprocessing.augment.SimCLRViewGenerator`
  substitutes blur, noise, stroke erosion/dilation, JPEG artefacts, slant and
  mild width cropping -- nuisances a scanner or camera genuinely introduces.
* **Pooling.** The representation is a global average over the feature map, so
  it does not depend on where in the canvas the text happens to sit.

The projection head is discarded after pretraining; only the encoder transfers.
That is the finding from the original paper, and it is why
:meth:`SimCLRModel.encoder_state_dict` exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.backbone import CNNEncoder, build_encoder

__all__ = ["ProjectionHead", "SimCLRModel", "NTXentLoss", "NTXentOutput"]

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Two-layer MLP mapping encoder features to the contrastive space.

    The non-linear head is the single highest-impact detail in SimCLR: the
    contrastive objective is invariant to the augmentations by construction,
    and if it acted directly on the encoder output it would strip information
    the downstream task still needs (here: stroke shape and position).  The
    head absorbs that invariance so the encoder does not have to.
    """

    def __init__(self, in_dim: int, *, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(slots=True)
class NTXentOutput:
    """Loss plus diagnostics for one contrastive step.

    Attributes:
        loss: the NT-Xent loss.
        accuracy: fraction of anchors whose nearest non-self neighbour is its
            true positive.  This is the metric to watch -- the loss alone is
            hard to read because its scale depends on the batch size, while
            this starts near ``1 / (2B - 1)`` and should climb steadily.
        positive_similarity: mean cosine similarity of the positive pairs.
        negative_similarity: mean cosine similarity of the negative pairs.
    """

    loss: torch.Tensor
    accuracy: float
    positive_similarity: float
    negative_similarity: float


class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy loss.

    For a batch of ``B`` images, two views give ``2B`` embeddings.  Each
    embedding's positive is the other view of the same image; the remaining
    ``2B - 2`` embeddings are negatives.  The loss is cross-entropy over the
    temperature-scaled cosine similarities.

    Args:
        temperature: lower values sharpen the distribution and weight hard
            negatives more heavily.  0.1 works well for the fairly homogeneous
            text crops used here; the paper's 0.5 under-trains on them.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> NTXentOutput:
        """Args:
            z_i: ``(B, D)`` projections of the first view.
            z_j: ``(B, D)`` projections of the second view.

        Returns:
            An :class:`NTXentOutput`.

        Raises:
            ValueError: the two view batches do not have the same shape, or
                the batch size is below 2 (no negatives exist).
        """
        if z_i.shape != z_j.shape:
            raise ValueError(f"View shapes must match, got {tuple(z_i.shape)} and {tuple(z_j.shape)}")
        if z_i.dim() != 2:
            raise ValueError(f"Expected (B, D) projections, got {tuple(z_i.shape)}")

        batch = z_i.size(0)
        if batch < 2:
            raise ValueError(
                f"NT-Xent needs a batch of at least 2 to have any negatives, got {batch}"
            )

        # float32 for the similarity matrix: under autocast the fp16 softmax
        # over 2B logits divided by a small temperature overflows.
        z = torch.cat([z_i, z_j], dim=0).float()
        z = F.normalize(z, p=2, dim=1)

        similarity = z @ z.t()  # (2B, 2B) cosine similarity

        # Mask the diagonal: an embedding must not be its own positive.
        self_mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        logits = similarity.masked_fill(self_mask, float("-inf")) / self.temperature

        # Positive of row i is row i+B, and of row i+B is row i.
        indices = torch.arange(batch, device=z.device)
        targets = torch.cat([indices + batch, indices], dim=0)

        loss = F.cross_entropy(logits, targets)

        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == targets).float().mean().item()

            positive_sim = similarity[torch.arange(2 * batch, device=z.device), targets].mean().item()
            negative_mask = ~self_mask
            negative_mask[torch.arange(2 * batch, device=z.device), targets] = False
            negative_sim = similarity[negative_mask].mean().item()

        return NTXentOutput(
            loss=loss,
            accuracy=accuracy,
            positive_similarity=positive_sim,
            negative_similarity=negative_sim,
        )


class SimCLRModel(nn.Module):
    """Encoder + projection head, trained with :class:`NTXentLoss`.

    >>> model = SimCLRModel(encoder_cfg={"type": "crnn_vgg"})
    >>> z_i, z_j = model(view_a, view_b)
    >>> out = NTXentLoss(0.1)(z_i, z_j)
    >>> out.loss.backward()
    """

    def __init__(
        self,
        *,
        encoder: CNNEncoder | None = None,
        encoder_cfg: Mapping[str, Any] | Any = None,
        in_channels: int = 1,
        projection_dim: int = 128,
        projection_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else build_encoder(encoder_cfg, in_channels=in_channels)
        self.in_channels = int(in_channels)
        self.projection = ProjectionHead(
            self.encoder.feature_dim,
            hidden_dim=int(projection_hidden),
            out_dim=int(projection_dim),
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "SimCLRModel":
        """Build from a loaded :class:`ml.config.Config`."""
        encoder_cfg = cfg.get("crnn.encoder", {})
        return cls(
            encoder_cfg=encoder_cfg,
            in_channels=int(cfg.get("image.channels", 1)),
            projection_dim=int(cfg.get("simclr.projection_dim", 128)),
            projection_hidden=int(cfg.get("simclr.projection_hidden", 512)),
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Encoder representation ``(B, feature_dim)`` -- what gets transferred."""
        return self.encoder.embed(images)

    def project(self, images: torch.Tensor) -> torch.Tensor:
        """Contrastive projection ``(B, projection_dim)``."""
        return self.projection(self.encoder.embed(images))

    def forward(self, view_i: torch.Tensor, view_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both views.

        The two views are concatenated into one batch before the forward pass.
        That is not merely an optimisation: the projection head contains
        BatchNorm, and running the views separately would normalise each with
        its own statistics, leaking the pair structure into the features and
        letting the model partly cheat the objective.
        """
        if view_i.shape != view_j.shape:
            raise ValueError(
                f"Both views must have the same shape, got {tuple(view_i.shape)} and {tuple(view_j.shape)}"
            )

        batch = view_i.size(0)
        combined = torch.cat([view_i, view_j], dim=0)
        projected = self.projection(self.encoder.embed(combined))
        return projected[:batch], projected[batch:]

    # -- checkpointing -----------------------------------------------------
    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """The encoder weights alone, ready for the recogniser to load.

        The projection head is intentionally excluded: it is a scaffold for the
        contrastive objective and has no counterpart in the OCR model.
        """
        return {key: value.detach().cpu().clone() for key, value in self.encoder.state_dict().items()}

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "SimCLR (contrastive self-supervised pretraining)",
            "encoder": type(self.encoder).__name__,
            "feature_dim": self.encoder.feature_dim,
            "projection_dim": self.projection.out_dim,
            "parameters": {
                "encoder": self.encoder.num_parameters(),
                "projection": sum(p.numel() for p in self.projection.parameters()),
                "total": self.num_parameters(),
            },
        }
