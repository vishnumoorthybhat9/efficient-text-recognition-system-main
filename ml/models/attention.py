"""Attention components for the recogniser.

Written from scratch rather than using ``nn.TransformerEncoderLayer`` for two
reasons: the attention weights are returned so the frontend can visualise
which image columns the model attended to, and the padding mask has to follow
the CTC input lengths exactly.

Why self-attention and not an attention *decoder*
-------------------------------------------------
A Bahdanau-style attention decoder emits its own output sequence, whose length
is unrelated to the input width.  That is incompatible with CTC, which needs
one output frame per input frame in monotonic order.  Placing multi-head
self-attention *between* the BiLSTM and the CTC classifier keeps ``T``
unchanged while still giving every frame a global receptive field -- so the
model can use evidence from the whole line to disambiguate one character,
which is exactly what a BiLSTM alone struggles with over long spans.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "AttentionBlock",
    "AttentionEncoder",
]


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Self-attention is permutation invariant, so without this the model would
    lose character order -- fatal for text recognition.  The encoding is fixed
    rather than learned so the model generalises to line widths it never saw
    during training.
    """

    def __init__(self, dim: int, *, max_positions: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"PositionalEncoding requires an even dimension, got {dim}")

        self.dim = int(dim)
        self.max_positions = int(max_positions)
        self.dropout = nn.Dropout(float(dropout))

        position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        table = torch.zeros(max_positions, dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)

        # Non-persistent: regenerated on construction, so it never bloats a
        # checkpoint or breaks loading when `max_positions` changes.
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to ``(B, T, D)``.

        Raises:
            ValueError: the sequence is longer than ``max_positions``.
        """
        seq_len = x.size(1)
        if seq_len > self.max_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds PositionalEncoding max_positions="
                f"{self.max_positions}; raise crnn.attention.max_positions."
            )
        return self.dropout(x + self.table[:, :seq_len, :].to(dtype=x.dtype))  # type: ignore[index]


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    Returns the averaged attention map alongside the output so it can be
    rendered as a heat map over the input columns.
    """

    def __init__(self, dim: int, *, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim**-0.5

        # One fused projection for Q, K and V: a single GEMM instead of three.
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.proj_dropout = nn.Dropout(float(dropout))

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:
            x: ``(B, T, D)``.
            key_padding_mask: ``(B, T)`` boolean, ``True`` marks a padded
                position that must not be attended to.
            need_weights: also return the head-averaged ``(B, T, T)`` map.

        Returns:
            ``(output, attention_weights_or_None)``.
        """
        batch, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, heads, T, T)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # broadcast over heads and queries
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.attn_dropout(weights), value)  # (B, heads, T, head_dim)

        merged = attended.transpose(1, 2).reshape(batch, seq_len, self.dim)
        output = self.proj_dropout(self.out_proj(merged))

        return output, (weights.mean(dim=1) if need_weights else None)


class AttentionBlock(nn.Module):
    """Pre-norm transformer block: self-attention + position-wise feed-forward.

    Pre-norm (rather than the original post-norm) is used because it trains
    stably without a learning-rate warm-up schedule tuned per depth, which
    matters for a model trained for a few dozen epochs on a laptop GPU.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            self.norm1(x), key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        return x, weights


class AttentionEncoder(nn.Module):
    """A stack of :class:`AttentionBlock` layers with positional encoding.

    Length preserving: input ``(B, T, D)`` -> output ``(B, T, D)``, which is
    what keeps the downstream CTC loss valid.
    """

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 1,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"AttentionEncoder needs at least one layer, got {layers}")

        self.positional = PositionalEncoding(dim, max_positions=max_positions, dropout=dropout)
        self.blocks = nn.ModuleList(
            AttentionBlock(dim, heads=heads, ff_dim=ff_dim, dropout=dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:
            x: ``(B, T, D)``.
            lengths: ``(B,)`` valid time steps per sample; positions beyond
                this are masked out of the attention.
            need_weights: return the attention map of the **last** layer.

        Returns:
            ``(output, last_layer_attention_or_None)``.
        """
        mask = _padding_mask(lengths, x.size(1), x.device) if lengths is not None else None

        out = self.positional(x)
        weights: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            is_last = index == len(self.blocks) - 1
            out, layer_weights = block(
                out, key_padding_mask=mask, need_weights=need_weights and is_last
            )
            if layer_weights is not None:
                weights = layer_weights

        return self.norm(out), weights


def _padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Build a ``(B, T)`` mask where ``True`` means "padded, ignore me"."""
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    return positions >= lengths.to(device=device, dtype=torch.long).unsqueeze(1)
