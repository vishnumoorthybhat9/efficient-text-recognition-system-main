"""
CNN + Transformer hybrid for handwritten-name recognition, trained with CTC.

Pipeline
    image (1 x 64 x 384)
      -> CNN backbone        local stroke / shape features, height collapsed to 1
      -> sequence (96 steps) one feature vector per horizontal position
      -> Transformer encoder global context across the whole word
      -> linear + softmax    per-step distribution over characters + blank
      -> CTC                 alignment-free training and decoding

Requires PyTorch. The pure-Python helpers live in ocr_utils.py.
"""
import math

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNBackbone(nn.Module):
    """
    (B, 1, 64, 384) -> (B, out_ch, 1, 96)
    Height goes 64 -> 32 -> 16 -> 8 -> 4 -> 1, width goes 384 -> 192 -> 96.
    """

    def __init__(self, out_ch=256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool2d(2, 2),              # 32 x 192
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),              # 16 x 96
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d((2, 1)),            # 8 x 96
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d((2, 1)),            # 4 x 96
            ConvBlock(512, 512),
        )
        # collapse the remaining height of 4 into one feature row
        self.collapse = nn.Sequential(
            nn.Conv2d(512, out_ch, kernel_size=(4, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.collapse(self.features(x))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the (B, T, D) sequence."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class OCRNet(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=4, num_layers=3,
                 ff_dim=1024, dropout=0.1):
        """num_classes = number of characters (blank is added internally)."""
        super().__init__()
        self.cnn = CNNBackbone(out_ch=d_model)
        self.pos = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        self.head = nn.Linear(d_model, num_classes + 1)  # index 0 = CTC blank

    def forward(self, x):
        """Returns log-probabilities of shape (B, T, num_classes + 1)."""
        f = self.cnn(x)                      # (B, D, 1, T)
        f = f.squeeze(2).permute(0, 2, 1)    # (B, T, D)
        f = self.pos(f)
        f = self.transformer(f)
        return self.head(f).log_softmax(dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
