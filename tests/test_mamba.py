"""Tests for the bidirectional selective-state-space OCR recogniser."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.mamba_ocr import MambaOCR, MambaSequenceLayer
from ml.text.charset import Charset


def test_mamba_sequence_layer_preserves_valid_outputs_with_right_padding() -> None:
    torch.manual_seed(7)
    layer = MambaSequenceLayer(
        8,
        d_state=3,
        conv_kernel=3,
        expansion=1,
        dropout=0.0,
    ).eval()
    valid_sequence = torch.randn(1, 5, 8)
    short = torch.cat((valid_sequence, torch.zeros(1, 3, 8)), dim=1)
    long = torch.cat((valid_sequence, torch.zeros(1, 7, 8)), dim=1)

    with torch.no_grad():
        short_output = layer(short, torch.tensor([5]))
        long_output = layer(long, torch.tensor([5]))

    assert torch.allclose(short_output[:, :5], long_output[:, :5], atol=1e-6)


def test_mamba_ocr_outputs_ctc_log_probabilities_and_finite_gradients() -> None:
    charset = Charset("abc ")
    model = MambaOCR(
        num_classes=charset.num_classes,
        d_state=4,
        layers=1,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 32, 128)

    output = model(images, input_lengths=torch.tensor([31, 16]))
    loss = -output.log_probs[:, :, 1].mean()
    loss.backward()

    assert output.logits.shape == (2, 31, charset.num_classes)
    assert output.log_probs.dtype == torch.float32
    assert output.input_lengths.tolist() == [31, 16]
    assert torch.allclose(
        output.log_probs.exp().sum(dim=-1),
        torch.ones_like(output.log_probs[:, :, 0]),
        atol=1e-5,
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.parameter_breakdown()["total"] == model.num_parameters()
