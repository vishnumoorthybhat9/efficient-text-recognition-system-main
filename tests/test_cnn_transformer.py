"""Tests for the CNN + Transformer + CTC recogniser."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.cnn_transformer_ocr import CNNTransformerOCR
from ml.inference.predictor import CRNNPredictor
from ml.datasets.ocr_dataset import ImageGeometry
from ml.config import Config
from ml.text.charset import Charset


def test_cnn_transformer_outputs_ctc_probabilities_and_finite_gradients() -> None:
    charset = Charset("abc ")
    model = CNNTransformerOCR(
        num_classes=charset.num_classes,
        d_model=32,
        layers=1,
        heads=4,
        ff_dim=64,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 32, 128)

    output = model(images, input_lengths=torch.tensor([31, 16]))
    loss = -output.log_probs[:, :, 1].mean()
    loss.backward()

    assert output.logits.shape == (2, 31, charset.num_classes)
    assert output.log_probs.dtype == torch.float32
    assert output.input_lengths.tolist() == [31, 16]
    assert output.attention is None
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


def test_cnn_transformer_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by heads"):
        CNNTransformerOCR(num_classes=5, d_model=30, heads=8)


def test_cnn_transformer_checkpoint_loads_as_its_own_family(tmp_path) -> None:
    charset = Charset("abc ")
    config = Config(
        {
            "model": {"family": "cnn_transformer"},
            "image": {"channels": 1},
            "crnn": {
                "encoder": {
                    "type": "crnn_vgg",
                    "channels": [2, 4, 8, 8, 16, 16, 16],
                    "dropout": 0.0,
                }
            },
            "cnn_transformer": {
                "d_model": 16,
                "layers": 1,
                "heads": 4,
                "ff_dim": 32,
                "dropout": 0.0,
                "max_positions": 128,
            },
        }
    )
    model = CNNTransformerOCR.from_config(config, charset)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "charset": charset.to_dict(),
            "config": config.to_dict(),
            "geometry": ImageGeometry(width=128).to_dict(),
            "run_name": "cnn_transformer",
        },
        checkpoint,
    )

    predictor = CRNNPredictor.from_checkpoint(checkpoint, device=torch.device("cpu"))

    assert isinstance(predictor.model, CNNTransformerOCR)
    assert predictor.describe()["family"] == "cnn_transformer"
    assert predictor.describe()["architecture"] == "CNN + Transformer encoder + CTC"
