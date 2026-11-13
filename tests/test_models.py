"""Tests for the network architectures and the SimCLR transfer path.

The transfer test is the important one: the entire research claim rests on the
supervised model genuinely starting from the self-supervised encoder's
weights, so it is checked tensor by tensor rather than by trusting a log line.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.attention import AttentionEncoder, MultiHeadSelfAttention, PositionalEncoding
from ml.models.backbone import ResNetSmallEncoder, VGGEncoder, build_encoder
from ml.models.crnn_attention import CRNNAttentionOCR
from ml.models.simclr import NTXentLoss, SimCLRModel
from ml.text.charset import Charset


@pytest.fixture(scope="module")
def small_charset() -> Charset:
    return Charset("abcdefghij0123456789 ")


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
def test_vgg_encoder_emits_the_expected_sequence_length() -> None:
    encoder = VGGEncoder()
    sequence = encoder(torch.zeros(2, 1, 32, 256))

    # Width is downsampled by 4 and the final valid 2x2 convolution trims one
    # column: 256 // 4 - 1 = 63.
    assert sequence.shape == (2, 63, 512)


@pytest.mark.parametrize(("width", "expected"), [(128, 31), (256, 63), (384, 95)])
def test_time_steps_scale_with_width(width: int, expected: int) -> None:
    assert VGGEncoder().output_length(width) == expected


def test_encoder_collapses_height_for_a_nonstandard_input() -> None:
    # A 64-px-tall input leaves residual height; the encoder must pool it away
    # rather than raise.
    sequence = VGGEncoder()(torch.zeros(1, 1, 64, 256))
    assert sequence.shape[0] == 1 and sequence.shape[2] == 512


def test_embed_returns_one_vector_per_image() -> None:
    embedding = VGGEncoder().embed(torch.zeros(3, 1, 32, 256))
    assert embedding.shape == (3, 512)


def test_resnet_encoder_satisfies_the_same_contract() -> None:
    encoder = ResNetSmallEncoder()
    sequence = encoder(torch.zeros(2, 1, 32, 256))
    assert sequence.shape[0] == 2 and sequence.shape[2] == encoder.feature_dim
    # The compact alternative must actually be smaller, which is the reason it
    # exists.
    assert encoder.num_parameters() < VGGEncoder().num_parameters()


def test_build_encoder_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown encoder type"):
        build_encoder({"type": "transformer-xl"})


def test_set_frozen_toggles_requires_grad() -> None:
    encoder = VGGEncoder()
    encoder.set_frozen(True)
    assert all(not p.requires_grad for p in encoder.parameters())
    encoder.set_frozen(False)
    assert all(p.requires_grad for p in encoder.parameters())


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------
def test_positional_encoding_preserves_shape() -> None:
    encoding = PositionalEncoding(64, max_positions=128)
    assert encoding(torch.zeros(2, 40, 64)).shape == (2, 40, 64)


def test_positional_encoding_rejects_an_over_long_sequence() -> None:
    encoding = PositionalEncoding(64, max_positions=16)
    with pytest.raises(ValueError, match="exceeds PositionalEncoding"):
        encoding(torch.zeros(1, 32, 64))


def test_attention_requires_divisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(10, heads=3)


def test_attention_is_length_preserving() -> None:
    encoder = AttentionEncoder(64, layers=2, heads=4, ff_dim=128)
    output, weights = encoder(torch.zeros(2, 30, 64), need_weights=True)

    # CTC requires one output frame per input frame; a length change here
    # would silently invalidate the loss.
    assert output.shape == (2, 30, 64)
    assert weights.shape == (2, 30, 30)


def test_attention_mask_excludes_padded_positions() -> None:
    encoder = AttentionEncoder(32, layers=1, heads=4, ff_dim=64).eval()
    lengths = torch.tensor([10, 30])

    with torch.no_grad():
        _, weights = encoder(torch.randn(2, 30, 32), lengths=lengths, need_weights=True)

    # The first sample may not attend past position 10.
    assert weights[0, :, 10:].abs().sum().item() == pytest.approx(0.0, abs=1e-5)
    assert weights[1, :, :].sum().item() > 0


# --------------------------------------------------------------------------
# recogniser
# --------------------------------------------------------------------------
def test_recogniser_output_shapes(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 32, 256), need_attention=True)

    assert output.logits.shape == (2, 63, small_charset.num_classes)
    assert output.log_probs.shape == output.logits.shape
    assert output.attention.shape == (2, 63, 63)
    assert output.input_lengths.tolist() == [63, 63]


def test_log_probs_are_normalised(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 32, 256))

    totals = output.log_probs.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    # CTCLoss is numerically unstable in half precision, so these must be fp32.
    assert output.log_probs.dtype == torch.float32


def test_ragged_input_lengths_take_the_packed_path(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes).eval()
    with torch.no_grad():
        output = model(torch.zeros(3, 1, 32, 256), input_lengths=torch.tensor([63, 30, 5]))

    assert output.log_probs.shape == (3, 63, small_charset.num_classes)
    assert output.input_lengths.tolist() == [63, 30, 5]


def test_zero_length_is_clamped_to_one(small_charset: Charset) -> None:
    # pack_padded_sequence raises on a zero length, and CTC needs a frame.
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 32, 256), input_lengths=torch.tensor([0, 70]))
    assert output.input_lengths.tolist() == [1, 63]


def test_wrong_channel_count_is_rejected(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    with pytest.raises(ValueError, match="input channel"):
        model(torch.zeros(1, 3, 32, 256))


def test_non_four_dimensional_input_is_rejected(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    with pytest.raises(ValueError, match="4-D"):
        model(torch.zeros(32, 256))


def test_parameter_breakdown_sums_to_the_total(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    breakdown = model.parameter_breakdown()
    parts = sum(v for k, v in breakdown.items() if k != "total")
    assert parts == breakdown["total"] == model.num_parameters()


def test_ctc_loss_is_finite_on_a_real_forward_pass(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes).eval()
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        output = model(torch.randn(2, 1, 32, 256))

    targets = torch.cat([
        torch.tensor(small_charset.encode("abc")),
        torch.tensor(small_charset.encode("de12")),
    ])
    loss = criterion(
        output.log_probs.permute(1, 0, 2),
        targets,
        output.input_lengths,
        torch.tensor([3, 4]),
    )
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# SimCLR and transfer
# --------------------------------------------------------------------------
def test_ntxent_loss_is_lower_for_well_separated_pairs() -> None:
    criterion = NTXentLoss(temperature=0.1)
    batch, dim = 8, 16

    # Matched views: each pair is identical and pairs differ from each other.
    base = torch.randn(batch, dim)
    easy = criterion(base, base.clone())
    # Random views: no pair structure at all.
    hard = criterion(torch.randn(batch, dim), torch.randn(batch, dim))

    assert easy.loss.item() < hard.loss.item()
    assert easy.accuracy == pytest.approx(1.0)


def test_ntxent_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        NTXentLoss()(torch.randn(1, 8), torch.randn(1, 8))


def test_ntxent_rejects_mismatched_views() -> None:
    with pytest.raises(ValueError, match="View shapes must match"):
        NTXentLoss()(torch.randn(4, 8), torch.randn(4, 16))


def test_simclr_projects_both_views() -> None:
    model = SimCLRModel(projection_dim=32).eval()
    with torch.no_grad():
        z_i, z_j = model(torch.zeros(4, 1, 32, 256), torch.zeros(4, 1, 32, 256))
    assert z_i.shape == z_j.shape == (4, 32)


def test_encoder_state_dict_excludes_the_projection_head() -> None:
    model = SimCLRModel(projection_dim=32)
    state = model.encoder_state_dict()
    assert state
    assert not any(key.startswith("projection") for key in state)


def test_simclr_weights_transfer_exactly(tmp_path, small_charset: Charset) -> None:
    """The core of the research claim: the transfer must be exact."""
    simclr = SimCLRModel(projection_dim=32)
    checkpoint = tmp_path / "encoder_best.pt"
    torch.save({"encoder_state_dict": simclr.encoder_state_dict()}, checkpoint)

    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    report = model.load_encoder_weights(checkpoint)

    assert report.loaded == report.total and report.fraction == 1.0
    assert report.missing == [] and report.unexpected == []

    source = simclr.encoder.state_dict()
    loaded = model.encoder.state_dict()
    assert set(source) == set(loaded)
    for key in source:
        assert torch.equal(source[key].cpu(), loaded[key].cpu()), f"{key} did not transfer"


def test_transfer_from_a_missing_file_raises(small_charset: Charset, tmp_path) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    with pytest.raises(FileNotFoundError):
        model.load_encoder_weights(tmp_path / "nope.pt")


def test_transfer_skips_mismatched_shapes(tmp_path, small_charset: Charset) -> None:
    """A checkpoint from a different encoder config must not load silently."""
    other = SimCLRModel(encoder_cfg={"type": "resnet_small"}, projection_dim=32)
    checkpoint = tmp_path / "other.pt"
    torch.save({"encoder_state_dict": other.encoder_state_dict()}, checkpoint)

    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    report = model.load_encoder_weights(checkpoint)
    # Nothing meaningful should transfer between different architectures.
    assert report.fraction < 1.0


def test_strict_transfer_raises_on_missing_tensors(tmp_path, small_charset: Charset) -> None:
    checkpoint = tmp_path / "partial.pt"
    torch.save({"encoder_state_dict": {}}, checkpoint)

    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    with pytest.raises(RuntimeError, match="Strict encoder transfer failed"):
        model.load_encoder_weights(checkpoint, strict=True)


def test_freeze_encoder_leaves_the_head_trainable(small_charset: Charset) -> None:
    model = CRNNAttentionOCR(num_classes=small_charset.num_classes)
    total = model.num_parameters(trainable_only=True)

    model.freeze_encoder(True)
    frozen = model.num_parameters(trainable_only=True)

    assert 0 < frozen < total
    # The recognition head must still train, or the frozen epochs do nothing.
    assert all(p.requires_grad for p in model.classifier.parameters())
