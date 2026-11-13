"""Tests for the YAML config loader and the experiment configuration.

The last two tests are the important ones: they assert that the two arms of
the research experiment differ in exactly one key. If a future edit changes
the learning rate of one arm, the comparison becomes meaningless, and these
tests fail rather than the result quietly becoming wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.config import Config, ConfigError, deep_merge, load_config


def test_deep_merge_merges_nested_mappings() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}


def test_deep_merge_replaces_lists_wholesale() -> None:
    # Merging lists element-wise would produce nonsense for e.g. channel widths.
    assert deep_merge({"channels": [1, 2, 3]}, {"channels": [9]}) == {"channels": [9]}


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_config_supports_attribute_key_and_dotted_access(config: Config) -> None:
    assert config.train.epochs == config["train"]["epochs"] == config.get("train.epochs")


def test_get_returns_the_default_for_a_missing_path(config: Config) -> None:
    assert config.get("nothing.here.at.all", "fallback") == "fallback"


def test_missing_attribute_raises(config: Config) -> None:
    with pytest.raises(AttributeError, match="no key"):
        _ = config.definitely_not_a_key


def test_missing_file_is_reported_clearly() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("no_such_config.yaml")


def test_overrides_are_applied_last() -> None:
    config = load_config("default.yaml", overrides={"train": {"epochs": 999}})
    assert config.train.epochs == 999


def test_to_dict_is_a_deep_copy(config: Config) -> None:
    payload = config.to_dict()
    payload["train"]["epochs"] = -1
    assert config.train.epochs != -1


def test_experiment_configs_inherit_from_default() -> None:
    experiment = load_config("experiments/baseline_scratch.yaml")
    default = load_config("default.yaml")

    # `extends` must bring across keys the experiment never mentions.
    assert experiment.get("image.width") == default.get("image.width")
    assert experiment.get("charset.characters") == default.get("charset.characters")
    assert experiment.name == "baseline_scratch"


def test_full_training_smoke_config_does_not_require_simclr_checkpoint() -> None:
    config = load_config("experiments/crnn_full_v2_smoke.yaml")

    assert config.name == "_smoke_crnn"
    assert config.get("model.init_encoder_from") is None
    assert config.get("train.epochs") == 1
    assert config.get("data.train_manifest") == "data/processed/combined_iam/train.tsv"


def test_full_iam_training_config_does_not_require_unproduced_simclr_checkpoint() -> None:
    config = load_config("experiments/crnn_full_v2.yaml")

    assert config.get("model.init_encoder_from") is None


@pytest.mark.parametrize(
    ("config_name", "run_name"),
    [
        ("experiments/simclr_pretrain.yaml", "simclr_pretrain"),
        ("experiments/baseline_scratch.yaml", "baseline_scratch"),
        ("experiments/simclr_finetune.yaml", "simclr_finetune"),
    ],
)
def test_self_supervised_checkpoints_land_in_registry_run_directory(
    config_name: str, run_name: str
) -> None:
    config = load_config(config_name)
    checkpoint_dir = Path(str(config.get("train.checkpoint_dir")))
    run_directory = checkpoint_dir / run_name

    assert run_directory == Path("artifacts/checkpoints") / run_name


@pytest.mark.parametrize(
    ("control_name", "treatment_name"),
    [
        ("experiments/baseline_scratch.yaml", "experiments/simclr_finetune.yaml"),
        ("experiments/hw_baseline_scratch.yaml", "experiments/hw_simclr_finetune.yaml"),
    ],
)
def test_experiment_arms_differ_only_in_the_encoder_initialisation(
    control_name: str, treatment_name: str
) -> None:
    """The validity of the whole comparison rests on this."""
    control = load_config(control_name).to_dict()
    treatment = load_config(treatment_name).to_dict()

    # Fields that are expected to differ, and why.
    expected_differences = {
        "name",             # the run's identity
        "model",            # init_encoder_from / freeze_encoder_epochs
        "_config_path",     # bookkeeping
    }

    differing = {
        key
        for key in set(control) | set(treatment)
        if control.get(key) != treatment.get(key)
    }
    unexpected = differing - expected_differences

    # `train` differs only in checkpoint_dir, which is just an output path.
    if "train" in unexpected:
        control_train = dict(control["train"])
        treatment_train = dict(treatment["train"])
        control_train.pop("checkpoint_dir", None)
        treatment_train.pop("checkpoint_dir", None)
        assert control_train == treatment_train, (
            "The two experiment arms have different optimisation settings, so a CER "
            "difference between them could not be attributed to the pretraining."
        )
        unexpected.discard("train")

    assert not unexpected, f"Unexpected differences between the experiment arms: {sorted(unexpected)}"


@pytest.mark.parametrize(
    ("control_name", "treatment_name"),
    [
        ("experiments/baseline_scratch.yaml", "experiments/simclr_finetune.yaml"),
        ("experiments/hw_baseline_scratch.yaml", "experiments/hw_simclr_finetune.yaml"),
    ],
)
def test_experiment_arms_share_the_labeled_subset_and_seed(
    control_name: str, treatment_name: str
) -> None:
    control = load_config(control_name)
    treatment = load_config(treatment_name)

    # Same subset size and same seed => the seeded permutation in
    # OCRLineDataset selects the identical samples for both arms.
    assert control.get("data.max_train_samples") == treatment.get("data.max_train_samples")
    assert control.get("train.seed") == treatment.get("train.seed")
    assert control.get("data.train_manifest") == treatment.get("data.train_manifest")
    assert control.get("train.epochs") == treatment.get("train.epochs")

    # Only the treatment arm starts from a pretrained encoder.
    assert not control.get("model.init_encoder_from")
    assert treatment.get("model.init_encoder_from")
