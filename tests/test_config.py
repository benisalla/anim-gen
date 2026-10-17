"""Configs reject bad values early; presets only use real fields and build."""

from pathlib import Path

import pytest

from anim_gen.config import (
    PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    check_presets,
    config_from_dict,
    config_to_dict,
)
from anim_gen.train import configs_from_args, parse_args


@pytest.mark.parametrize(
    "build",
    [
        lambda: DataConfig(dataset="imagenet"),
        lambda: DataConfig(image_size=48),  # not a power of two
        lambda: ModelConfig(arch="vae"),
        lambda: ModelConfig(prediction="x0"),
        lambda: ModelConfig(arch="unet", image_size=16, channel_mult=(1, 2, 2, 2)),  # 2x2 bottom
        lambda: TrainConfig(loss="wasserstein"),
        lambda: TrainConfig(precision="fp16"),
        lambda: TrainConfig(steps=0),
        lambda: TrainConfig(ema_decay=1.0),
    ],
)
def test_invalid_configs_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_channels_follow_the_dataset():
    assert DataConfig(dataset="mnist").channels == 1
    assert DataConfig(dataset="cifar10").channels == 3


def test_family():
    assert ModelConfig(arch="dcgan").family == "gan"
    assert ModelConfig(arch="resnet").family == "gan"
    assert ModelConfig(arch="unet").family == "diffusion"


def test_presets_only_use_known_fields():
    check_presets()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_builds_valid_configs(name):
    data, model, _train = configs_from_args(parse_args(["--preset", name]))
    assert model.arch == PRESETS[name]["arch"]
    assert data.dataset == PRESETS[name]["dataset"]


def test_flags_override_the_preset():
    args = parse_args(
        ["--preset", "cifar10-diffusion", "--batch-size", "7", "--channel-mult", "1", "2"]
    )
    data, model, train = configs_from_args(args)
    assert train.batch_size == 7
    assert model.channel_mult == (1, 2)  # list from argparse -> tuple
    assert data.dataset == "cifar10"


def test_boolean_flags():
    data, _, _ = configs_from_args(parse_args(["--preset", "cifar10-dcgan", "--no-hflip"]))
    assert data.hflip is False


def test_config_round_trip():
    model = ModelConfig(arch="unet", channel_mult=(1, 2, 4), attn_resolutions=(8, 4))
    assert config_from_dict(ModelConfig, config_to_dict(model)) == model
    train = TrainConfig(output_dir=Path("x/y"), resume=Path("a.pt"))
    assert config_from_dict(TrainConfig, config_to_dict(train)) == train
