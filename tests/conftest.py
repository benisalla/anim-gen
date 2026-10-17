"""Shared test helpers. Everything runs on CPU, offline, with tiny models."""

import dataclasses

import pytest
import torch

from anim_gen.config import ModelConfig, TrainConfig

# Tiny models are faster single-threaded (no thread start-up per op).
torch.set_num_threads(1)


def tiny_model_config(arch: str, num_classes: int = 0, **overrides) -> ModelConfig:
    """A model small enough to train for a few steps in well under a second."""
    base = {
        "arch": arch, "image_size": 16, "channels": 3, "num_classes": num_classes,
        "z_dim": 16, "g_channels": 8, "d_channels": 8, "max_channels": 32,
        "unet_channels": 8, "channel_mult": (1, 2), "num_res_blocks": 1, "attn_resolutions": (8,),
        "timesteps": 50, "dropout": 0.0,
    }  # fmt: skip
    base.update(overrides)
    return ModelConfig(**base)


def tiny_train_config(tmp_path, **overrides) -> TrainConfig:
    base = {
        "output_dir": tmp_path / "run", "steps": 6, "batch_size": 8, "num_workers": 0,
        "log_every": 2, "sample_every": 3, "ckpt_every": 3, "num_samples": 4, "sample_steps": 5,
        "warmup_steps": 1, "ema_decay": 0.9, "device": "cpu",
    }  # fmt: skip
    base.update(overrides)
    return TrainConfig(**base)


@pytest.fixture
def replace():
    return dataclasses.replace
