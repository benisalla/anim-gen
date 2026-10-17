"""End-to-end: both task families train, log, sample, checkpoint and resume."""

import json

import pytest
import torch
from conftest import tiny_model_config, tiny_train_config

from anim_gen.config import DataConfig
from anim_gen.data import build_dataset
from anim_gen.runtime import resolve_device, resolve_precision
from anim_gen.tasks import build_task
from anim_gen.train import (
    LAST_CHECKPOINT,
    build_and_train,
    build_loader,
    endless,
    load_checkpoint,
    main,
    train,
)

DATA = DataConfig(dataset="shapes", image_size=16, conditional=True)


def run(tmp_path, arch, steps=6, callback=None, fid_fn=None, **train_overrides):
    model_config = tiny_model_config(arch, num_classes=3)
    train_config = tiny_train_config(tmp_path, steps=steps, **train_overrides)
    task = build_task(model_config, train_config, torch.device("cpu"))
    loader = build_loader(build_dataset(DATA), train_config, torch.device("cpu"))
    state = train(
        task, loader, DATA, model_config, train_config, ["circle", "square", "triangle"],
        callback=callback, fid_fn=fid_fn,
    )  # fmt: skip
    return task, state, train_config


@pytest.mark.parametrize(
    "arch,overrides",
    [
        ("dcgan", {"loss": "nonsaturating"}),
        ("resnet", {"loss": "hinge", "diffaug": "color,translation,cutout", "d_steps": 2}),
        ("resnet", {"loss": "wgan-gp"}),
        ("dcgan", {"loss": "lsgan", "r1_gamma": 1.0}),
        ("unet", {}),
    ],
)
def test_training_writes_everything(tmp_path, arch, overrides):
    events = []
    _task, state, cfg = run(tmp_path, arch, callback=events.append, **overrides)
    out = cfg.output_dir
    assert state.step == 6
    assert (out / LAST_CHECKPOINT).exists()
    assert (out / "real.png").exists()
    assert sorted(p.name for p in (out / "samples").iterdir()) == [
        "step_0000003.png", "step_0000006.png",
    ]  # fmt: skip
    logged = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    train_steps = [e["step"] for e in logged if e["kind"] == "train"]
    assert train_steps == [2, 4, 6]
    assert [e["step"] for e in events if e["kind"] == "train"] == train_steps
    key = "loss" if arch == "unet" else "g_loss"
    assert all(key in e for e in logged if e["kind"] == "train")
    if overrides.get("loss") == "wgan-gp":
        assert "gp" in logged[0]
    if overrides.get("r1_gamma"):
        assert "r1" in logged[0]


@pytest.mark.parametrize("arch", ["dcgan", "unet"])
def test_resume_continues_where_it_stopped(tmp_path, arch):
    _, _, cfg = run(tmp_path, arch, steps=3)
    ckpt = cfg.output_dir / LAST_CHECKPOINT
    _task, state, _ = run(tmp_path, arch, steps=6, resume=ckpt)
    assert state.step == 6
    logged = [
        json.loads(line) for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    # 2 and 3 (final log line of the first run) are kept, then the resumed run adds 4 and 6
    assert [e["step"] for e in logged if e["kind"] == "train"] == [2, 3, 4, 6]


def test_resume_restores_weights_exactly(tmp_path):
    task, _, cfg = run(tmp_path, "resnet", steps=3)
    fresh = build_task(task.model_config, cfg, torch.device("cpu"))
    state = load_checkpoint(cfg.output_dir / LAST_CHECKPOINT, fresh)
    assert state.step == 3
    for name, net in task.networks.items():
        for a, b in zip(net.state_dict().values(), fresh.networks[name].state_dict().values()):
            assert torch.equal(a, b)
    for name, opt in task.optimizers.items():
        assert (
            opt.state_dict()["state"].keys() == fresh.optimizers[name].state_dict()["state"].keys()
        )


def test_fresh_run_starts_a_new_metrics_file(tmp_path):
    run(tmp_path, "dcgan", steps=4)
    run(tmp_path, "dcgan", steps=2)  # same output dir, no resume
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines if '"train"' in line] == [2]


def test_fid_hook_and_best_checkpoint(tmp_path):
    scores = iter([5.0, 3.0, 4.0])
    _, state, cfg = run(tmp_path, "dcgan", steps=6, fid_every=2, fid_fn=lambda task: next(scores))
    assert state.best_fid == 3.0
    assert (cfg.output_dir / "best.pt").exists()
    best = torch.load(cfg.output_dir / "best.pt", weights_only=True)
    assert best["step"] == 4


def test_diffusion_learns_on_shapes(tmp_path):
    """Real (if short) learning signal: the denoising loss drops well below 1."""
    events = []
    run(
        tmp_path, "unet", steps=120, callback=events.append, log_every=20, sample_every=1000,
        ckpt_every=1000, lr=3e-3, batch_size=16,
    )  # fmt: skip
    losses = [e["loss"] for e in events if e["kind"] == "train"]
    assert losses[-1] < 0.5 * losses[0]


def test_endless_loader_crosses_epochs(tmp_path):
    cfg = tiny_train_config(tmp_path, batch_size=4)
    loader = build_loader(build_dataset(DataConfig(dataset="shapes", image_size=8, limit=8)), cfg,
                          torch.device("cpu"))  # fmt: skip
    batches = endless(loader)
    assert len([next(batches) for _ in range(5)]) == 5  # 2 batches per epoch


def test_loader_rejects_a_dataset_smaller_than_a_batch(tmp_path):
    cfg = tiny_train_config(tmp_path, batch_size=32)
    with pytest.raises(ValueError, match="fewer than one batch"):
        build_loader(build_dataset(DataConfig(dataset="shapes", image_size=8, limit=8)), cfg,
                     torch.device("cpu"))  # fmt: skip


def test_device_and_precision_selection():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_precision("auto", torch.device("cpu")) is None
    assert resolve_precision("bf16", torch.device("cpu")) is torch.bfloat16
    assert resolve_precision("fp32", torch.device("cuda")) is None


def test_bf16_autocast_trains_on_cpu(tmp_path):
    _, state, _ = run(tmp_path, "resnet", steps=2, precision="bf16")
    assert state.step == 2


def test_build_and_train_fills_model_shape_from_data(tmp_path):
    from anim_gen.config import ModelConfig

    task, _state = build_and_train(
        DataConfig(dataset="shapes", image_size=8, conditional=True, limit=16),
        ModelConfig(arch="dcgan", g_channels=4, d_channels=4, max_channels=8, z_dim=4),
        tiny_train_config(tmp_path, steps=2, batch_size=4),
    )
    assert task.model_config.image_size == 8
    assert task.model_config.num_classes == 3


def test_cli_runs_a_preset(tmp_path):
    main([
        "--preset", "shapes-dcgan", "--steps", "2", "--batch-size", "4", "--log-every", "1",
        "--limit", "8", "--output-dir", str(tmp_path / "cli"), "--device", "cpu",
    ])  # fmt: skip
    assert (tmp_path / "cli" / LAST_CHECKPOINT).exists()


def test_cli_lists_presets(capsys):
    main(["--list-presets"])
    assert "cifar10-resnet" in capsys.readouterr().out


def test_time_budget_stops_and_saves(tmp_path):
    _, state, cfg = run(tmp_path, "dcgan", steps=50, log_every=2, time_budget_hours=1e-9)
    assert state.step == 2  # first check happens at the first log step
    assert torch.load(cfg.output_dir / LAST_CHECKPOINT, weights_only=True)["step"] == 2
