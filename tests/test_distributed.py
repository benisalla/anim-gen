"""Multi-process training: 2 CPU processes (gloo) end with identical weights."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import tiny_model_config, tiny_train_config

from anim_gen.config import DataConfig
from anim_gen.data import build_dataset
from anim_gen.runtime import get_dist_info
from anim_gen.tasks import build_task
from anim_gen.train import build_loader, train


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world_size, port, arch, out_dir, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_num_threads(1)
    try:
        torch.manual_seed(rank)  # different initial weights per process on purpose
        loss = "wgan-gp" if arch != "unet" else "hinge"  # wgan-gp: double backward across ranks
        model_config = tiny_model_config(arch, num_classes=3)
        train_config = tiny_train_config(out_dir, steps=3, loss=loss, batch_size=4)
        data = DataConfig(dataset="shapes", image_size=16, conditional=True, limit=64)
        task = build_task(model_config, train_config, torch.device("cpu"))
        loader = build_loader(build_dataset(data), train_config, torch.device("cpu"))
        assert get_dist_info().world_size == world_size
        state = train(task, loader, data, model_config, train_config, ["a", "b", "c"])
        flat = {
            name: torch.cat([p.detach().flatten() for p in net.parameters()])
            for name, net in task.networks.items()
        }
        results[rank] = (state.step, flat)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("arch", ["resnet", "unet"])
def test_two_processes_stay_in_sync(tmp_path, arch):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(2, free_port(), arch, tmp_path, results), nprocs=2, join=True)
    (step0, nets0), (step1, nets1) = results[0], results[1]
    assert step0 == step1 == 3
    for name in nets0:
        assert torch.allclose(nets0[name], nets1[name], atol=1e-6), name
    # only rank 0 writes files
    assert (tmp_path / "run" / "last.pt").exists()
