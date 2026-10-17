"""Where and how training runs: seeding, device, precision, multi-GPU.

Multi-GPU / multi-node: launch the same command with torchrun, e.g.
    torchrun --nproc_per_node=4 -m imagegen.train --preset cifar10-resnet
torchrun starts one process per GPU and sets RANK / WORLD_SIZE / LOCAL_RANK.

We do NOT wrap the networks in DistributedDataParallel. A GAN step runs the
discriminator several times per backward pass, and the gradient penalties
(WGAN-GP, R1) need a double backward; DDP supports neither well. Instead every
process computes its own gradients and `sync_gradients` averages them with one
all-reduce per optimizer step (the approach StyleGAN2-ADA uses). Weights start
equal (`broadcast_module`) and stay equal because every process then applies
the same averaged update.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

try:  # numpy is optional; we only need it to seed its RNG
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def seed_everything(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def resolve_device(name: str = "auto") -> torch.device:
    """Turn "auto" | "cpu" | "cuda" | "mps" into a torch.device."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # under torchrun: cuda:LOCAL_RANK
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype, or None for plain fp32.

    "auto" picks bf16 only on GPUs that run it natively (Ampere and newer:
    A100, RTX 30xx/40xx, H100). Older GPUs such as the T4 / P100 on Kaggle and
    Colab stay in fp32. We do not offer fp16: GAN losses and gradient penalties
    overflow easily in fp16, and bf16 has the range of fp32.
    """
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
            return torch.bfloat16
        return None
    raise ValueError(f"unknown precision: {precision!r}")


def autocast(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager:
    """Mixed-precision context; does nothing when dtype is None."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


@dataclass(frozen=True)
class DistInfo:
    """Who am I in a multi-process run? Defaults describe a single process."""

    rank: int = 0  # global id of this process (0 .. world_size-1)
    world_size: int = 1  # total number of processes (= total GPUs)
    local_rank: int = 0  # id of this process on its own machine

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Only the main process logs, saves grids and writes checkpoints."""
        return self.rank == 0


def setup_distributed(device_name: str = "auto") -> DistInfo:
    """Join the process group if we were launched by torchrun; no-op otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = device_name in ("auto", "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    # NCCL is the fast backend for GPUs; gloo works on CPU (used by the tests).
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


def get_dist_info() -> DistInfo:
    """Current process-group state (single process if none was started)."""
    if not (dist.is_available() and dist.is_initialized()):
        return DistInfo()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


@contextmanager
def local_main_first(info: DistInfo) -> Iterator[None]:
    """Let local rank 0 run the block first (downloads); the others then reuse the files."""
    if info.enabled and info.local_rank != 0:
        dist.barrier()
    yield
    if info.enabled and info.local_rank == 0:
        dist.barrier()


def broadcast_module(module: nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every process."""
    if not get_dist_info().enabled:
        return
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=0)


def sync_gradients(parameters: Iterable[nn.Parameter]) -> None:
    """Average gradients over all processes with a single all-reduce."""
    info = get_dist_info()
    if not info.enabled:
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])  # one big message is much faster
    dist.all_reduce(flat)
    flat /= info.world_size
    offset = 0
    for g in grads:
        g.copy_(flat[offset : offset + g.numel()].view_as(g))
        offset += g.numel()


def all_reduce_mean(values: dict[str, float], device: torch.device) -> dict[str, float]:
    """Average logged scalars over processes (no-op on one process)."""
    if not get_dist_info().enabled or not values:
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    tensor /= get_dist_info().world_size
    return dict(zip(keys, tensor.tolist(), strict=True))


__all__ = [
    "DistInfo",
    "all_reduce_mean",
    "autocast",
    "broadcast_module",
    "get_dist_info",
    "local_main_first",
    "resolve_device",
    "resolve_precision",
    "seed_everything",
    "setup_distributed",
    "sync_gradients",
]
