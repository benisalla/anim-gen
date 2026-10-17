"""Generate images from a trained checkpoint.

    imagegen-sample --checkpoint checkpoints/run/last.pt --num 64 --out samples.png
    imagegen-sample --checkpoint ... --class dog --num 16 --out dogs.png
    imagegen-sample --checkpoint ... --num 1000 --folder generated/      # one PNG per image
    imagegen-sample --checkpoint ... --interpolate 8 --out walk.png      # latent walks

Sampling always uses the EMA weights stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import torch

from imagegen.config import DataConfig, ModelConfig, config_from_dict
from imagegen.images import save_grid, save_images
from imagegen.runtime import resolve_device, resolve_precision
from imagegen.tasks import Task, build_task

logger = logging.getLogger(__name__)


def load_task(
    path: Path, device: torch.device | str = "cpu", precision: str = "fp32"
) -> tuple[Task, DataConfig, list[str]]:
    """Rebuild the model from the configs saved in a checkpoint and load its weights.

    Returns (task, data_config, class_names). The task has no optimizers.
    """
    device = torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_config = config_from_dict(ModelConfig, checkpoint["model_config"])
    data_config = config_from_dict(DataConfig, checkpoint["data_config"])
    task = build_task(model_config, None, device, resolve_precision(precision, device))
    task.load_state_dict(checkpoint["task"])
    return task, data_config, list(checkpoint["class_names"])


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation: stays on the shell where Gaussian noise lives.

    A straight line between two noise vectors passes through low-norm points
    the model never saw in training; slerp keeps the norm about constant.
    """
    a_flat, b_flat = a.flatten(1), b.flatten(1)
    cos = (a_flat * b_flat).sum(1) / (a_flat.norm(dim=1) * b_flat.norm(dim=1))
    omega = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))[:, None]
    so = torch.sin(omega)
    mixed = (torch.sin((1 - t) * omega) / so) * a_flat + (torch.sin(t * omega) / so) * b_flat
    return mixed.view_as(a)


def resolve_labels(
    task: Task, class_names: list[str], name: str | None, n: int, generator: torch.Generator
) -> torch.Tensor | None:
    """Labels for n images: the named class, or random classes, or None."""
    if not task.conditional:
        if name is not None:
            raise ValueError("this model is unconditional: --class is not available")
        return None
    if name is None:
        return task.random_labels(n, generator)
    if name.isdigit():
        index = int(name)
    elif name in class_names:
        index = class_names.index(name)
    else:
        raise ValueError(f"unknown class {name!r}; choose from {class_names}")
    return torch.full((n,), index, device=task.device, dtype=torch.long)


@torch.no_grad()
def generate(
    task: Task,
    num: int,
    labels: torch.Tensor | None = None,
    batch_size: int = 64,
    generator: torch.Generator | None = None,
    **kwargs,
) -> torch.Tensor:
    """`num` images in [-1, 1], generated in batches, returned on the CPU."""
    out = []
    for start in range(0, num, batch_size):
        n = min(batch_size, num - start)
        batch_labels = None if labels is None else labels[start : start + n]
        noise = task.noise(n, generator)
        out.append(task.generate(noise, batch_labels, generator=generator, **kwargs).cpu())
    return torch.cat(out)


@torch.no_grad()
def interpolate(
    task: Task, rows: int, cols: int = 8, labels: torch.Tensor | None = None,
    generator: torch.Generator | None = None, **kwargs,
) -> torch.Tensor:  # fmt: skip
    """`rows` walks of `cols` images, each between two random noise inputs."""
    start, end = task.noise(rows, generator), task.noise(rows, generator)
    frames = [slerp(start, end, t) for t in torch.linspace(0, 1, cols).tolist()]
    noise = torch.stack(frames, dim=1).flatten(0, 1)  # row-major: [rows * cols, ...]
    if labels is not None:
        labels = labels[:rows].repeat_interleave(cols)
    return task.generate(noise, labels, generator=generator, **kwargs).cpu()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate images from a trained checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num", type=int, default=64, help="number of images")
    p.add_argument("--out", type=Path, default=Path("samples.png"), help="grid PNG")
    p.add_argument("--folder", type=Path, default=None, help="also save every image here")
    p.add_argument("--class", dest="class_name", default=None, help="class name or id")
    p.add_argument("--interpolate", type=int, metavar="ROWS", default=0, help="latent walks")
    p.add_argument("--steps", type=int, default=100, help="DDIM steps (diffusion only)")
    p.add_argument("--eta", type=float, default=0.0, help="0 = DDIM, 1 = DDPM-like noise")
    p.add_argument("--guidance-scale", type=float, default=1.0, help="classifier-free guidance")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="fp32", choices=["auto", "fp32", "bf16"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = resolve_device(args.device)
    task, _, class_names = load_task(args.checkpoint, device, args.precision)
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = {"steps": args.steps, "eta": args.eta, "guidance_scale": args.guidance_scale}

    if args.interpolate:
        labels = resolve_labels(task, class_names, args.class_name, args.interpolate, generator)
        images = interpolate(task, args.interpolate, 8, labels, generator, **kwargs)
        save_grid(images, args.out, nrow=8)
    else:
        labels = resolve_labels(task, class_names, args.class_name, args.num, generator)
        images = generate(task, args.num, labels, args.batch_size, generator, **kwargs)
        save_grid(images, args.out, nrow=max(1, int(math.sqrt(args.num))))
        if args.folder is not None:
            save_images(images, args.folder)
            logger.info("wrote %d images to %s", len(images), args.folder)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()


__all__ = ["generate", "interpolate", "load_task", "slerp"]
