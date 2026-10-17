"""Fréchet Inception Distance (FID): how close generated images are to real ones.

Both image sets go through an ImageNet-trained Inception-v3; each set of
2048-d features is summarised by its mean mu and covariance S, and

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))

Lower is better; 0 means identical statistics. Note: we use torchvision's
Inception weights, not the TensorFlow ones of the original paper, so numbers
are close to but not exactly comparable with published FIDs. Compare runs of
this repo with each other, at the same number of samples.

    imagegen-fid --checkpoint checkpoints/run/last.pt --num-samples 10000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from imagegen.runtime import get_dist_info

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatures(nn.Module):
    """Images in [-1, 1] (any size, 1 or 3 channels) -> [B, 2048] pool features."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()  # keep the 2048-d pooled features, drop the classifier
        self.net = net.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None])

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = ((x + 1) / 2 - self.mean) / self.std
        return self.net(x)


class FeatureStats:
    """Running sum and sum of outer products -> mean and covariance (float64).

    Sums can be added across processes, which makes distributed FID exact.
    """

    def __init__(self, dim: int, device: torch.device | str = "cpu") -> None:
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def add(self, features: torch.Tensor) -> None:
        f = features.to(self.sum.device, torch.float64)
        self.n += f.shape[0]
        self.sum += f.sum(dim=0)
        self.outer += f.T @ f

    def all_reduce(self) -> FeatureStats:
        if get_dist_info().enabled:
            for t in (self.n, self.sum, self.outer):
                dist.all_reduce(t)
        return self

    def mean_cov(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n.clamp(min=2)
        mu = self.sum / n
        cov = (self.outer - n * torch.outer(mu, mu)) / (n - 1)  # unbiased, like numpy.cov
        return mu, cov


def frechet_distance(
    mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor
) -> float:
    """FID between two Gaussians.

    Tr((S1 S2)^(1/2)) = sum of the square roots of the eigenvalues of S1 S2
    (they are real and >= 0 in exact arithmetic; we keep the real part).
    """
    mu1, cov1, mu2, cov2 = (t.double().cpu() for t in (mu1, cov1, mu2, cov2))
    eigenvalues = torch.linalg.eigvals(cov1 @ cov2)
    tr_sqrt = eigenvalues.sqrt().real.sum()
    fid = (mu1 - mu2).pow(2).sum() + cov1.trace() + cov2.trace() - 2 * tr_sqrt
    return float(fid.clamp(min=0))


def _shard(n: int) -> range:
    """The slice of 0..n-1 this process handles."""
    info = get_dist_info()
    return range(info.rank, n, info.world_size)


@torch.no_grad()
def dataset_stats(
    dataset: Dataset,
    extractor: nn.Module,
    device: torch.device,
    max_images: int | None = None,
    batch_size: int = 100,
    cache: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature mean/cov of (up to `max_images` of) a dataset, cached to `cache`."""
    if cache is not None and Path(cache).exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        return saved["mu"], saved["cov"]
    n = min(len(dataset), max_images or len(dataset))
    loader = DataLoader(Subset(dataset, list(_shard(n))), batch_size=batch_size, num_workers=2)
    stats = FeatureStats(2048, device)
    for images, _ in loader:
        stats.add(extractor(images.to(device)))
    mu, cov = stats.all_reduce().mean_cov()
    mu, cov = mu.cpu(), cov.cpu()
    if cache is not None and get_dist_info().is_main:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": mu, "cov": cov, "n": n}, cache)
    return mu, cov


@torch.no_grad()
def generated_stats(
    task,
    extractor: nn.Module,
    num_samples: int,
    batch_size: int = 100,
    seed: int = 0,
    **generate_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature mean/cov of `num_samples` images from the task's EMA model."""
    info = get_dist_info()
    generator = torch.Generator().manual_seed(seed + info.rank)
    stats = FeatureStats(2048, task.device)
    todo = len(_shard(num_samples))
    while todo > 0:
        n = min(batch_size, todo)
        labels = task.random_labels(n, generator)
        noise = task.noise(n, generator)
        images = task.generate(noise, labels, generator=generator, **generate_kwargs)
        stats.add(extractor(images))
        todo -= n
    mu, cov = stats.all_reduce().mean_cov()
    return mu.cpu(), cov.cpu()


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: FID of a checkpoint against its training dataset."""
    from imagegen.data import build_dataset
    from imagegen.runtime import resolve_device
    from imagegen.sample import load_task

    p = argparse.ArgumentParser(description="FID of a trained checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--steps", type=int, default=100, help="DDIM steps (diffusion)")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--data-dir", type=Path, default=None, help="override the dataset location")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = resolve_device(args.device)
    task, data_config, _ = load_task(args.checkpoint, device)
    if args.data_dir is not None:
        data_config = dataclasses.replace(data_config, data_dir=args.data_dir)
    dataset = build_dataset(data_config, train=False)  # no flips: compare to real images
    extractor = InceptionFeatures().to(device)
    real = dataset_stats(dataset, extractor, device, max_images=args.num_samples)
    fake = generated_stats(
        task, extractor, args.num_samples, args.batch_size, args.seed,
        steps=args.steps, eta=args.eta, guidance_scale=args.guidance_scale,
    )  # fmt: skip
    fid = frechet_distance(*real, *fake)
    print(f"FID ({args.num_samples} samples): {fid:.2f}")


if __name__ == "__main__":
    main()


__all__ = [
    "FeatureStats",
    "InceptionFeatures",
    "dataset_stats",
    "frechet_distance",
    "generated_stats",
]
