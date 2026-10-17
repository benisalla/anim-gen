"""DiffAugment (Zhao et al. 2020): differentiable augmentation for data-efficient GANs.

With a few thousand images (AFHQ has ~5k per class) the discriminator
memorises the training set and training collapses. Augmenting *both* real and
fake images, with operations gradients can flow through, stops that without
the generator learning to produce augmented images.

    policy "color"       random brightness, saturation, contrast
    policy "translation" random shift by up to 1/8 of the image, zero padded
    policy "cutout"      zero a random square of half the image size
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2) + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) + 0.5) + mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    b, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    shift_y = torch.randint(-sh, sh + 1, (b, 1, 1), device=x.device)
    shift_x = torch.randint(-sw, sw + 1, (b, 1, 1), device=x.device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    # +1 because of the 1-pixel zero padding below; out-of-range reads hit the padding.
    y = torch.clamp(grid_y[None] + shift_y + 1, 0, h + 1)
    xs = torch.clamp(grid_x[None] + shift_x + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    batch = torch.arange(b, device=x.device)[:, None, None]
    return padded.permute(0, 2, 3, 1)[batch, y, xs].permute(0, 3, 1, 2).contiguous()


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    # top-left corner of the cut square; it may stick out of the image
    top = torch.randint(0, h + 1 - ch % 2, (b, 1, 1), device=x.device) - ch // 2
    left = torch.randint(0, w + 1 - cw % 2, (b, 1, 1), device=x.device) - cw // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    cut = (grid_y >= top) & (grid_y < top + ch) & (grid_x >= left) & (grid_x < left + cw)
    return x * (~cut)[:, None].to(x.dtype)


POLICIES = {
    "color": (rand_brightness, rand_saturation, rand_contrast),
    "translation": (rand_translation,),
    "cutout": (rand_cutout,),
}


def parse_policy(policy: str) -> list[str]:
    """ "color,translation" -> ["color", "translation"]; validates the names."""
    names = [p.strip() for p in policy.split(",") if p.strip()]
    unknown = set(names) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown DiffAugment policy {sorted(unknown)}; use {sorted(POLICIES)}")
    return names


def diff_augment(x: torch.Tensor, policy: str) -> torch.Tensor:
    """Apply the policy to a batch in [-1, 1]. Empty policy = identity."""
    for name in parse_policy(policy):
        for fn in POLICIES[name]:
            x = fn(x)
    return x


__all__ = ["POLICIES", "diff_augment", "parse_policy"]
