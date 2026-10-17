"""Building blocks shared by the GAN networks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm


def sn(module: nn.Module, enabled: bool = True) -> nn.Module:
    """Wrap a layer in spectral normalisation (largest singular value kept at 1).

    It bounds how fast the discriminator's output can change with its input
    (a Lipschitz constraint), the single most effective GAN stabiliser.
    """
    return spectral_norm(module) if enabled else module


def raw_weight(module: nn.Module) -> torch.Tensor:
    """The weight tensor to initialise in place (the original one under spectral norm)."""
    if parametrize.is_parametrized(module, "weight"):
        return module.parametrizations.weight.original
    return module.weight


def gan_widths(size: int, base: int, cap: int) -> dict[int, int]:
    """Channels per resolution: `base` at the top, doubling at each lower one, capped.

    gan_widths(32, 64, 512) -> {32: 64, 16: 128, 8: 256, 4: 512}
    """
    widths = {}
    r = size
    while r >= 4:
        widths[r] = min(cap, base * size // r)
        r //= 2
    return widths


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm whose scale and shift depend on the class label.

    With num_classes == 0 it is a plain BatchNorm2d. Each class gets its own
    (gamma, beta), initialised to (1, 0) so training starts like plain BN.
    """

    def __init__(self, channels: int, num_classes: int = 0) -> None:
        super().__init__()
        self.conditional = num_classes > 0
        self.bn = nn.BatchNorm2d(channels, affine=not self.conditional)
        if self.conditional:
            self.embed = nn.Embedding(num_classes, 2 * channels)
            nn.init.ones_(self.embed.weight[:, :channels])
            nn.init.zeros_(self.embed.weight[:, channels:])

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        out = self.bn(x)
        if not self.conditional:
            return out
        gamma, beta = self.embed(y).chunk(2, dim=1)
        return gamma[:, :, None, None] * out + beta[:, :, None, None]


class SelfAttention(nn.Module):
    """SAGAN self-attention: every pixel can look at every other pixel.

    Convolutions only see a small neighbourhood; attention lets the network
    keep far-apart parts consistent (two eyes of the same animal).
    The output is x + gamma * attention(x) with gamma starting at 0, so the
    block is the identity at first and the network learns how much to use it.
    """

    def __init__(self, channels: int, use_sn: bool = False) -> None:
        super().__init__()
        inner = max(1, channels // 8)
        self.query = sn(nn.Conv2d(channels, inner, 1, bias=False), use_sn)
        self.key = sn(nn.Conv2d(channels, inner, 1, bias=False), use_sn)
        self.value = sn(nn.Conv2d(channels, channels // 2, 1, bias=False), use_sn)
        self.out = sn(nn.Conv2d(channels // 2, channels, 1, bias=False), use_sn)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)  # [B, HW, c/8]
        k = self.key(x).flatten(2).transpose(1, 2)  # [B, HW, c/8]
        v = self.value(x).flatten(2).transpose(1, 2)  # [B, HW, c/2]
        attended = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        attended = attended.transpose(1, 2).reshape(b, c // 2, h, w)
        return x + self.gamma * self.out(attended)


__all__ = ["ConditionalBatchNorm2d", "SelfAttention", "gan_widths", "raw_weight", "sn"]
